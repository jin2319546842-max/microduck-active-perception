"""Planar navigation surrogate; no scripted routes or scan events in training."""
from collections import deque
from pathlib import Path
import json
import math
import numpy as np

CONFIG = json.loads((Path(__file__).parents[1]/'configs/curriculum.json').read_text())
TEMPLATES = {
    'empty': ([], (-.8, -1.4), (.8, 1.4)),
    'single': ([(.65, 0, 2.7, .1)], (0, -1), (.2, 1)),
    'enclosure': ([(-2,0,.09,4),(2,0,.09,4),(0,2,4,.09),(0,-2,4,.09),(0,-.65,.09,2.7)], (-.95,0), (.75,-.45)),
    'tee': ([(.4,1.5,2.8,.1),(0,.5,.1,2),(.2,-1.8,3.6,.1)], (-.9,-.18), (1.3,-.2)),
    'three_sides': ([(.2,1.05,1.8,.1),(-1.35,0,.1,1),(.2,-1.05,2.2,.1)], (0,-1.95), (.65,0)),
    'staggered': ([(.6,-.6,2.8,.1),(-.65,1.05,2.7,.1)], (0,-1.85), (.5,1.85)),
}

def sample_scene(stage, seed, split='train'):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    if not 0 <= stage < len(CONFIG['stages']):
        raise ValueError('Unknown stage')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), stage, ('train','validation','test').index(split)]))
    cfg = CONFIG['stages'][stage]
    family = str(rng.choice(cfg['families']))
    walls, start, goal = TEMPLATES[family]
    scale = rng.uniform(*cfg['scale_range'])
    mirror = int(rng.choice([-1, 1]))
    shift = rng.uniform(-.12,.12,2)
    def point(p):return (np.asarray(p)*[mirror,1]*scale+shift).tolist()
    transformed = [[*point((x,y)), w*scale,h*scale] for x,y,w,h in walls]
    return dict(family=family,stage=stage,seed=int(seed),split=split,mirror=mirror,
                walls=transformed,start=point(start),goal=point(goal),yaw_jitter=float(rng.uniform(-.15,.15)),
                noise_std_m=cfg['noise_std_m'],dropout=cfg['dropout'])

class NavigationEnv:
    """Actor sees narrow ToF + proprioception + goal bearing, never wall geometry."""
    obs_dim=135  # 64 range + 64 valid + goal xy (2), head sin/cos (2), previous action (3)
    def __init__(self, stage=0, split='train', history=None, max_steps=None, head_mode='learned'):
        self.stage=stage;self.split=split
        self.history_len=history or CONFIG['history_length']
        self.max_steps=max_steps or CONFIG['max_episode_steps']
        if head_mode not in ('learned','fixed','sweep'):raise ValueError('Unknown head mode')
        self.head_mode=head_mode

    def reset(self,seed=0):
        self.scene=sample_scene(self.stage,seed,self.split)
        self.rng=np.random.default_rng(np.random.SeedSequence([seed, 871, ('train','validation','test').index(self.split)]))
        self.pos=np.array(self.scene['start']);self.goal=np.array(self.scene['goal']);self.walls=np.array(self.scene['walls']).reshape(-1,4)
        delta=self.goal-self.pos;self.yaw=math.atan2(delta[1],delta[0])+self.scene['yaw_jitter']
        self.head=0.;self.previous_action=np.zeros(3);self.steps=0;self.travel=0.;self.head_travel=0.;self.ended=False
        obs=self.observation();self.history=deque([obs.copy() for _ in range(self.history_len)],maxlen=self.history_len)
        return np.stack(self.history)

    def ray(self,angle,elevation=0):
        best=CONFIG['range_m'];origin=np.array([*self.pos,.218]);direction=np.array([math.cos(angle)*math.cos(elevation),math.sin(angle)*math.cos(elevation),math.sin(elevation)])
        for x,y,w,h in self.walls:
            lo,hi=0.,best
            for p,d,mn,mx in zip(origin,direction,[x-w/2,y-h/2,0],[x+w/2,y+h/2,.44]):
                if abs(d)<1e-9:
                    if p<mn or p>mx:hi=-1;break
                else:
                    a,b=sorted(((mn-p)/d,(mx-p)/d));lo=max(lo,a);hi=min(hi,b)
            if lo<=hi:best=min(best,lo)
        return best

    def clearance(self,p):
        return min((math.hypot(max(abs(p[0]-x)-w/2,0),max(abs(p[1]-y)-h/2,0)) for x,y,w,h in self.walls),default=4.)

    def target(self):
        # Privileged supervision only: body-relative directional free distances.
        return np.array([self.ray(self.yaw+a)/4 for a in np.linspace(-math.pi,math.pi,12,endpoint=False)],np.float32)

    def observation(self):
        depth=np.array([[self.ray(self.yaw+self.head+math.radians((3.5-c)*5.5),math.radians((3.5-r)*5.5)) for c in range(8)] for r in range(8)])
        valid=(depth<4)&(self.rng.random((8,8))>=self.scene['dropout'])
        depth=np.where(valid,np.clip(depth+self.rng.normal(0,self.scene['noise_std_m'],(8,8)),0,4),4)
        dx,dy=self.goal-self.pos;c,s=math.cos(self.yaw),math.sin(self.yaw)
        proprio=[np.clip((c*dx+s*dy)/6,-1,1),np.clip((-s*dx+c*dy)/6,-1,1),math.sin(self.head),math.cos(self.head),*self.previous_action]
        return np.concatenate((depth.ravel()/4,valid.ravel(),proprio)).astype(np.float32)

    def step(self,action):
        if self.ended:raise RuntimeError('reset required after terminal/truncated step')
        action=np.asarray(action,np.float32)
        if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Expected three finite actions')
        action=np.clip(action,-1,1);dt=CONFIG['control_dt'];before=np.linalg.norm(self.goal-self.pos)
        self.steps+=1;old_head=self.head
        self.yaw+=float(action[1])*1.2*dt
        if self.head_mode=='learned':self.head=np.clip(self.head+float(action[2])*1.8*dt,-1.4,1.4)
        elif self.head_mode=='fixed':self.head=0.
        else:self.head=1.25*math.sin(self.steps*dt*1.2)
        # High-level planar controller; this does not simulate leg dynamics.
        speed=float(action[0])*.3
        dest=self.pos+speed*dt*np.array([math.cos(self.yaw),math.sin(self.yaw)])
        collision=any(self.clearance(p)<CONFIG['robot_radius_m'] for p in np.linspace(self.pos,dest,5))
        outside=bool(np.max(np.abs(dest))>3.5)
        if not collision and not outside:self.travel+=float(np.linalg.norm(dest-self.pos));self.pos=dest
        self.head_travel+=abs(self.head-old_head)
        success=bool(np.linalg.norm(self.goal-self.pos)<.18 and not collision and not outside)
        reward=3*(before-np.linalg.norm(self.goal-self.pos))-.01-.003*float(np.square(action-self.previous_action).sum())
        reward+=10*success-10*(collision or outside)
        self.previous_action=action.copy();self.history.append(self.observation())
        terminated=bool(success or collision or outside);truncated=bool(self.steps>=self.max_steps and not terminated);self.ended=terminated or truncated
        info=dict(success=success,collision=bool(collision),out_of_bounds=outside,timeout=truncated,path_length=self.travel,head_travel_rad=self.head_travel,stage=self.stage)
        return np.stack(self.history),float(reward),terminated,truncated,info
