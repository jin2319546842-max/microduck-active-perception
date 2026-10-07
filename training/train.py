"""APPLE-inspired SAC prototype, not the authors' exact implementation."""
import argparse, copy, json, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .env import CONFIG, NavigationEnv
from .model import Agent

def update(agent,target,optimizer,batch,device,perception_weight=1.):
    h,a,r,hn,terminal,label=[torch.as_tensor(np.stack(v),dtype=torch.float32,device=device) for v in zip(*batch)]
    z=agent.encode(h)
    perception=(agent.prediction(z)-label).square().mean(-1,keepdim=True)
    # Recompute the prediction penalty using today's model, not an old stored reward.
    with torch.no_grad():
        next_action,next_logp=agent.act(agent.encode(hn))
        tq1,tq2=target.qs(target.encode(hn),next_action)
        bellman=r[:,None]-perception_weight*perception.detach()+.99*(1-terminal[:,None])*(torch.minimum(tq1,tq2)-.1*next_logp)
    q1,q2=agent.qs(z,a);critic=F.mse_loss(q1,bellman)+F.mse_loss(q2,bellman)
    # Freeze critic parameters for actor gradients while preserving critic loss graph.
    qparams=list(agent.q1.parameters())+list(agent.q2.parameters())
    for p in qparams:p.requires_grad_(False)
    chosen,logp=agent.act(z);aq1,aq2=agent.qs(z.detach(),chosen)
    actor=(.1*logp-torch.minimum(aq1,aq2)).mean()
    for p in qparams:p.requires_grad_(True)
    loss=critic+actor+perception_weight*perception.mean()
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.parameters(),10);optimizer.step()
    with torch.no_grad():
        for p,tp in zip(agent.parameters(),target.parameters()):tp.lerp_(p,.005)
    result=dict(critic=float(critic.detach()),actor=float(actor.detach()),perception=float(perception.mean().detach()))
    if not all(np.isfinite(list(result.values()))):raise RuntimeError('Nonfinite loss')
    return result

def evaluate(agent,stage,episodes=20,split='validation',max_steps=None,head_mode='learned'):
    device=next(agent.parameters()).device;rows=[];agent.eval()
    for seed in range(episodes):
        env=NavigationEnv(stage,split,max_steps=max_steps,head_mode=head_mode);h=env.reset(seed);errors=[]
        for _ in range(env.max_steps):
            with torch.no_grad():
                z=agent.encode(torch.tensor(h[None],device=device));action,_=agent.act(z,True)
                errors.append(float((agent.prediction(z)-torch.tensor(env.target()[None],device=device)).square().mean()))
            h,_,done,trunc,info=env.step(action[0].cpu().numpy())
            if done or trunc:break
        rows.append({**info,'perception_mse':float(np.mean(errors))})
    agent.train()
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}

def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    device=torch.device(args.device);agent=Agent().to(device);target=copy.deepcopy(agent);optim=torch.optim.Adam(agent.parameters(),lr=3e-4)
    replay=[];capacity=5000;stage=0;stage_start=0;passes=0;episode=0;updates=0
    env=NavigationEnv(stage);h=env.reset(args.seed);records=[];last_loss={}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        if step<=args.warmup:action=np.random.uniform(-1,1,3).astype(np.float32)
        else:
            with torch.no_grad():action=agent.act(agent.encode(torch.tensor(h[None],device=device)))[0][0].cpu().numpy()
        label=env.target();hn,reward,done,trunc,info=env.step(action)
        item=(h.copy(),action.copy(),reward,hn.copy(),float(done),label.copy())
        if len(replay)<capacity:replay.append(item)
        else:replay[(step-1)%capacity]=item
        h=hn
        if step>=args.warmup and len(replay)>=args.batch:
            last_loss=update(agent,target,optim,random.sample(replay,args.batch),device,args.perception_weight);updates+=1
        if done or trunc:
            records.append(dict(step=step,episode=episode,**info));episode+=1
            sampled=stage if stage==0 or random.random()>=CONFIG['previous_stage_probability'] else random.randrange(stage)
            env=NavigationEnv(sampled);h=env.reset(args.seed+episode)
        if step%args.eval_every==0:
            metrics=evaluate(agent,stage,args.eval_episodes)
            gate=CONFIG['promotion'];passed=metrics['success']>=gate['success_rate'] and metrics['collision']<=gate['max_collision_rate']
            passes=passes+1 if passed and step-stage_start>=gate['min_stage_steps'] else 0
            records.append(dict(step=step,validation=metrics,stage=stage,consecutive_passes=passes))
            if passes>=gate['consecutive_passes'] and stage<4:
                stage+=1;stage_start=step;passes=0
                # Explicit curriculum reset; do not count the interrupted episode as success.
                records.append(dict(step=step,event='curriculum_advance',new_stage=stage))
                episode+=1;env=NavigationEnv(stage);h=env.reset(args.seed+episode)
            print(json.dumps(dict(step=step,stage=stage,loss=last_loss,validation=metrics)),flush=True)
    checkpoint=dict(model=agent.state_dict(),steps=args.steps,updates=updates,stage=stage,seed=args.seed,
                    config=CONFIG,args=vars(args),status='unvalidated_prototype',optimizer=optim.state_dict(),target=target.state_dict())
    torch.save(checkpoint,out/'checkpoint.pt')
    (out/'metrics.json').write_text(json.dumps(dict(steps=args.steps,updates=updates,stage=stage,last_loss=last_loss,records=records),indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(out),steps=args.steps,updates=updates,stage=stage,last_loss=last_loss)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=200000);p.add_argument('--warmup',type=int,default=1000)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='runs/apple-sac');p.add_argument('--eval-every',type=int,default=5000);p.add_argument('--eval-episodes',type=int,default=20)
    p.add_argument('--perception-weight',type=float,default=1.)
    args=p.parse_args()
    if min(args.steps,args.batch,args.eval_every,args.eval_episodes)<1 or args.warmup<0 or args.perception_weight<0:p.error('Invalid numeric argument')
    train(args)
