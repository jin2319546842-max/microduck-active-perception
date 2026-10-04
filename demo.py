"""Screenshot reproduction: official Microduck geometry, scripted kinematics, MuJoCo rays/rendering."""
from pathlib import Path
import argparse, io, json, math, queue, threading, time
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from PIL import Image
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).parent
MODEL = ROOT/'upstream/src/mjlab_microduck/robot/microduck'
OUT = ROOT/'output'
OUT.mkdir(exist_ok=True)
# Wall entries: center x/y, full length x/y. Routes: x/y, optional scan.
SCENES = [
 dict(name='围墙隔板 · 左侧绕行', walls=[(-2,0,.09,4),(2,0,.09,4),(0,2,4,.09),(0,-2,4,.09),(0,-.65,.09,2.7)], start=(-.95,0), goal=(.75,-.45), route=[(-.48,0,'scan'),(-.48,1.22),(.7,1.22),(.75,-.45)]),
 dict(name='T 形墙 · 右侧开口', walls=[(.4,1.5,2.8,.1),(0,.5,.1,2),(.2,-1.8,3.6,.1)], start=(-.9,-.18), goal=(1.3,-.2), route=[(-.48,-.18,'scan'),(-.48,-1.1),(.75,-1.1),(1.3,-.2)]),
 dict(name='三面障碍 · 从开口进入', walls=[(.2,1.05,1.8,.1),(-1.35,0,.1,1),(.2,-1.05,2.2,.1)], start=(0,-1.95), goal=(.65,0), route=[(0,-1.48,'scan'),(-1.1,-1.48),(-1.1,-.72),(-.55,-.72),(.65,0)]),
 dict(name='交错墙 · 连续左右观察', walls=[(.6,-.6,2.8,.1),(-.65,1.05,2.7,.1)], start=(0,-1.85), goal=(.5,1.85), route=[(0,-1.1,'scan'),(-1.28,-1.1),(-1.28,.25),(0,.25),(0,.56,'scan'),(1.18,.56),(1.18,1.85),(.5,1.85)])
]

def build_model(index):
    spec=ET.parse(MODEL/'robot_walk.xml').getroot()
    spec.find('compiler').set('meshdir',str(MODEL/'assets'))
    asset=spec.find('asset')
    ET.SubElement(asset,'texture',name='checker',type='2d',builtin='checker',mark='edge',rgb1='.12 .2 .28',rgb2='.065 .13 .20',markrgb='.22 .32 .39',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='checker',texuniform='true',texrepeat='5 5',reflectance='.03')
    visual=ET.SubElement(spec,'visual')
    ET.SubElement(visual,'global',offwidth='1200',offheight='800')
    ET.SubElement(visual,'headlight',diffuse='.7 .7 .7',ambient='.35 .35 .35',specular='0 0 0')
    ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
    world=spec.find('worldbody')
    ET.SubElement(world,'light',pos='-2 -3 7',dir='.2 .3 -1',directional='true',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='floor',type='plane',size='20 20 .1',material='floor_mat')
    for k,(x,y,w,h) in enumerate(SCENES[index]['walls']):
        ET.SubElement(world,'geom',name=f'wall_{k}',type='box',pos=f'{x} {y} .22',size=f'{w/2} {h/2} .22',rgba='1 .29 .025 1',group='4')
    x,y=SCENES[index]['goal']
    ET.SubElement(world,'geom',name='goal',type='cylinder',pos=f'{x} {y} .007',size='.16 .006',rgba='.05 1 .09 1',contype='0',conaffinity='0',group='5')
    xml=ET.tostring(spec,encoding='unicode')
    (OUT/f'scene_{index+1}.xml').write_text(xml,encoding='utf8')
    return mj.MjModel.from_xml_string(xml)

class Demo:
    def __init__(self,index=0):
        self.index=index;self.scene=SCENES[index];self.model=build_model(index);self.data=mj.MjData(self.model)
        self.home=np.array([0,-.087266,-.457924,-.004940,.452984,.349066,.349066,0,0,0,.087266,.457924,.004940,-.452984])
        self.joints={self.model.joint(i).name:int(self.model.jnt_qposadr[i]) for i in range(1,self.model.njnt)}
        self.pos=np.array(self.scene['start'],float);first=self.scene['route'][0];self.yaw=math.atan2(first[1]-self.pos[1],first[0]-self.pos[0])
        self.head=0.;self.elapsed=0.;self.timer=0.;self.i=0;self.phase='接近障碍';self.left=None;self.right=None;self.choice='';self.speed=0.;self.scans=[];self.min_clearance=999.
        self.walk_blend=0.;self.stride=0.;self.turn_rate=0.
        self.camera=mj.MjvCamera();self.camera.lookat[:]=[0,0,0];self.camera.distance=6.25;self.camera.azimuth=90;self.camera.elevation=-62
        self.apply_pose()
        # Mount the ideal ToF origin to the actual moving head. Its forward
        # direction uses the calibrated local vector of the neutral head.
        self.sensor_body=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,'yaw_roll_motion')
        self.sensor_forward=self.data.xmat[self.sensor_body].reshape(3,3).T @ np.array([math.cos(self.yaw),math.sin(self.yaw),0.])

    def apply_pose(self):
        t=self.elapsed;b=self.walk_blend;stride=self.stride
        # Bounded, continuous irregularity: no independent per-frame randomness.
        sway=b*(.005*math.sin(stride)+.0015*math.sin(t*7.3))
        roll=b*(.028*math.sin(stride)+.006*math.sin(t*8.7))+.003*math.sin(t*2.1)
        pitch=b*(.012*math.sin(2*stride+.4)+.005*math.sin(t*6.3))
        body_yaw=self.yaw+b*.012*math.sin(stride+.6)
        self.data.qpos[:]=0;self.data.qpos[:2]=self.pos;self.data.qpos[2]=.12
        self.data.qpos[:2]+=sway*np.array([-math.sin(self.yaw),math.cos(self.yaw)])
        cr,sr=math.cos(roll/2),math.sin(roll/2);cp,sp=math.cos(pitch/2),math.sin(pitch/2);cy,sy=math.cos(body_yaw/2),math.sin(body_yaw/2)
        self.data.qpos[3:7]=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy];self.data.qpos[7:]=self.home
        if b>0:
            gait=math.sin(stride)*.16*b
            # Joint order is read from the model below; gait is presentation-only.
            for name in self.joints:
                if 'hip_pitch' in name:self.data.qpos[self.joints[name]]+=gait
                if 'knee' in name:self.data.qpos[self.joints[name]]+=.06*b*math.sin(stride)
                if 'hip_roll' in name:self.data.qpos[self.joints[name]]-=roll*.4
            self.data.qpos[2]+=.0045*b*(.5-.5*math.cos(2*stride))
        tremor=.006*math.sin(t*8.1)+.004*math.sin(t*13.7+.8)
        self.data.qpos[self.joints['head_yaw']]=self.head+tremor-b*.009*math.sin(stride+.3)
        self.data.qpos[self.joints['head_pitch']]+=.006*math.sin(t*3.3)+b*.013*math.sin(stride-.4)
        self.data.qpos[self.joints['head_roll']]-=roll*.55
        mj.mj_forward(self.model,self.data)

    def ray(self,angle,elevation=0):
        origin=self.data.xpos[self.sensor_body].copy()
        forward=self.data.xmat[self.sensor_body].reshape(3,3) @ self.sensor_forward
        angle=math.atan2(forward[1],forward[0])+(angle-self.yaw-self.head)
        direction=np.array([math.cos(angle)*math.cos(elevation),math.sin(angle)*math.cos(elevation),math.sin(elevation)])
        gid=np.array([-1],np.int32)
        distance=mj.mj_ray(self.model,self.data,origin,direction,np.array([0,0,0,0,1,0],np.uint8),1,-1,gid)
        return float(distance) if 0<=distance<4 else 4.

    def observe(self):return min(self.ray(self.yaw+self.head+a) for a in [-.07,0,.07])

    def tick(self,dt):
        previous_speed=self.speed
        self.elapsed+=dt;self.speed=0
        if self.phase=='到达目标':return
        if self.phase in ['向左观察','向右观察','头部回中']:
            self.timer+=dt;t=self.timer;a=math.radians(75)
            ease=lambda u:u*u*(3-2*u)
            settle=lambda u:.025*math.exp(-5*u)*math.sin(15*u)
            if t<1.1:self.head=a*ease(t/1.1);self.phase='向左观察'
            elif t<2:self.head=a+settle(t-1.1);self.left=self.observe()
            elif t<3.8:self.head=a-2*a*ease((t-2)/1.8);self.phase='向右观察'
            elif t<4.7:self.head=-a-settle(t-3.8);self.right=self.observe()
            elif t<5.6:self.head=-a*(1-ease((t-4.7)/.9));self.phase='头部回中'
            else:
                self.head=0.;self.phase='比较通行空间';self.timer=0
                self.choice='左侧' if self.left>self.right else '右侧'
                self.scans.append(dict(left=self.left,right=self.right,choice=self.choice))
        elif self.phase=='比较通行空间':
            self.timer+=dt
            if self.timer>1.2:self.phase='绕行';self.i+=1
        else:
            if self.i>=len(self.scene['route']):self.phase='到达目标';return
            target=self.scene['route'][self.i];delta=np.array(target[:2])-self.pos;dist=np.linalg.norm(delta)
            if dist<.003:
                if len(target)>2:self.phase='向左观察';self.timer=0;self.left=self.right=None;self.choice=''
                else:self.i+=1
            else:
                desired=math.atan2(delta[1],delta[0]);turn=math.atan2(math.sin(desired-self.yaw),math.cos(desired-self.yaw))
                desired_rate=max(-1.25,min(1.25,turn*4))
                self.turn_rate+=(desired_rate-self.turn_rate)*(1-math.exp(-dt/ .13))
                self.yaw+=max(-abs(turn),min(abs(turn),self.turn_rate*dt))
                if abs(turn)<.04:
                    target_speed=min(.23,math.sqrt(2*.38*dist))*(1+.055*math.sin(self.stride*2)+.025*math.sin(self.elapsed*3.7))
                    self.speed=min(target_speed,previous_speed+.42*dt)
                    self.pos+=delta/dist*min(dist,dt*self.speed)
        self.walk_blend+=(min(1,self.speed/.18)-self.walk_blend)*(1-math.exp(-dt/.16))
        self.stride+=dt*(11.8+.7*math.sin(self.elapsed*1.9))*self.walk_blend
        self.apply_pose()
        for x,y,w,h in self.scene['walls']:
            gap=math.hypot(max(abs(self.data.qpos[0]-x)-w/2,0),max(abs(self.data.qpos[1]-y)-h/2,0))
            self.min_clearance=min(self.min_clearance,gap)

    def state(self):
        depth=[[self.ray(self.yaw+self.head+math.radians((3.5-c)*5.5),math.radians((3.5-r)*5.5)) for c in range(8)] for r in range(8)]
        return dict(scene=self.index,name=self.scene['name'],phase=self.phase,elapsed=round(self.elapsed,2),head=round(math.degrees(self.data.qpos[self.joints['head_yaw']]),1),speed=round(self.speed,3),goal_distance=round(float(np.linalg.norm(self.pos-np.array(self.scene['goal']))),3),left=self.left,right=self.right,choice=self.choice,depth=depth,scans=self.scans)

def render(renderer,demo,follow=False):
    if follow:
        demo.camera.lookat[:]=[*demo.pos,.08];demo.camera.distance=1.6;demo.camera.elevation=-35
    else:demo.camera.lookat[:]=[0,0,0];demo.camera.distance=6.25;demo.camera.elevation=-62
    option=mj.MjvOption();option.geomgroup[4]=1;option.geomgroup[5]=1;option.geomgroup[3]=0
    renderer.update_scene(demo.data,camera=demo.camera,scene_option=option)
    return renderer.render()

app=Flask(__name__);commands=queue.Queue();shared={'jpeg':b'','state':{},'running':False};lock=threading.Lock()
@app.get('/')
def home():return send_from_directory(ROOT,'index.html')
@app.get('/state')
def state():
    with lock:return jsonify(shared['state'])
@app.post('/command')
def command():
    data=request.get_json();commands.put(data);return jsonify(ok=True)
@app.get('/stream')
def stream():
    def frames():
        while True:
            with lock:jpg=shared['jpeg']
            if jpg:yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg+b'\r\n'
            time.sleep(.05)
    return Response(frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

def serve():
    demo=Demo();renderer=mj.Renderer(demo.model,height=700,width=1000);running=False;follow=False;rate=1;prev=time.perf_counter()
    threading.Thread(target=lambda:app.run(host='127.0.0.1',port=8765,threaded=True,use_reloader=False),daemon=True).start()
    while True:
        while not commands.empty():
            cmd=commands.get();action=cmd.get('action')
            if action in ['scene','restart']:
                idx=int(cmd.get('scene',demo.index));renderer.close();demo=Demo(max(0,min(3,idx)));renderer=mj.Renderer(demo.model,height=700,width=1000)
                running=action=='restart'
            elif action=='play':running=not running
            elif action=='view':follow=bool(cmd.get('follow'))
            elif action=='speed':rate=max(.5,min(3,float(cmd.get('rate',1))))
        now=time.perf_counter();dt=min(.1,now-prev);prev=now
        if running:
            demo.tick(dt*rate)
            if demo.phase=='到达目标':running=False
        pixels=render(renderer,demo,follow);buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='JPEG',quality=88)
        result=demo.state();result['running']=running
        with lock:shared.update(jpeg=buf.getvalue(),state=result)
        time.sleep(.015)

def verify():
    result=[]
    for index in range(4):
        demo=Demo(index)
        with mj.Renderer(demo.model,height=700,width=1000) as renderer:
            Image.fromarray(render(renderer,demo)).save(OUT/f'scene_{index+1}.png')
            for k in range(12000):
                demo.tick(.025)
                if demo.phase=='到达目标':break
        row=demo.state();row['minimum_center_wall_clearance']=demo.min_clearance;result.append(row)
        assert demo.phase=='到达目标',row
        assert demo.min_clearance>.12,row
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([{k:r[k] for k in ['name','phase','scans','minimum_center_wall_clearance']} for r in result],ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    verify() if args.verify else serve()
