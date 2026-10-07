import argparse,json
from pathlib import Path
import torch
from .model import Agent
from .train import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--output',default='runs/evaluation.json')
    p.add_argument('--max-steps',type=int,default=None);p.add_argument('--head-mode',choices=['learned','fixed','sweep'],default='learned')
    args=p.parse_args();torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True);agent=Agent();agent.load_state_dict(checkpoint['model'])
    result=dict(checkpoint_steps=checkpoint['steps'],split=args.split,episodes=args.episodes,head_mode=args.head_mode,
                stages=[evaluate(agent,s,args.episodes,args.split,args.max_steps,args.head_mode) for s in range(5)])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
