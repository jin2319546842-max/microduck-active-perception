import torch
from torch import nn
from torch.distributions import Normal

class Agent(nn.Module):
    def __init__(self, obs_dim=135, history=8, width=64):
        super().__init__();self.history=history
        self.project=nn.Linear(obs_dim,width);self.position=nn.Parameter(torch.zeros(1,history,width))
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2)
        self.actor=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Linear(64,6))
        self.prediction=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Linear(64,12),nn.Sigmoid())
        def q():return nn.Sequential(nn.Linear(width+3,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.q1=q();self.q2=q()

    def encode(self,history):
        n=history.shape[1];mask=torch.triu(torch.ones(n,n,device=history.device,dtype=torch.bool),1)
        return self.encoder(self.project(history)+self.position[:,:n],mask=mask)[:,-1]

    def act(self,z,deterministic=False):
        mean,log_std=self.actor(z).chunk(2,-1);dist=Normal(mean,log_std.clamp(-5,1).exp())
        raw=mean if deterministic else dist.rsample();a=torch.tanh(raw)
        logp=(dist.log_prob(raw)-torch.log(1-a.square()+1e-6)).sum(-1,keepdim=True)
        return a,logp

    def qs(self,z,a):
        x=torch.cat((z,a),-1);return self.q1(x),self.q2(x)
