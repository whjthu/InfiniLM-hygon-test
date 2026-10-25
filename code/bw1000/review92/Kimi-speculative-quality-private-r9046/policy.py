"""CPU reference for anchored speculative verification. No native SDK dependency."""
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class Decision:
    accepted:int
    emitted:tuple
    commit_inputs:tuple
    pending_anchor:int|None
    needs_replay:bool
    terminal:bool

def normalize(xs):
    xs=tuple(map(float,xs));s=sum(xs)
    if not xs or s<=0 or any(not math.isfinite(x)or x<0 for x in xs):raise ValueError('invalid probabilities')
    return tuple(x/s for x in xs)
def sample(xs,rng):
    u=rng.random();a=0
    for i,x in enumerate(normalize(xs)):
        a+=x
        if u<a:return i
    return len(xs)-1

def finish(anchor,proposals,a,correction,eos=(),remaining=None):
    proposals=tuple(proposals)
    if not 0<=a<=len(proposals):raise ValueError('invalid prefix')
    out=proposals[:a]+(correction,)
    if remaining is not None:
        if remaining<1:raise ValueError('remaining must be positive')
        out=out[:remaining]
    terminal=False
    for i,x in enumerate(out):
        if x in eos:out=out[:i+1];terminal=True;break
    terminal=terminal or remaining is not None and len(out)>=remaining
    # Correction/bonus is emitted now and processed as the NEXT cycle's anchor.
    inputs=(anchor,)+out[:-1]
    return Decision(a,out,inputs,None if terminal else out[-1],not terminal and len(inputs)<len(proposals)+1,terminal)
def greedy(anchor,proposals,target_top1,eos=(),remaining=None):
    proposals=tuple(proposals);target_top1=tuple(target_top1)
    if len(target_top1)!=len(proposals)+1:raise ValueError('expected K+1 target rows')
    a=0
    while a<len(proposals)and proposals[a]==target_top1[a]:a+=1
    return finish(anchor,proposals,a,target_top1[a],eos,remaining)
def rejection(anchor,proposals,p,q,rng,eos=(),remaining=None):
    """q must be the REAL conditional selector distribution used to sample y.
    Raw head q is not valid when a DFlash2 path selector changes proposal law.
    """
    proposals=tuple(proposals)
    if len(p)!=len(proposals)+1 or len(q)!=len(proposals):raise ValueError('row alignment')
    p=list(map(normalize,p));q=list(map(normalize,q))
    if any(len(v)!=len(p[0])for v in p+q):raise ValueError('vocabulary')
    for i,y in enumerate(proposals):
        if not 0<=y<len(p[i])or q[i][y]<=0:raise ValueError('proposal outside q support')
        if rng.random()*q[i][y]>=p[i][y]:
            residual=[max(x-z,0)for x,z in zip(p[i],q[i])]
            if sum(residual)<=0:raise ArithmeticError('zero residual after rejection')
            return finish(anchor,proposals,i,sample(residual,rng),eos,remaining)
    return finish(anchor,proposals,len(proposals),sample(p[-1],rng),eos,remaining)
class ReplayStateModel:
    """Transaction ownership check, not a numeric simulation of KDA."""
    def __init__(self,prefix=()):self.active=tuple(prefix);self.shadow=None
    def verify(self,inputs):self.shadow=self.active+tuple(inputs)
    def commit(self,d):
        if d.terminal:self.shadow=None;return
        wanted=self.active+d.commit_inputs
        if d.needs_replay:self.shadow=self.active+d.commit_inputs
        if self.shadow!=wanted:raise AssertionError('rejected state leaked')
        self.active=self.shadow;self.shadow=None
