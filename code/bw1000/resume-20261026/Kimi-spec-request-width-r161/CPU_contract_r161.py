"""CPU scheduling evidence only; no model, Torch, GPU or performance claim."""
import json
from types import SimpleNamespace
from request_width_r161 import request_width,effective_widths

def row(accepted,proposed=7,terminal=False):
 return dict(accepted_proposals=accepted,proposed=proposed,terminal=terminal)
def req(logical,history):
 return SimpleNamespace(logical=logical,cycles=list(history),cap=512,tokens=[1]*20,past=300,done=False)

checks=[]
assert request_width([],8,8,enabled=True)==8
assert request_width([row(0)]*3,8,8,enabled=True)==8
checks.append('four prior full-width nonterminal observations required')
low=[row(0),row(1),row(2),row(1)]
assert request_width(low,8,8,enabled=True)==4
assert request_width(low,1,8,enabled=True)==8
assert request_width(low,2,8,enabled=True)==8
assert request_width(low,3,8,enabled=True)==8
checks.append('C1 C2 C3 preserved W8; no repeat of uniform C1 shrink negative')
for n in range(3):assert request_width(low+[row(1,3)]*n,8,8,enabled=True)==4
assert request_width(low+[row(1,3)]*3,8,8,enabled=True)==8
checks.append('narrowed rows probe W8 every fourth cycle; no censored acceptance oracle')
assert request_width([row(7),row(5),row(6),row(7)],8,8,enabled=True)==8
assert request_width(low+[row(7)],8,8,enabled=True)==4
assert request_width(low+[row(7),row(7)],8,8,enabled=True)==8
checks.append('two high-acceptance probes restore W8 without current-cycle feedback')
assert request_width(low,8,8,enabled=False)==8
assert request_width(low,16,4,enabled=True)==4
assert request_width(low,32,2,enabled=True)==2
checks.append('defaultOFF and existing C16/C32 width budget unchanged')
group=[req(i,low if i<4 else[row(7)]*4)for i in range(1,9)]
assert list(effective_widths(group,8,enabled=True).values())==[4]*3+[8]*5
assert sum(effective_widths(group,8,enabled=True).values())==52
assert sum(effective_widths(group,8,enabled=False).values())==64
checks.append('ragged C8 T52 vs baseline64; only prefix width changes, not arithmetic or commitment')
group[0].cap=21;group[1].past=1023
x=effective_widths(group,8,enabled=True)
assert x[1]==1 and x[2]==1
checks.append('remaining output/context cap can reduce W to non-power-of-two existing tail rule')
old=[dict(v)for v in low]
answer=request_width(old,8,8,enabled=True)
assert old==low and answer==4
checks.append('history readonly; current target outcome is not an input')
print(json.dumps({'passed':True,'CPU_only':True,'model':False,'GPU':False,'checks':checks,'real_cycle_replay':False,'wall_speedup':'UNKNOWN','quality_verified':False}))
