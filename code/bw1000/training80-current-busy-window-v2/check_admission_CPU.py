"""New gates only; no sysfs, subprocess, Slurm or old sampler fixture replay."""
import json
from admission import parent_gate,step_gate

nodes=['n%02d'%i for i in range(40)]
line='JobId=1087656 UserId=alias(154459) JobState=RUNNING NumNodes=40 AllocTRES=cpu=2560,node=40,gres/dcu=320 NodeList=fixture'
assert parent_gate(line,'1087656',154459,nodes,lambda _:nodes)['JobState']=='RUNNING'
assert len(step_gate(['StepId=1087656.batch State=RUNNING','StepId=1087656.extern State=RUNNING'],'1087656'))==2
bad=[]
for name,call in [
 ('wrong_owner',lambda:parent_gate(line,'1087656',99,nodes,lambda _:nodes)),
 ('changed_membership',lambda:parent_gate(line,'1087656',154459,nodes,lambda _:nodes[:-1])),
 ('numeric_active',lambda:step_gate(['StepId=1087656.999 State=RUNNING'],'1087656')),
 ('unknown_TDB',lambda:step_gate(['StepId=1087656.TDB State=PENDING'],'1087656')),
 ('missing_step_inventory',lambda:step_gate([],'1087656'))]:
 try:call()
 except ValueError:bad.append(name)
 else:raise AssertionError(name)
print(json.dumps(dict(valid=2,rejected=bad,actual_sampling=False,SSH=False,Slurm=False)))
