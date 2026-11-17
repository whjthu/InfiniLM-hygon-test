"""Only new clock IPC boundaries, no Job/HTTP or scientific parser execution."""
import json
from clock_ipc import validate_gate


def run():
    epoch=1794942931.9674995;deadline=1794943051.9674995
    gate=dict(memory_limit_bytes=2147483648,deadline=deadline,CPU_count=2,worker_affinity_mask=3,job_affinity_verified=True)
    normal=validate_gate(json.dumps(gate).encode(),epoch,deadline,3)
    assert type(normal['deadline']) is float and normal['deadline']==deadline
    bad=[]
    def reject(name,raw,ep=epoch,dl=deadline,mask=3):
        try:validate_gate(raw,ep,dl,mask)
        except (ValueError,KeyError,TypeError):bad.append(name);return
        raise AssertionError('Accepted '+name)
    for key,value,name in [('deadline',deadline+.25,'different_deadline'),('deadline',True,'boolclock'),('deadline',float('inf'),'nonfinite'),('CPU_count',True,'boolCPU'),('CPU_count',1,'CPU1'),('memory_limit_bytes',2147483647,'wrongmemory'),('job_affinity_verified',False,'unverified'),('worker_affinity_mask',7,'wrongmask')]:
        candidate=dict(gate);candidate[key]=value;reject(name,json.dumps(candidate))
    candidate=dict(gate);del candidate['deadline'];reject('missingdeadline',json.dumps(candidate))
    reject('changedepoch',json.dumps(gate),ep=epoch-1)
    reject('boolactualmask',json.dumps(gate),mask=True)
    reject('duplicate',json.dumps(gate)[:-1]+',"deadline":'+str(deadline)+'}')
    return dict(passed=True,normal=1,rejected=len(bad),rejected_cases=bad,HTTP_GETs=0,Jobs=0,scientific_contracts_rerun=0,
        original_fractional_deadline=deadline,old_failure_preserved=True)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
