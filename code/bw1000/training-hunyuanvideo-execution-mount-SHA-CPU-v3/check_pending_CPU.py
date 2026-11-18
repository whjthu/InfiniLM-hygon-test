"""One new purestdlib classifier boundary; never queries Slurm or imports worker."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from pending_placeholder import fields,versions,classify,VERSION

D=Path(__file__).resolve().parent
G=D.parents[2]
PIN='e40bf4440aec450c4d9d104c7e9f1ca00c88442f91a5a5f3fb9a1870a318574b'

def main():
    path=G/'local-only/T09-execution-mount-CPU-original-failure-first-read/stdout.private'
    raw=path.read_bytes();assert len(raw)==3620 and hashlib.sha256(raw).hexdigest()==PIN
    found=[]
    def visit(value):
        if isinstance(value,str):
            found.extend(line for line in value.splitlines() if 'StepId=1103606.TDB ' in line)
        elif isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(json.loads(raw));assert len(found)==1
    row=fields(found[0])
    hosts=['f12r2n'+str(n).zfill(2) for n in range(10,14)]
    hosts+=['f13r1n'+str(n).zfill(2) for n in (1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,19,20)]
    hosts+=['f13r2n'+str(n).zfill(2) for n in range(1,12)]
    hosts+=['f13r3n13','f13r3n14']+['f13r4n'+str(n).zfill(2) for n in range(5,10)]
    assert len(hosts)==40
    passed=[];rejected=[]
    def reject(name,fn):
        try:fn()
        except ValueError as error:rejected.append(dict(name=name,reason=str(error)))
        else:raise AssertionError('BAD_BOUNDARY_ACCEPTED:'+name)
    v=versions(['slurm '+VERSION],['slurm '+VERSION],['SLURM_VERSION = '+VERSION]);passed.append('actual_exact_three_versions')
    assert classify(row,hosts,hosts,154459)=='PENDING_NOTIFICATION_PLACEHOLDER_NO_ALLOCATED_TRES'
    passed.append('actual_saved_TDB_complete_shape')
    numeric=dict(row,StepId='1103606.188',State='RUNNING')
    assert classify(numeric,hosts,['f13r1n05','f13r1n07'],154459)=='NUMBERED_STEP_REQUIRES_NODE_CONFLICT_CHECK'
    passed.append('numbered_remains_subject_to_original_node_guard')
    for key in ('StepId','State','StartTime','Name','TRES','UserId','NodeList','SrunHost:Pid'):
        bad=copy.deepcopy(row);del bad[key]
        reject('missing_'+key,lambda bad=bad:classify(bad,hosts,hosts,154459))
    for key,value in [('State','RUNNING'),('State','COMPLETING'),('StartTime','2026-11-18T00:00:00'),
                      ('Name','named'),('TRES','cpu=2'),('UserId','1'),('StepId','1103606.TBD'),
                      ('StepId','1103606.unknown'),('StepId','1.TDB'),('SrunHost:Pid','zz-login02:0'),
                      ('SrunHost:Pid','zz-login02:2147483648'),('SrunHost:Pid','a..b:3940770'),
                      ('SrunHost:Pid','-bad:3940770'),('AllocTRES','gres/dcu=1')]:
        bad=dict(row,**{key:value})
        reject('changed_'+key+'_'+value,lambda bad=bad:classify(bad,hosts,hosts,154459))
    reject('step_nodes_mismatch',lambda:classify(row,hosts,hosts[:-1]+['foreign'],154459))
    reject('duplicate_step_nodes',lambda:classify(row,hosts,hosts[:-1]+hosts[:1],154459))
    reject('duplicate_parent_nodes',lambda:classify(row,hosts[:-1]+hosts[:1],hosts,154459))
    reject('duplicate_field',lambda:fields(found[0]+' State=PENDING'))
    bad=dict(numeric);del bad['NodeList']
    reject('numbered_missing_nodes',lambda:classify(bad,hosts,[],154459))
    reject('client_version_changed',lambda:versions(['slurm 24.05.8'],['slurm '+VERSION],['SLURM_VERSION = '+VERSION]))
    reject('server_version_changed',lambda:versions(['slurm '+VERSION],['slurm '+VERSION],['SLURM_VERSION = 24.05.8']))
    reject('server_version_duplicate',lambda:versions(['slurm '+VERSION],['slurm '+VERSION],['SLURM_VERSION = '+VERSION]*2))
    result=dict(passed=True,normal_count=len(passed),rejected_count=len(rejected),normal=passed,rejected=rejected,
        actual_small_record=dict(path=str(path.relative_to(G)).replace('\\','/'),bytes=len(raw),SHA256=PIN),
        actual_row=row,version_evidence=v,remote_calls=0,new_steps=0,weight_reads=0,Torch_SDK_imports=0,
        original_request_state='UNKNOWN_NO_PROCESS_TERMINAL_INFERENCE',old_fixtures_repeated=False)
    out=G/'local-only/training-hunyuanvideo-execution-mount-SHA-CPU-v2-local-checks'
    out.mkdir(parents=True,exist_ok=False)
    target=out/'classifier-first-result.json'
    target.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    b=target.read_bytes()
    print(json.dumps(dict(passed=True,normal=len(passed),rejected=len(rejected),receipt=str(target),bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())))

if __name__=='__main__':main()
