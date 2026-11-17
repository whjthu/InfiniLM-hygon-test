"""Exercise the actual argv and timeout constructors, plus terminal visibility."""
import argparse
import json
from pathlib import Path

from budget import command,limits,terminal_timeout
from slurm_accounting import wait_terminal
from inside import placement
from launch_once import accounting_for
from runtime_gate import original_cpu330_accounting


def check():
    valid=[];rejected=[]
    for elapsed in (0,20.1,119.2,199.9,300,420,499.9,500):
        argv,budget=command(['apptainer','exec'],'/source/inside.py',
            dict(framework='/framework',container='/container'),'test',1600,2600,1000+elapsed,2000+elapsed)
        left=600-elapsed
        assert '--mem=16384M' in argv and '--gres=none' in argv and '--time='+budget['slurm_time'] in argv
        assert '--jobid=1103606' in argv and '--nodelist=f13r1n11' in argv
        assert '--jobid=1087656' not in argv and '--nodelist=a01r4n14' not in argv
        assert 60<=budget['slurm_seconds']<=480 and budget['slurm_seconds']%60==0
        assert 0<budget['inside_seconds']<=450 and budget['inside_seconds']+10<=budget['slurm_seconds']
        assert budget['client_seconds']>budget['inside_seconds'] and budget['client_seconds']<=left-40
        assert budget['slurm_seconds']<=left-40 and budget['work_deadline']<=1525
        assert '--kill-after=10s '+str(budget['inside_seconds']) in argv[-1]
        assert '--deadline 1600' in argv[-1] and '--work-deadline '+str(budget['work_deadline']) in argv[-1]
        valid.append(dict(elapsed=elapsed,**budget))
    def reject(name,fn):
        try:fn()
        except ValueError:rejected.append(name)
        else:raise AssertionError('accepted '+name)
    for elapsed in (500.01,525,599,600,601):
        reject('late_'+str(elapsed),lambda elapsed=elapsed:limits(1600,2600,1000+elapsed,2000+elapsed))
    reject('expired_monotonic',lambda:limits(1600,2600,1000,2600))
    reject('NaN',lambda:limits(float('nan'),2600,1000,2000))
    reject('infinite',lambda:limits(1600,float('inf'),1000,2000))
    assert terminal_timeout(1600,2600,2004,1000,2000)==4
    assert terminal_timeout(1003,2600,2020,1000,2000)==3
    assert terminal_timeout(1600,2600,2020,1000,2000)==12
    reject('terminal_own_expired',lambda:terminal_timeout(1600,2600,2000,1000,2000))
    reject('terminal_global_expired',lambda:terminal_timeout(1000,2600,2020,1000,2000))
    queried=[]
    def fixture_query(*argv):
        parent=argv[argv.index('-j')+1];queried.append(parent)
        if parent=='1087656':
            return ['1087656.330|training-qwen-treasury-explicit-dev-CPU-v1|COMPLETED|0:0|20|a01r4n14|cpu=8,mem=12G,node=1']
        assert parent=='1103606'
        return ['1103606.999|test|COMPLETED|0:0|1|f13r1n11|cpu=8,mem=16G,node=1']
    own=accounting_for(fixture_query,'1103606','test')
    original=original_cpu330_accounting(lambda parent,label:accounting_for(fixture_query,parent,label))
    assert own[0]=='1103606.999' and original[0]=='1087656.330' and queried==['1103606','1087656']
    assert placement(dict(SLURM_JOB_ID='1103606',SLURM_STEP_ID='999',SLURM_STEP_GPUS=''),'f13r1n11')==dict(
        parent='1103606',node='f13r1n11',step='999')
    reject('inside_old_parent',lambda:placement(dict(SLURM_JOB_ID='1087656',SLURM_STEP_ID='999'),'f13r1n11'))
    reject('inside_old_node',lambda:placement(dict(SLURM_JOB_ID='1103606',SLURM_STEP_ID='999'),'a01r4n14'))
    reject('inside_GPU_assigned',lambda:placement(dict(SLURM_JOB_ID='1103606',SLURM_STEP_ID='999',SLURM_STEP_GPUS='0'),'f13r1n11'))
    clock=[0.0]
    rows=iter([None,['1103606.999','test','RUNNING','0:0','1'],
               ['1103606.999','test','COMPLETED','0:0','2']])
    result=wait_terminal(lambda:next(rows),5,interval=1,max_observations=5,
        monotonic=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
    assert result['terminal_visible'] and len(result['observations'])==3
    missing=wait_terminal(lambda:None,1,max_observations=2,monotonic=lambda:1,sleep=lambda _:None)
    assert not missing['terminal_visible'] and missing['row'] is None
    return dict(state='LOCAL_ACTUAL_BUDGET_ARGV_AND_TERMINAL_BOUNDARIES_PASS',
        valid_actual_argv_cases=valid,rejected_cases=rejected,terminal_timeout_valid_cases=3,
        terminal_visibility_cases=2,resource_routes=dict(own_parent='1103606',own_node='f13r1n11',
            original_CPU330_parent='1087656',query_parents=queried,inside_owner_checked=True,fixture_handles_not_actual=True),
        actual_native_calls=0,actual_Slurm_steps=0,GPU_cards=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=check()
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],valid_actual_argv_cases=len(result['valid_actual_argv_cases']),
        rejected_cases=len(result['rejected_cases']),actual_Slurm_steps=0,GPU_cards=0)))
