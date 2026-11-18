"""Four new purestdlib boundaries; no actual process priority changes."""
import hashlib
import json
from pathlib import Path
from nice_policy import apply_nice

def main():
    results=[]
    for inherited in (0,10,19):
        state=[inherited];increments=[]
        def read():return state[0]
        def increment(value):
            assert value>=0
            increments.append(value);state[0]=min(19,state[0]+value)
        record=apply_nice(read,increment)
        assert record['requested_nice']==record['actual_nice']==max(inherited,10)
        assert increments==[max(inherited,10)-inherited]
        results.append(dict(inherited=inherited,passed=True,record=record,actual_OS_priority_calls=0))
    calls=[]
    try:apply_nice(lambda:0,lambda value:calls.append(value))
    except ValueError as error:
        assert str(error)=='ACTUAL_NICE_NOT_LOW_ENOUGH' and calls==[10]
        results.append(dict(inherited=0,actual_after=0,rejected=True,reason=str(error)))
    else:raise AssertionError('UNCHANGED_HIGH_PRIORITY_ACCEPTED')
    G=Path(__file__).resolve().parents[3]
    out=G/'local-only/training-hunyuanvideo-execution-mount-SHA-CPU-v3-local-checks'
    out.mkdir(exist_ok=False)
    result=dict(passed=True,normal_count=3,rejected_count=1,new_boundaries=4,results=results,
        remote_calls=0,new_Slurm_steps=0,weight_reads=0,Torch_SDK_imports=0,old_fixtures_repeated=False)
    b=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    path=out/'nice-first-result.json';path.write_bytes(b)
    print(json.dumps(dict(passed=True,normal=3,rejected=1,path=str(path),bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())))

if __name__=='__main__':main()
