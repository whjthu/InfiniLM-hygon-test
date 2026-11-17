"""Only newly added private diagnostic byte bound; no old29 or GPU fixture replay."""
import json
from failure_private import failure_record


def run():
    raw=failure_record('ValueError','initial',None,'synthetic failure','synthetic trace')
    assert len(raw)<=16384 and json.loads(raw)['phase']=='initial'
    try:failure_record('ValueError','initial',None,'x'*4097,'synthetic trace')
    except ValueError:rejected=1
    else:raise AssertionError('Oversize diagnostic accepted')
    return dict(passed=True,normal=1,rejected=rejected,old3valid7reject_rerun=False,Torch_imports=0,state_reads=0,GPU_calls=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
