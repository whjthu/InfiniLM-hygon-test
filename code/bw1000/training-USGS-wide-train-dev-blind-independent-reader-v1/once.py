"""Local once reader, original120 epoch and exact qualified WindowsJob guard."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

D=Path(__file__).resolve().parent
G=D.parents[2]
sys.path.insert(0,str(D))


def save(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def controller(publication,mf_sha):
    epoch=time.time();mono=time.monotonic();deadline=epoch+120
    import re
    assert re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',mf_sha)
    def left():
        value=min(deadline-time.time(),mono+120-time.monotonic())-8
        if value<=0:raise TimeoutError('ORIGINAL120_EPOCH')
        return value
    from verify_package import verify
    verify()
    raw=(D/'MANIFEST.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==mf_sha
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*args],timeout=min(10,left()))
    assert git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git'
    assert git('rev-parse','HEAD').decode().strip()==publication
    assert git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split()==[publication,'refs/heads/bw1000-experiments']
    prefix=D.relative_to(G).as_posix()
    assert git('cat-file','blob',publication+':'+prefix+'/MANIFEST.json')==raw
    assert not git('status','--porcelain=v1','--untracked-files=all','--',prefix)
    for name in json.loads(raw)['files']:
        assert git('cat-file','blob',publication+':'+prefix+'/'+name)==(D/name).read_bytes()
    binding=json.loads((D/'BINDING.json').read_bytes());out=G/binding['output_namespace']
    assert not out.exists() and not out.is_symlink();out.mkdir(mode=0o700)
    request=uuid.uuid4().hex
    context=dict(request=request,publication=publication,source_manifest_SHA256=mf_sha,epoch=epoch,deadline=deadline,
                 original_request=binding['original_request'],original_publication=binding['original_publication'])
    save(out/'intent.json',context)
    result=dict(context,passed=False,job=None,execution_state='UNKNOWN',replay_allowed=False)
    try:
        from job_guard import guarded_run
        save(out/'attempt.json',dict(context,about_to_start_worker=True))
        job=guarded_run([sys.executable,'-B','-X','utf8',str(D/'once.py'),'--worker','--origin',str(epoch),
                         '--output',str(out)],out/'job-gate.json',out,deadline,mono+120)
        result['job']=job;result['execution_state']='RETURNED'
        p=out/'independent-result.json'
        if p.is_file() and not p.is_symlink() and p.stat().st_size<=131072:
            b=p.read_bytes();v=json.loads(b)
            result['result_pin']=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
            result['passed']=bool(job['returncode']==0 and job['timed_out'] is False and v['passed'] is True
                and v['epoch']==epoch and v['deadline']==deadline and v['verified_charts']==30 and v['verified_QA']==150)
    except BaseException as error:
        result['error_type']=type(error).__name__;result['error_SHA256']=hashlib.sha256(str(error).encode()).hexdigest()
    save(out/'outcome.json',result);print(json.dumps(result,sort_keys=True));return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication');p.add_argument('--source-manifest-sha256')
    p.add_argument('--worker',action='store_true');p.add_argument('--origin',type=float);p.add_argument('--output')
    a=p.parse_args()
    if a.worker:
        assert a.origin and math.isfinite(a.origin) and 0<=time.time()-a.origin<120
        from verify_package import verify
        verify()
        binding=json.loads((D/'BINDING.json').read_bytes())
        assert Path(a.output).resolve()==(G/binding['output_namespace']).resolve()
        from read_data import main
        raise SystemExit(main(a.origin,a.origin+120,a.output))
    raise SystemExit(controller(a.publication,a.source_manifest_sha256))
