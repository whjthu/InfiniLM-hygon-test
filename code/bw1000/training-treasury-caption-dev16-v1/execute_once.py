"""One future root-published local public-metadata acquire/select scope; no SSH."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys
import time
import uuid

G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
SOURCE=G/'code/bw1000/training-treasury-caption-dev16-v1'
RUN=G/'local-only/training-treasury-caption-dev16-v1-acquire-select-original'


def main(publication,expected_claim):
    epoch,started=time.time(),time.monotonic();deadline,monotonic_end=epoch+300,started+300
    if RUN.exists():raise RuntimeError('Original exclusive scope exists; reconcile, never replay')
    RUN.mkdir();request=uuid.uuid4().hex
    def save(name,value):
        with (RUN/name).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    save('intent.json',dict(request=request,epoch=epoch,deadline=deadline,publication=publication,
        claim_SHA256=expected_claim,maximum_invocations=1,replay_allowed=False,planned_images=16))
    try:
        if not re.fullmatch('[0-9a-f]{40}',publication) or not re.fullmatch('[0-9a-f]{64}',expected_claim):raise ValueError('Invalid publication/claim')
        claim_bytes=(SOURCE/'CLAIM.json').read_bytes()
        if hashlib.sha256(claim_bytes).hexdigest()!=expected_claim:raise ValueError('Published claim SHA differs')
        claim=json.loads(claim_bytes)
        if (claim['whole_epoch_seconds'],claim['memory_limit_bytes'],claim['maximum_HTTP_open_attempts'])!=(300,2*1024**3,1):
            raise ValueError('Resource or request budget changed')
        manifest=json.loads((SOURCE/'MF.json').read_bytes())
        actual={p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
        if actual!=set(manifest)|{'MF.json'}:raise ValueError('Source file set differs')
        for name,pin in manifest.items():
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe source member')
            path=SOURCE/name;raw=path.read_bytes()
            if path.is_symlink() or len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['SHA256']:
                raise ValueError('Source bytes differ')
        git=['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-c','core.fsmonitor=false','-C',str(G)]
        def run_git(argv):
            remaining=min(deadline-time.time(),monotonic_end-time.monotonic())-10
            if remaining<=0:raise TimeoutError('Original epoch exhausted before fetch')
            return subprocess.check_output(git+argv,stderr=subprocess.DEVNULL,timeout=min(10,remaining))
        rel=SOURCE.relative_to(G).as_posix()
        if run_git(['show',publication+':'+rel+'/MF.json'])!=(SOURCE/'MF.json').read_bytes():raise ValueError('Publication sourceMF differs')
        if run_git(['show',publication+':'+rel+'/CLAIM.json'])!=claim_bytes:raise ValueError('Publication claim differs')
        if run_git(['status','--porcelain','--untracked-files=all','--',rel]).strip():raise ValueError('Ownsource dirty/untracked')
        from job_guard import guarded_run
        gate=RUN/'memory-gate.json'
        argv=[sys.executable,'-B','-u',str(SOURCE/'worker.py'),'--mode','fetch','--directory',str(RUN),
              '--gate',str(gate),'--deadline',str(deadline),'--monotonic-end',str(monotonic_end)]
        guarded=guarded_run(argv,gate,RUN,deadline,monotonic_end)
        result_path=RUN/'worker-result.json';worker=json.loads(result_path.read_bytes()) if result_path.is_file() else None
        passed=bool(guarded['returncode']==0 and not guarded['timed_out'] and worker
            and worker['state']=='PUBLIC_MEMBER_ACQUIRED_METADATA_SELECTION_RECORDED'
            and worker['actual_metadata_selected']==16 and worker['missing_slots']==0
            and worker['selection']['missing_reference_groups']==0)
        outcome=dict(request=request,passed=passed,epoch=epoch,deadline=deadline,guarded_process=guarded,
            worker=worker,replay_allowed=False,evaluation_ready=False,training_GO=False,GPU_GO=False,
            qualification='Bounded public metadata acquisition/selection only; not dataset/media/scorer/model qualification',
            pending_denominator=16 if worker is None else worker['missing_slots'])
    except BaseException as error:
        outcome=dict(request=request,passed=False,epoch=epoch,deadline=deadline,error_type=type(error).__name__,
            replay_allowed=False,evaluation_ready=False,training_GO=False,GPU_GO=False,planned_images=16,missing_slots=16)
    save('outcome.json',outcome)
    print(json.dumps(dict(passed=outcome['passed'],request=request,evaluation_ready=False,training_GO=False,GPU_GO=False)))
    return 0 if outcome['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--claim-sha256',required=True)
    a=p.parse_args();raise SystemExit(main(a.publication,a.claim_sha256))
