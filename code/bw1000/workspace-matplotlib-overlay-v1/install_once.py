"""Prepared isolated installation; root GO false until reviewed source publication."""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from overlay_env import D,G,RUN,OVERLAY
from job_guard import guarded_run

PYTHON=Path('C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')


def need(ok,message):
    if not ok:raise ValueError(message)


def main(publication,manifest_sha):
    claim=json.loads((D/'CLAIM.json').read_bytes())
    need(claim['root_GO'] is True and claim['install_GO'] is True,'ROOT_GO_FALSE_PREPARATION_ONLY')
    need(re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',manifest_sha),'exact publication/MF required')
    need(Path(sys.executable).resolve()==PYTHON.resolve() and sys.version_info[:3]==(3,12,14),'fixed Python required')
    epoch=time.time();mono=time.monotonic()+300;deadline=epoch+300
    mfraw=(D/'SOURCE_FILES.json').read_bytes();need(hashlib.sha256(mfraw).hexdigest()==manifest_sha,'source MF differs')
    published=subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',publication+':'+D.relative_to(G).as_posix()+'/SOURCE_FILES.json'],timeout=12)
    need(published==mfraw,'published source MF differs')
    for name,pin in json.loads(mfraw)['files'].items():
        raw=(D/name).read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'source bytes differ')
    git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
    need(not subprocess.check_output(git+['status','--porcelain=v1','--untracked-files=all','--',D.relative_to(G).as_posix()],timeout=12).strip(),
         'own source is dirty or unpublished')
    need(not subprocess.check_output(git+['diff','--cached','--name-only'],timeout=12).strip(),'index must be empty')
    lock=json.loads((D/'WHEEL_LOCK.json').read_bytes())
    for name,version in lock['existing_runtime'].items():need(md.version(name)==version,'shared dependency identity changed')
    need(md.version('pip')=='26.2.2','fixed pip version differs')
    RUN.mkdir(mode=0o700)  # Exclusive: failure/timeout leaves original scope, never replay/overwrite.
    for name in ('wheels','overlay','tmp','mplconfig'):(RUN/name).mkdir()
    (RUN/'install-lock.json').write_bytes((D/'WHEEL_LOCK.json').read_bytes())
    (RUN/'intent.json').write_text(json.dumps(dict(publication=publication,MF_SHA256=manifest_sha,epoch=epoch,deadline=deadline,
       python=str(PYTHON),CPU=2,memory_commit_limit_bytes=2*1024**3,total_owned_scope_cap_bytes=128*1024**2,
       wheel_GETs_max=7,wheel_download_bytes=lock['wheel_download_bytes'],shared_runtime_changed=False,
       global_environment_changed=False,WDI_GETs=0,Torch_imports=0,GPU=0),indent=2)+'\n',encoding='utf8')
    command=[str(PYTHON),'-B','-X','utf8',str(D/'install_worker.py'),str(epoch)]
    job=guarded_run(command,RUN/'job-gate.json',RUN,deadline,mono)
    worker=json.loads((RUN/'worker-result.json').read_bytes()) if (RUN/'worker-result.json').exists() else None
    total=sum(path.stat().st_size for path in RUN.rglob('*') if path.is_file())
    need(total<128*1024**2,'actual owned scope exceeds128MiB')
    result=dict(passed=bool(job['returncode']==0 and not job['timed_out'] and not job['directory_limit_exceeded'] and worker and worker['passed']),
        publication=publication,epoch=epoch,deadline=deadline,job=job,worker=worker,owned_scope_bytes=total,
        overlay=str(OVERLAY),overlay_bytes=sum(path.stat().st_size for path in OVERLAY.rglob('*') if path.is_file()),
        scope_cap_bytes=128*1024**2,shared_runtime_changed=False,global_environment_changed=False,WDI_GETs=0,Torch_imports=0,GPU=0)
    (RUN/'outcome.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publication',required=True);parser.add_argument('--source-manifest-sha',required=True)
    args=parser.parse_args();raise SystemExit(main(args.publication,args.source_manifest_sha))
