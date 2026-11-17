"""Root-published unique metadata-only execution; no action by default."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from job_guard import guarded_run
from metadata import canonical,need,url_gate

SOURCE=Path(__file__).resolve().parent
G=SOURCE.parents[2]
RUN=G/'local-only/training-CORD-v2-official-metadata-once-v1-original'


def main(publication,claim_sha,root_go):
    need(root_go,'root GO defaultfalse; prepared only')
    need(re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',claim_sha),'Bad source/claim pin')
    epoch=time.time();deadline=epoch+120;monotonic_end=time.monotonic()+120
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),*args],cwd=G,timeout=min(10,max(.01,deadline-time.time()-8)))
    need(not git('status','--porcelain','--untracked-files=all','--',SOURCE.relative_to(G).as_posix()).strip() and not git('diff','--cached','--name-only').strip(),'Own source dirty or index nonempty')
    need(hashlib.sha256((SOURCE/'CLAIM.json').read_bytes()).hexdigest()==claim_sha,'Claim SHA changed')
    mf=json.loads((SOURCE/'MANIFEST.json').read_bytes())
    for name,pin in mf['files'].items():
        path=SOURCE/name;need(path.is_file() and not path.is_symlink(),'Invalid source file');raw=path.read_bytes()
        need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'] and git('show',publication+':'+path.relative_to(G).as_posix())==raw,'Source unpublished/changed')
    need(git('show',publication+':'+(SOURCE/'MANIFEST.json').relative_to(G).as_posix())==(SOURCE/'MANIFEST.json').read_bytes(),'Published MF changed')
    c=json.loads((SOURCE/'CONTRACT.json').read_bytes());url_gate(c,json.loads((SOURCE/'WEB_ATTEMPTS.json').read_bytes()))
    need(time.time()<deadline-30,'Original epoch exhausted before unique Job')
    RUN.mkdir() # exclusive, including UNKNOWN/failed prior intent; never replay
    intent=dict(request=uuid.uuid4().hex,epoch=epoch,deadline=deadline,publication=publication,claim_SHA256=claim_sha,
        source_manifest_SHA256=hashlib.sha256((SOURCE/'MANIFEST.json').read_bytes()).hexdigest(),URL=c['URL'],planned_GETs=1,body_cap_bytes=2097152,
        CPU=2,job_commit_limit_bytes=2147483648,actual_media_scope=False,actual_parquet_body_scope=False)
    (RUN/'intent.json').write_bytes(canonical(intent))
    argv=[sys.executable,'-B','-X','utf8',str(SOURCE/'worker.py'),'--run',str(RUN),'--epoch',str(epoch),'--deadline',str(deadline)]
    job=guarded_run(argv,RUN/'job-gate.json',RUN,deadline,monotonic_end)
    (RUN/'outcome.json').write_bytes(canonical(dict(**intent,known_terminal=True,job=job,Dataset_GO=False,training_GO=False,GPU_GO=False)))
    print(json.dumps(dict(request=intent['request'],known_terminal=True,returncode=job['returncode'],timed_out=job['timed_out'])))
    return job['returncode']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--claim-sha256',required=True);p.add_argument('--root-go',action='store_true')
    a=p.parse_args();raise SystemExit(main(a.publication,a.claim_sha256,a.root_go))
