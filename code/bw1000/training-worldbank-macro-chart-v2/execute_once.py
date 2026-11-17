"""Prepared unique bounded local scope; no execution until root publishes/reviews it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from contracts import canonical,need
from job_guard import guarded_run

SOURCE=Path(__file__).resolve().parent
G=SOURCE.parents[2]
RUN=G/'local-only/training-worldbank-macro-chart-v2-original'


def git(*args):return subprocess.check_output(['git',*args],cwd=G,timeout=10)


def main(publication,claim_sha,root_go):
    need(root_go,'Root GO default false; preparation only')
    need(re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',claim_sha),'Bad publication/pin')
    epoch=time.time();deadline=epoch+300;mono=time.monotonic()+300
    need(not git('status','--porcelain','--untracked-files=all','--',SOURCE.relative_to(G).as_posix()).strip(),'Own source dirty/unpublished')
    need(not git('diff','--cached','--name-only').strip(),'Index not empty')
    claim=(SOURCE/'CLAIM.json').read_bytes();need(hashlib.sha256(claim).hexdigest()==claim_sha,'Claim changed')
    manifest=json.loads((SOURCE/'MANIFEST.json').read_bytes())
    for name,pin in manifest['files'].items():
        path=SOURCE/name;need(path.is_file() and not path.is_symlink(),'Missing/symlink source')
        raw=path.read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'Source pin mismatch')
        need(git('show',publication+':'+path.relative_to(G).as_posix())==raw,'Publication source bytes differ')
    failure_blob=git('show',publication+':results/bw1000/2026-11-17/training-worldbank-original-country-metadata-timeout-independent-terminal.json')
    need(len(failure_blob)==3827 and hashlib.sha256(failure_blob).hexdigest()=='17ffc175903046ca1286a1043ede5ecab15eff39eadb0ce8a7d1f1636685f00a','Published original independent failure prerequisite differs')
    need(git('show',publication+':'+(SOURCE/'MANIFEST.json').relative_to(G).as_posix())==(SOURCE/'MANIFEST.json').read_bytes(),'Published manifest differs')
    need(time.time()<deadline-30,'Original epoch too late before unique launch')
    # mkdir remains exclusive even if interrupted after namespace creation: UNKNOWN never replays.
    RUN.mkdir()
    intent=dict(request=uuid.uuid4().hex,epoch=epoch,deadline=deadline,publication=publication,claim_SHA256=claim_sha,
        source_manifest_SHA256=hashlib.sha256((SOURCE/'MANIFEST.json').read_bytes()).hexdigest(),CPU=2,memory_limit_bytes=2147483648,
        planned_GETs=3,HTTP_body_cap_bytes=6291456,prior_attempted_GETs=4,combined_attempted_GETs_max=7,planned_facts=504,planned_charts=72,planned_QA=648)
    (RUN/'intent.json').write_bytes(canonical(intent))
    argv=[sys.executable,'-B','-X','utf8',str(SOURCE/'worker.py'),'--run',str(RUN),'--epoch',str(epoch),'--deadline',str(deadline)]
    job=guarded_run(argv,RUN/'job-gate.json',RUN,deadline,mono)
    (RUN/'outcome.json').write_bytes(canonical(dict(**intent,job=job,known_terminal=True,Dataset_GO=False,training_GO=False,GPU_GO=False)))
    print(json.dumps(dict(request=intent['request'],known_terminal=True,returncode=job['returncode'],timed_out=job['timed_out'])))
    return job['returncode']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--claim-sha256',required=True);p.add_argument('--root-go',action='store_true')
    a=p.parse_args();raise SystemExit(main(a.publication,a.claim_sha256,a.root_go))
