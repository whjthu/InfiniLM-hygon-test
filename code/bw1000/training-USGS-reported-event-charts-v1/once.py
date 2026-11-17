"""One published offline derivation, no HTTP/model/Slurm. Root dispatch only."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from contracts import canonical,need,sha
from clock_ipc import validate_gate
from job_guard import current_affinity,guarded_run

S=Path(__file__).resolve().parent;G=S.parents[2];RUN=G/'local-only/training-USGS-reported-event-charts-v1-original'


def worker(epoch,deadline):
    mono=time.monotonic()+max(0,deadline-time.time())
    def remaining():
        left=min(deadline-time.time(),mono-time.monotonic())-8;need(left>0,'Original120epoch exhausted');return left
    while not (RUN/'job-gate.json').exists():remaining();time.sleep(.02)
    validate_gate((RUN/'job-gate.json').read_bytes(),epoch,deadline,current_affinity()['process_mask'])
    (RUN/'worker-enter.lock').open('x').close();plan=json.loads((S/'PLAN.json').read_bytes())
    result=dict(passed=False,epoch=epoch,deadline=deadline,HTTP_GETs=0,source_slots=50,planned_charts=None,planned_QA=None,
        completed_chart_slots=0,completed_QA_slots=0,Dataset_GO=False,training_GO=False,GPU_GO=False,model_calls=0)
    def progress(phase,info):
        for key in ['planned_charts','planned_QA','completed_chart_slots','completed_QA_slots']:
            if key in info:result[key]=info[key]
        with (RUN/'progress.jsonl').open('ab') as f:f.write(canonical(dict(phase=phase,time=time.time(),**info)))
    try:
        remaining();source=G/plan['source_path'];docs={}
        for name,pin in plan['source_pins'].items():
            p=source/name;need(p.is_file() and not p.is_symlink(),'Exact qualified saved source unavailable')
            before=p.stat();raw=p.read_bytes();after=p.stat();need(before.st_size==after.st_size==pin['bytes'] and before.st_mtime_ns==after.st_mtime_ns and sha(raw)==pin['SHA256'],'Exact qualified source changed/unstable');docs[name]=json.loads(raw)
        independent=docs['result.json'];need(independent['passed'] is True and independent['actual_event_slots']==independent['qualified_numeric_slots']==50
            and independent['unqualified_retained_slots']==0 and independent['body_SHA256']==plan['original_source_body_SHA256']
            and independent['producer_passed'] is False and independent['metadata_count'] is None,'Independent source qualification differs')
        from build import build
        manifest=build(docs['source-records-private.json']['rows'],plan,RUN/'dataset',remaining,progress);remaining()
        result.update(passed=manifest['failed_charts']==0,state='DATA_ARTIFACTS_ONLY_PENDING_VISUAL_NATIVE_EFFECT',
            actual_charts=manifest['actual_charts'],actual_inputs=manifest['actual_inputs'],retained_QA_slots=manifest['retained_QA_slots'],
            closure_groups=manifest['closure_groups'],source_geographic_groups=manifest['source_geographic_groups'],source_alias_groups=manifest['source_alias_groups'],split_group_counts=manifest['split_group_counts'],
            failed_charts=manifest['failed_charts'],actual_PNG_bytes=manifest['actual_PNG_bytes'],manifest_SHA256=sha((RUN/'dataset/manifest-private.json').read_bytes()),
            original_source_numeric_slots=50,producer_failed_preserved=True,visual_readability='PENDING',native_processor='NOT_RUN')
    except Exception as e:result.update(state='KNOWN_FAILED_NO_REBUILD_OR_BACKFILL',error_type=type(e).__name__,error=str(e));progress('failure',dict(error_type=type(e).__name__,error=str(e)))
    (RUN/'worker-result.json').write_bytes(canonical(result));return 0 if result['passed'] else 1


def controller(pub,claim,go):
    need(go,'Root GO defaultfalse');need(re.fullmatch('[0-9a-f]{40}',pub or '') and re.fullmatch('[0-9a-f]{64}',claim or ''),'Bad published source identity')
    epoch=time.time();deadline=epoch+120;mono=time.monotonic()+120
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),*args],cwd=G,timeout=min(10,max(.01,deadline-time.time()-8)))
    need(not git('status','--porcelain','--untracked-files=all','--',S.relative_to(G).as_posix()).strip() and not git('diff','--cached','--name-only').strip(),'Own source dirty/index nonempty')
    need(sha((S/'CLAIM.json').read_bytes())==claim,'Claim changed')
    for name,pin in json.loads((S/'MANIFEST.json').read_bytes())['files'].items():
        p=S/name;need(p.is_file() and not p.is_symlink(),'Ownsource missing/symlink');raw=p.read_bytes()
        need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'] and git('show',pub+':'+p.relative_to(G).as_posix())==raw,'Published source differs')
    need(git('show',pub+':'+(S/'MANIFEST.json').relative_to(G).as_posix())==(S/'MANIFEST.json').read_bytes(),'Published MF differs')
    need(time.time()<deadline-30,'Epoch exhausted before unique Job');RUN.mkdir()
    intent=dict(request=uuid.uuid4().hex,publication=pub,claim_SHA256=claim,source_manifest_SHA256=sha((S/'MANIFEST.json').read_bytes()),epoch=epoch,deadline=deadline,HTTP_GETs=0,CPU=2,job_process_commit_bytes=2147483648,source_slots=50,max_charts=36,max_QA=180)
    (RUN/'intent.json').write_bytes(canonical(intent));job=guarded_run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--worker','--epoch',str(epoch),'--deadline',str(deadline)],RUN/'job-gate.json',RUN,deadline,mono)
    (RUN/'outcome.json').write_bytes(canonical(dict(**intent,known_terminal=True,job=job,Dataset_GO=False,training_GO=False,GPU_GO=False)))
    print(json.dumps(dict(request=intent['request'],known_terminal=True,returncode=job['returncode'],timed_out=job['timed_out'])));return job['returncode']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--epoch',type=float);p.add_argument('--deadline',type=float);p.add_argument('--publication');p.add_argument('--claim-sha256');p.add_argument('--root-go',action='store_true');a=p.parse_args()
    raise SystemExit(worker(a.epoch,a.deadline) if a.worker else controller(a.publication,a.claim_sha256,a.root_go))
