"""Prepared unique local source admission; root publication/go required, no GPU/model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from job_guard import current_affinity,guarded_run
from clock_ipc import validate_gate
from sample import audit,canonical,contract_gate,get_once,js,need,sha

SOURCE=Path(__file__).resolve().parent;G=SOURCE.parents[2]
RUN=G/'local-only/training-USGS-fixed-history-source-v2-original'


def worker(directory,epoch,deadline):
    need(directory.resolve()==RUN.resolve(),'Only exact original namespace')
    started=time.monotonic();mono=started+max(0,deadline-time.time())
    def remaining():
        left=min(deadline-time.time(),mono-time.monotonic())-8;need(left>0,'Original120s epoch exhausted');return left
    while not (directory/'job-gate.json').exists():remaining();time.sleep(.02)
    gate=validate_gate((directory/'job-gate.json').read_bytes(),epoch,deadline,current_affinity()['process_mask'])
    (directory/'worker-enter.lock').open('x').close()
    c=js((SOURCE/'CONTRACT.json').read_bytes());result=dict(passed=False,epoch=epoch,deadline=deadline,maximum_requested_event_slots=50,
        actual_GET_attempts=0,actual_body_bytes=0,source_accessible=False,Dataset_GO=False,training_GO=False,GPU_GO=False,charts_generated=0,QA_generated=0,model_calls=0)
    def progress(phase,details):
        if phase=='GET.enter':result['actual_GET_attempts']+=1
        if phase=='GET.body':result['actual_body_bytes']=details['bytes']
        with (directory/'progress.jsonl').open('ab') as stream:stream.write(canonical(dict(phase=phase,time=time.time(),**details)))
    try:
        contract_gate(c);remaining();raw=get_once(directory/'original-GeoJSON-private.json',c,remaining,progress)
        parsed=js(raw);summary=audit(parsed,c);remaining();result['source_accessible']=True
        (directory/'event-identities-and-numerics-private.json').write_bytes(canonical(summary))
        result.update(passed=summary['all_source_numeric_slots_qualified'],state='FIXED_HISTORY_SAMPLE_SOURCE_ONLY' if summary['all_source_numeric_slots_qualified'] else 'SOURCE_RESPONDED_BUT_KEEP_EMPTY_OR_INVALID_SLOTS_NO_DATASET_GO',
            body_SHA256=sha(raw),actual_event_slots=summary['actual_event_slots'],qualified_numeric_slots=summary['qualified_numeric_slots'],
            unqualified_retained_slots=summary['unqualified_retained_slots'],event_ids_ordered_SHA256=summary['event_ids_ordered_SHA256'],sample_limit_reached=summary['sample_limit_reached'],
            source_manifest_SHA256=sha((directory/'event-identities-and-numerics-private.json').read_bytes()),api_version=summary['api_version'],generated_epoch_ms=summary['generated_epoch_ms'])
    except Exception as error:
        result.update(state='KNOWN_FAILED_SINGLE_SOURCE_NO_RETRY_OR_DIFFERENT_DATE_FALLBACK',error_type=type(error).__name__,error=str(error))
        progress('failure',dict(error_type=type(error).__name__,error=str(error)))
    result['wall_seconds']=time.monotonic()-started;(directory/'worker-result-private.json').write_bytes(canonical(result))
    return 0 if result['passed'] else 1



def original_preGET_failure(git):
    binding=json.loads((SOURCE/'ORIGINAL_PREGET_FAILURE.json').read_bytes())
    old=G/binding['namespace']
    for name,pin in binding['files'].items():
        p=old/name;need(p.is_file() and not p.is_symlink(),'Original evidence missing/symlink')
        raw=p.read_bytes();need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'Original failed evidence changed')
    intent=json.loads((old/'intent.json').read_bytes());outcome=json.loads((old/'outcome.json').read_bytes())
    need(intent['request']==outcome['request']==binding['request'] and intent['publication']==outcome['publication']==binding['publication']
        and intent['source_manifest_SHA256']==outcome['source_manifest_SHA256']==binding['source_manifest_SHA256'],'Original request/source identity differs')
    job=outcome['job']
    need(outcome['known_terminal'] is True and job['returncode']==1 and job['timed_out'] is False
        and job['WindowsJobObject_limits_verified'] is True and job['worker_CPU_count']==2
        and job['job_commit_limit_bytes']==2147483648,'Original child not known terminated with original Job')
    old_source=git('show',binding['publication']+':'+binding['once_relative'])
    need(sha(old_source)==binding['once_SHA256'],'Original source control-flow pin differs')
    text=old_source.decode('utf-8');trace=(old/'worker.stderr').read_text(encoding='utf-8')
    need('line 25, in worker' in trace and trace.rstrip().endswith('ValueError: Actual Job/epoch differs'),'Original failure location differs')
    positions=[text.index(x) for x in ["gate=js((directory/'job-gate.json').read_bytes())",'need(deadline-epoch==120',"(directory/'worker-enter.lock')",'result=dict(passed=False','raw=get_once(']]
    need(positions==sorted(positions),'Original preGET failure ordering differs')
    public=binding.get('public_failure')
    need(isinstance(public,dict),'Published original failure evidence not bound')
    blob=git('show',public['publication']+':'+public['path'])
    need(sha(blob)==public['SHA256'] and len(blob)==public['bytes'],'Published original failure evidence changed')
    return dict(request=binding['request'],publication=binding['publication'],actual_GET_attempts=0,
        proof='Original known terminated Job + pinned traceback + exact published source gate-before-getter control flow',
        failure_binding_SHA256=sha((SOURCE/'ORIGINAL_PREGET_FAILURE.json').read_bytes()),public_failure_SHA256=public['SHA256'])


def controller(publication,claim_sha,root_go):
    need(root_go,'Root GO defaultfalse; prepared only')
    need(re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',claim_sha),'Bad source/claim pin')
    epoch=time.time();deadline=epoch+120;mono=time.monotonic()+120
    def git(*a):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),*a],cwd=G,timeout=min(10,max(.01,deadline-time.time()-8)))
    need(not git('status','--porcelain','--untracked-files=all','--',SOURCE.relative_to(G).as_posix()).strip() and not git('diff','--cached','--name-only').strip(),'Own source dirty or index nonempty')
    need(sha((SOURCE/'CLAIM.json').read_bytes())==claim_sha,'Claim changed')
    for name,pin in js((SOURCE/'MANIFEST.json').read_bytes())['files'].items():
        p=SOURCE/name;need(p.is_file() and not p.is_symlink(),'Missing/symlink source');raw=p.read_bytes()
        need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'] and git('show',publication+':'+p.relative_to(G).as_posix())==raw,'Published source differs')
    need(git('show',publication+':'+(SOURCE/'MANIFEST.json').relative_to(G).as_posix())==(SOURCE/'MANIFEST.json').read_bytes(),'Published MF differs')
    prior=original_preGET_failure(git)
    c=js((SOURCE/'CONTRACT.json').read_bytes());contract_gate(c);need(time.time()<deadline-30,'Epoch exhausted before original unique Job')
    RUN.mkdir();intent=dict(request=uuid.uuid4().hex,epoch=epoch,deadline=deadline,publication=publication,claim_SHA256=claim_sha,
        source_manifest_SHA256=sha((SOURCE/'MANIFEST.json').read_bytes()),URL=c['URL'],planned_GETs=1,body_cap_bytes=1048576,maximum_event_slots=50,CPU=2,job_commit_limit_bytes=2147483648,original_preGET_failure=prior)
    (RUN/'intent.json').write_bytes(canonical(intent))
    job=guarded_run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--worker','--run',str(RUN),'--epoch',str(epoch),'--deadline',str(deadline)],RUN/'job-gate.json',RUN,deadline,mono)
    (RUN/'outcome.json').write_bytes(canonical(dict(**intent,known_terminal=True,job=job,Dataset_GO=False,training_GO=False,GPU_GO=False)))
    print(json.dumps(dict(request=intent['request'],known_terminal=True,returncode=job['returncode'],timed_out=job['timed_out'])))
    return job['returncode']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--epoch',type=float);p.add_argument('--deadline',type=float)
    p.add_argument('--publication');p.add_argument('--claim-sha256');p.add_argument('--root-go',action='store_true');a=p.parse_args()
    raise SystemExit(worker(a.run,a.epoch,a.deadline) if a.worker else controller(a.publication,a.claim_sha256,a.root_go))
