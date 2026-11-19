"""Unique 90-second WindowsJob source acquisition/parse/selection, never training."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from prepare import need,canonical,sha,parse,select
from job_guard import guarded_run,current_affinity

S=Path(__file__).resolve().parent;G=S.parents[2];RUN=G/'local-only'/('training-USGS-multievent-source-v2-original')

def save(name,value):
    with (RUN/name).open('xb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())

def verify():
    manifest=json.loads((S/'SOURCE_FILES.json').read_bytes())
    for name,pin in manifest['files'].items():
        need(Path(name).name==name,'exact flat source member')
        p=S/name;raw=p.read_bytes();need(not p.is_symlink() and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'source changed')
    return manifest

def worker(epoch):
    deadline=epoch+90;mono=time.monotonic()+deadline-time.time()
    def left():
        value=min(deadline-time.time(),mono-time.monotonic())-8;need(value>0,'original90 epoch exhausted');return value
    while not (RUN/'job-gate.json').exists():left();time.sleep(.02)
    gate=json.loads((RUN/'job-gate.json').read_bytes())
    need(gate['deadline']==epoch+85 and gate['memory_limit_bytes']==2*1024**3 and gate['CPU_count']==2 and gate['job_affinity_verified'] is True
         and gate['worker_affinity_mask']==current_affinity()['process_mask'],'actual Job/epoch/2CPU gate differs')
    (RUN/'worker-enter.lock').open('x').close();verify();p=json.loads((S/'PLAN.json').read_bytes())
    result=dict(passed=False,epoch=epoch,deadline=deadline,planned_GETs=1,actual_GETs=None,source_event_count=None,
        planned_charts=p['chart_slots'],planned_future_QA=144,planned_future_QA_per_split={'train':72,'dev':36,'blind':36},
        selected_charts={'train':0,'dev':0,'blind':0},missing_charts=dict(p['chart_slots']),selected_group_counts={'train':0,'dev':0,'blind':0},
        actual_images=0,actual_QA=0,actual_native_calls=0,actual_model_calls=0,training_GO=False,Dataset_GO=False,GPU_GO=False,Model_GO=False)
    try:
        need(p['root_GO'] is p['CPU_GO'] is p['source_GET_GO'] is True,'data source scope not adopted')
        old=[]
        for spec in p['old_sources']:
            need(type(spec['bytes'])is int and type(spec['SHA256'])is str and len(spec['SHA256'])==64,'old event-metadata pin missing before GET')
            q=G/spec['path'];before=q.stat();need(not q.is_symlink() and before.st_size==spec['bytes']<=4*1024**2,'old metadata bounds')
            body=q.read_bytes();after=q.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and sha(body)==spec['SHA256'],'old metadata identity changed')
            old+=json.loads(body)['rows'];left()
        old_input_records=len(old)
        # Exact duplicated event snapshots collapse; differing historical snapshots remain
        # witnesses and are transitively joined by their shared aliases, never discarded.
        old=list({canonical({k:r[k] for k in ('id','aliases','time_UTC_epoch_ms','values')}):r for r in old}.values())
        result.update(old_input_metadata_records=old_input_records,old_unique_snapshots=len(old),old_unique_IDs=len({r['id'] for r in old}))
        from http_once import get_once
        raw,http=get_once(RUN,p,left);result.update(actual_GETs=1,source_complete=True,source_SHA256=sha(raw),HTTP=http)
        left();rows,audit=parse(raw,p);result['source_event_count']=audit['source_features']
        save('source-audit-private.json',audit);left();selection=select(old,rows,p);left()
        if audit['rejected']:selection['passed']=False;selection['state']='SOURCE_INVALID_SLOTS_STOP_NO_SUBSTITUTION'
        save('selection-private.json',selection)
        result.update(passed=selection['passed'],state=selection['state'],numeric_qualified=audit['numeric_qualified'],source_rejected=len(audit['rejected']),
            selected_charts=selection['selected_charts'],missing_charts=selection['missing_charts'],selected_group_counts=selection['selected_group_counts'],
            new_closure_groups=selection['new_internal_closure_groups'],excluded_old_events=selection['excluded_old_closure_events'],singleton_charts=0,constant_display_charts=0)
    except BaseException as error:
        result.update(state='KNOWN_LOCAL_FAILURE_NO_RETRY_OR_REFILL',error_type=type(error).__name__,private_reason=str(error)[:512])
        hp=RUN/'http-receipt-private.json'
        if hp.exists():
            h=json.loads(hp.read_bytes());result.update(actual_GETs=int(h['attempted']),HTTP=h,source_complete=h['complete'])
        else:result.update(actual_GETs=0,source_complete=False)
    result['finished_epoch']=time.time();save('worker-result.json',result);return 0 if result['passed'] else 1

def controller(pub,claim_sha,root_go):
    epoch=time.time();mono=time.monotonic()+90;deadline=epoch+90
    def left():
        value=min(deadline-time.time(),mono-time.monotonic())-8;need(value>0,'single origin90 exhausted');return value
    need(root_go and len(pub)==40 and len(claim_sha)==64,'published root invocation required');verify()
    c=json.loads((S/'CLAIM.json').read_bytes());need(c['root_GO'] is c['CPU_GO'] is c['source_GET_GO'] is True and sha((S/'CLAIM.json').read_bytes())==claim_sha,'published claim required')
    def git(*a):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*a],timeout=min(10,left()))
    need(git('cat-file','blob',pub+':'+(S/'SOURCE_FILES.json').relative_to(G).as_posix())==(S/'SOURCE_FILES.json').read_bytes(),'published MF differs')
    need(not git('status','--porcelain=v1','--untracked-files=all','--',S.relative_to(G).as_posix()).strip(),'source dirty')
    RUN.mkdir(mode=0o700);intent=dict(request_id=uuid.uuid4().hex,publication=pub,claim_SHA256=claim_sha,source_manifest_SHA256=sha((S/'SOURCE_FILES.json').read_bytes()),epoch=epoch,deadline=deadline,planned_GETs=1,planned_charts={'train':12,'dev':6,'blind':6},planned_future_QA=144)
    save('intent.json',intent)
    try:
        job=guarded_run([sys.executable,'-B','-X','utf8',str(S/'once.py'),'--worker','--epoch',str(epoch)],RUN/'job-gate.json',RUN,epoch+85,mono-5)
    except BaseException as error:
        save('outcome.json',dict(**intent,known_local_terminal=False,passed=False,job_error_type=type(error).__name__,
            source_outcome='UNKNOWN_JOB_FAILURE_REQUIRES_ORIGINAL_RECEIPTS_NO_REPLAY',actual_training_calls=0,actual_GPU_calls=0,replay_allowed=False))
        return 1
    path=RUN/'worker-result.json';wr=json.loads(path.read_bytes()) if path.exists() else None
    save('outcome.json',dict(**intent,known_local_terminal=True,job=job,worker_result_SHA256=sha(path.read_bytes()) if path.exists() else None,
        passed=job['returncode']==0 and bool(wr and wr['passed']),source_outcome='SEE_PERSISTED_WORKER_AND_HTTP_RECEIPT' if wr else 'UNKNOWN_AFTER_LOCAL_TERMINAL_REQUIRES_ORIGINAL_PARTIAL_RECONCILE',
        actual_training_calls=0,actual_GPU_calls=0,replay_allowed=False))
    print(json.dumps(dict(request_id=intent['request_id'],known_local_terminal=True,returncode=job['returncode'],passed=bool(wr and wr['passed']))));return 0 if job['returncode']==0 and wr and wr['passed'] else 1

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--worker',action='store_true');q.add_argument('--epoch',type=float);q.add_argument('--publication');q.add_argument('--claim-sha256');q.add_argument('--root-go',action='store_true');a=q.parse_args()
    raise SystemExit(worker(a.epoch) if a.worker else controller(a.publication,a.claim_sha256,a.root_go))
