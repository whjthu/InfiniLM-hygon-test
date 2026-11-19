"""One bounded offline re-audit and input-window declaration. No HTTP or labels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from prepare import need,sha,canonical,parse,select
from job_guard import guarded_run,current_affinity

S=Path(__file__).resolve().parent;G=S.parents[2];RUN=G/'local-only/training-USGS-multievent-offline-v3-original'

def save(name,value):
    with (RUN/name).open('xb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())

def stable(spec):
    path=G/spec['path'];before=path.lstat()
    need(path.is_file() and not path.is_symlink() and 0<before.st_size==spec['bytes']<=4*1024**2,'exact cached metadata/raw bounds')
    raw=path.read_bytes();after=path.lstat()
    need(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')) and sha(raw)==spec['SHA256'],'original cached file changed')
    return raw

def verify():
    m=json.loads((S/'SOURCE_FILES.json').read_bytes())
    need(set(m['files'])=={f.name for f in S.iterdir() if f.is_file() and f.name!='SOURCE_FILES.json'},'exact offline source inventory')
    for name,pin in m['files'].items():
        need(Path(name).name==name,'flat offline source member');raw=(S/name).read_bytes()
        need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'offline source pin differs')

def worker(epoch):
    deadline=epoch+90;mono=time.monotonic()+deadline-time.time()
    def left():
        value=min(deadline-time.time(),mono-time.monotonic())-8;need(value>0,'single offline90 epoch exhausted');return value
    while not (RUN/'job-gate.json').exists():left();time.sleep(.02)
    gate=json.loads((RUN/'job-gate.json').read_bytes())
    need(gate['deadline']==epoch+85 and gate['memory_limit_bytes']==2*1024**3 and gate['CPU_count']==2 and gate['job_affinity_verified'] is True
         and gate['worker_affinity_mask']==current_affinity()['process_mask'],'actual offline Job/clock/CPU differs')
    (RUN/'worker-enter.lock').open('x').close();verify();p=json.loads((S/'PLAN.json').read_bytes())
    result=dict(passed=False,epoch=epoch,deadline=deadline,source_event_denominator=711,numeric_qualified=None,invalid_source_slots=None,
        planned_input_windows=p['chart_slots'],selected_input_windows={'train':0,'dev':0,'blind':0},missing_input_windows=dict(p['chart_slots']),
        selected_group_counts={'train':0,'dev':0,'blind':0},actual_HTTP_calls=0,actual_QA=0,actual_gold=0,actual_images=0,actual_native_calls=0,
        actual_model_calls=0,Dataset_GO=False,training_GO=False,GPU_GO=False,Model_GO=False,original_v2_failure_retained=True)
    def no_external(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):raise RuntimeError('offline source worker external operation forbidden')
    sys.addaudithook(no_external)
    try:
        need(p['root_GO'] is p['CPU_GO'] is True and p['maximum_GETs']==0 and p['source_GET_GO'] is False,'offline-only adoption')
        raw=stable(p['source_snapshot']);left();rows,audit=parse(raw,p)
        audit['v2_preferred_net_gate_reclassification']=[dict(slot=i,qualified_in_v3=any(r['slot']==i for r in rows)) for i in [30,38,261,433,587,624,634,677]]
        audit['preferred_network_counts']={key:sum(r['preferred_net']==key for r in rows) for key in sorted({r['preferred_net'] for r in rows})}
        save('source-audit-private.json',audit)
        result.update(numeric_qualified=audit['numeric_qualified'],invalid_source_slots=len(audit['rejected']))
        need(audit['source_features']==711 and audit['numeric_qualified']+len(audit['rejected'])==711,'full711 source denominator')
        if audit['rejected']:
            result['state']='SOURCE_INVALID_SLOTS_STOP_NO_SUBSTITUTION'
        else:
            old=[]
            for spec in p['old_sources']:old+=json.loads(stable(spec))['rows'];left()
            input_count=len(old);need(input_count==750,'three old metadata source denominator')
            old=list({canonical({k:r[k] for k in ('id','aliases','time_UTC_epoch_ms','values')}):r for r in old}.values())
            selected=select(old,rows,p,left);left()
            declaration=dict(schema='FROZEN_PRE_QA_INPUT_SELECTION_V3',source_snapshot=p['source_snapshot'],seed=p['seed'],old_sources=p['old_sources'],
                group_assignment=selected['capacities'],planned_input_windows=p['chart_slots'],selected_input_windows=selected['selected_charts'],
                missing_input_windows=selected['missing_charts'],selected_group_counts=selected['selected_group_counts'],DP=selected['DP'],
                windows=[dict(id=c['id'],group=c['group'],split=c['split'],event_ids=[r['id'] for r in c['rows']],input_rows_SHA256=sha(canonical(c['rows']))) for c in selected['charts']],
                QA_gold_images_present=False,passed=selected['passed'])
            # Declaration is durably sealed BEFORE writing the private numerical windows.
            save('input-selection-declaration-private.json',declaration);left()
            declaration_pin=dict(bytes=(RUN/'input-selection-declaration-private.json').stat().st_size,SHA256=sha((RUN/'input-selection-declaration-private.json').read_bytes()))
            save('input-windows-private.json',selected);left()
            result.update(passed=selected['passed'],state=selected['state'],selected_input_windows=selected['selected_charts'],missing_input_windows=selected['missing_charts'],
                selected_group_counts=selected['selected_group_counts'],new_closure_groups=selected['new_internal_closure_groups'],eligible_window_groups=selected['eligible_window_groups'],
                DP=selected['DP'],old_input_metadata_records=input_count,old_unique_snapshots=len(old),excluded_old_closure_events=selected['excluded_old_closure_events'],
                selected_declaration=declaration_pin,selected_windows_SHA256=sha((RUN/'input-windows-private.json').read_bytes()),singleton_windows=0,constant_displayed_columns_windows=0)
    except BaseException as error:
        result.update(state='KNOWN_OFFLINE_FAILURE_NO_RETRY_OR_REFILL',error_type=type(error).__name__,private_reason=str(error)[:512])
    result['finished_epoch']=time.time();save('worker-result.json',result);return 0 if result['passed'] else 1

def controller(publication,claim_sha,root_go):
    epoch=time.time();deadline=epoch+90;mono=time.monotonic()+90
    def left():
        value=min(deadline-time.time(),mono-time.monotonic())-8;need(value>0,'original offline90 exhausted');return value
    need(root_go and len(publication)==40 and len(claim_sha)==64,'root published invocation required');verify()
    claim=json.loads((S/'CLAIM.json').read_bytes());p=json.loads((S/'PLAN.json').read_bytes())
    need(claim['root_GO'] is claim['CPU_GO'] is True and claim['source_GET_GO'] is False and sha((S/'CLAIM.json').read_bytes())==claim_sha,'offline published claim required')
    def git(*a):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*a],timeout=min(10,left()))
    need(git('cat-file','blob',publication+':'+(S/'SOURCE_FILES.json').relative_to(G).as_posix())==(S/'SOURCE_FILES.json').read_bytes(),'published offline MF differs')
    prior=p['prior_failure'];blob=git('cat-file','blob',publication+':'+prior['path'])
    need(len(blob)==prior['bytes'] and sha(blob)==prior['SHA256'],'published original v2 failure prerequisite differs')
    need(not git('status','--porcelain=v1','--untracked-files=all','--',S.relative_to(G).as_posix()).strip(),'offline source dirty')
    RUN.mkdir(mode=0o700);intent=dict(request_id=uuid.uuid4().hex,publication=publication,claim_SHA256=claim_sha,source_manifest_SHA256=sha((S/'SOURCE_FILES.json').read_bytes()),
        epoch=epoch,deadline=deadline,planned_HTTP=0,source_event_denominator=711,planned_input_windows=p['chart_slots'],prior_failure=prior)
    save('intent.json',intent)
    try:job=guarded_run([sys.executable,'-B','-X','utf8',str(S/'once.py'),'--worker','--epoch',str(epoch)],RUN/'job-gate.json',RUN,epoch+85,mono-5)
    except BaseException as error:
        save('outcome.json',dict(**intent,passed=False,known_local_terminal=False,job_error_type=type(error).__name__,state='UNKNOWN_JOB_OUTCOME_NO_REPLAY'));return 1
    path=RUN/'worker-result.json';wr=json.loads(path.read_bytes()) if path.exists() else None
    passed=bool(job['returncode']==0 and wr and wr['passed'])
    save('outcome.json',dict(**intent,passed=passed,known_local_terminal=True,job=job,worker_result_SHA256=sha(path.read_bytes()) if path.exists() else None,
        actual_HTTP_calls=0,actual_model_calls=0,actual_QA_gold_images=0,replay_allowed=False,state='LOCAL_TERMINAL_INPUT_ONLY' if wr else 'LOCAL_TERMINAL_WORKER_RESULT_MISSING_NO_REPLAY'))
    print(json.dumps(dict(request_id=intent['request_id'],passed=passed,returncode=job['returncode'],known_local_terminal=True,actual_HTTP_calls=0)));return 0 if passed else 1

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--worker',action='store_true');q.add_argument('--epoch',type=float);q.add_argument('--publication');q.add_argument('--claim-sha256');q.add_argument('--root-go',action='store_true');a=q.parse_args()
    raise SystemExit(worker(a.epoch) if a.worker else controller(a.publication,a.claim_sha256,a.root_go))
