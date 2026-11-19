"""One original120 epoch for qualified-environment build and separate independent audit."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from contracts import need,canonical,sha
from job_guard import guarded_run,current_affinity

S=Path(__file__).resolve().parent;G=S.parents[2];RUN=G/'local-only/training-USGS-multievent-render144-v1-original'
def save(name,value):
    with (RUN/name).open('xb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
def verify():
    m=json.loads((S/'SOURCE_FILES.json').read_bytes())
    need(set(m['files'])=={p.name for p in S.iterdir() if p.is_file() and p.name!='SOURCE_FILES.json'},'exact source inventory')
    for n,pin in m['files'].items():
        p=S/n;raw=p.read_bytes();need(Path(n).name==n and not p.is_symlink() and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'source bytes changed')
    return m
def worker(epoch,phase,phase_deadline):
    need(math.isfinite(epoch) and epoch<=time.time()<epoch+120 and phase in ('build','audit'),'original finite120 epoch')
    expected=epoch+(95 if phase=='build' else 115);need(phase_deadline==expected,'same-origin phase deadline')
    mono=time.monotonic()+phase_deadline-time.time()
    def left():
        n=min(phase_deadline-time.time(),mono-time.monotonic())-8;need(n>0,'original phase work reserve exhausted');return n
    gate=RUN/(phase+'-job-gate.json')
    while not gate.exists():left();time.sleep(.02)
    v=json.loads(gate.read_bytes());need(v['deadline']==expected and v['memory_limit_bytes']==2*1024**3 and v['CPU_count']==2
        and v['job_affinity_verified'] is True and v['worker_affinity_mask']==current_affinity()['process_mask'],'actual phase Job/CPU/epoch not qualified')
    (RUN/(phase+'-enter.lock')).open('x').close();verify();plan=json.loads((S/'PLAN.json').read_bytes())
    result=dict(passed=False,phase=phase,epoch=epoch,deadline=epoch+120,planned_images=24,planned_QA=144,planned_split_QA=plan['planned_QA'],
        HTTP=0,actual_native_calls=0,actual_model_calls=0,actual_training_calls=0,Dataset_GO=False,GPU_GO=False,Model_GO=False)
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):raise RuntimeError('offline data worker external operation forbidden')
    sys.addaudithook(offline)
    try:
        need(plan['root_GO'] is plan['CPU_GO'] is True and (phase!='audit' or plan['READ_GO'] is True),'phase not adopted')
        if phase=='build':
            from build import run
            result.update(run(RUN,plan,left))
        else:
            from audit import check
            result.update(check(RUN,plan,left))
    except BaseException as e:result.update(state='KNOWN_DATA_PHASE_FAILURE_NO_REPLAY',error_type=type(e).__name__,private_reason=str(e)[:512])
    result['finished_epoch']=time.time();save(phase+'-result.json',result);return 0 if result['passed'] else 1
def controller(pub,claim_sha,root_go,origin):
    epoch=time.time() if origin is None else origin;need(math.isfinite(epoch) and 0<epoch<=time.time()<epoch+120,'original120 origin')
    deadline=epoch+120;mono=time.monotonic()+deadline-time.time()
    def left():
        n=min(deadline-time.time(),mono-time.monotonic())-5;need(n>0,'original120 exhausted');return n
    need(root_go and len(pub)==40 and len(claim_sha)==64,'root published invocation');verify()
    c=json.loads((S/'CLAIM.json').read_bytes());p=json.loads((S/'PLAN.json').read_bytes())
    need(c['root_GO'] is c['CPU_GO'] is c['READ_GO'] is True and sha((S/'CLAIM.json').read_bytes())==claim_sha,'build/read claim not adopted')
    def git(*a):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*a],timeout=min(10,left()))
    need(git('cat-file','blob',pub+':'+(S/'SOURCE_FILES.json').relative_to(G).as_posix())==(S/'SOURCE_FILES.json').read_bytes(),'published MF differs')
    raw=git('cat-file','blob',pub+':'+p['prerequisite']['path']);need(len(raw)==p['prerequisite']['bytes'] and sha(raw)==p['prerequisite']['SHA256'],'actual input qualification prerequisite differs')
    need(not git('status','--porcelain=v1','--untracked-files=all','--',S.relative_to(G).as_posix()).strip(),'own source dirty')
    RUN.mkdir(mode=0o700);intent=dict(request_id=uuid.uuid4().hex,publication=pub,claim_SHA256=claim_sha,source_manifest_SHA256=sha((S/'SOURCE_FILES.json').read_bytes()),epoch=epoch,deadline=deadline,
        source_pins=p['input_pins'],planned_images=24,planned_QA=144,planned_split_QA=p['planned_QA'],HTTP=0,actual_training_calls=0)
    save('intent.json',intent);jobs={};passed=False;error=None
    try:
        for phase,offset in [('build',95),('audit',115)]:
            left();need(time.time()<epoch+offset-12,'insufficient original phase budget')
            directory=RUN/(phase+'-job');directory.mkdir()
            argv=[sys.executable,'-B','-X','utf8',str(S/'once.py'),'--worker','--phase',phase,'--epoch',str(epoch),'--phase-deadline',str(epoch+offset)]
            job=guarded_run(argv,RUN/(phase+'-job-gate.json'),directory,epoch+offset,mono-(120-offset));jobs[phase]=job
            save(phase+'-job-result.json',job)
            need(job['returncode']==0 and job['timed_out'] is False,'data phase failed; no rerun')
            v=json.loads((RUN/(phase+'-result.json')).read_bytes());need(v['passed'] is True,'data phase result failed')
        passed=True
    except BaseException as e:error=dict(type=type(e).__name__,private_reason=str(e)[:256])
    save('outcome.json',dict(**intent,passed=passed,jobs=jobs,error=error,producer_result_SHA256=sha((RUN/'build-result.json').read_bytes()) if (RUN/'build-result.json').exists() else None,
        independent_audit_result_SHA256=sha((RUN/'audit-result.json').read_bytes()) if (RUN/'audit-result.json').exists() else None,
        actual_native_calls=0,actual_model_calls=0,Model_GO=False,training_GO=False,replay_allowed=False))
    print(json.dumps(dict(request_id=intent['request_id'],passed=passed,HTTP=0,planned_images=24,planned_QA=144)));return 0 if passed else 1
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--worker',action='store_true');q.add_argument('--phase',choices=['build','audit']);q.add_argument('--epoch',type=float);q.add_argument('--phase-deadline',type=float)
    q.add_argument('--publication');q.add_argument('--claim-sha256');q.add_argument('--root-go',action='store_true');q.add_argument('--origin',type=float);a=q.parse_args()
    raise SystemExit(worker(a.epoch,a.phase,a.phase_deadline) if a.worker else controller(a.publication,a.claim_sha256,a.root_go,a.origin))
