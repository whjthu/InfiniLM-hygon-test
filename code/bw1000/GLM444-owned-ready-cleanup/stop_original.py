"""Bounded cleanup of eight pinned, zero-client, startup-stalled residencies.

Run once on the original CPU keeper node. Never starts a Model or edits source.
"""
from pathlib import Path
import datetime,hashlib,json,os,re,signal,socket,subprocess,time,traceback

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/GLM-W4-fewshot-quality-r444-eight-slot-v1/deployment-v2'
MF='b56fc770fa2e4a466a732718d51dfe734ba17bc0e8f70e9fb33f8e427b498833'
NATIVE='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
GPU=['910376.873','910374.1387','910371.781','910371.783','910380.580','910380.582','910376.875','910374.1389']
CPU=['910407.'+str(i)for i in range(928,936)]
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
P=R/'results/GLM444-owned-ready-cleanup-20261104-a.json'
state=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU_submitted=0,planned_HTTP=512,actual_HTTP_before_abort=0,reason='Loaded16ownerready without public MODEL_READY or client; no legal in-place recovery identified',phase='PREFLIGHT',slots=[],signals=[],STOP_committed=False,completed=False)
frozen=[];end=time.monotonic()+165

def save():
    temp=P.with_suffix('.writing-'+str(os.getpid()));temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(P)
def call(argv,limit=5):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=min(limit,max(.5,end-time.monotonic())))
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def accounts():
    p=call(['sacct','-S','2026-11-03','-j','910371,910374,910376,910380,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'])
    assert p['returncode']==0,p
    return {x.split('|')[0]:x.split('|')for x in p['stdout'].splitlines()if'|'in x}
def process(pid,slot):
    p=Path('/proc')/str(pid);assert p.stat().st_uid==os.getuid()
    args=[a.decode()for a in(p/'cmdline').read_bytes().split(b'\0')if a]
    assert str(D/'controller.py')in args and args[args.index('--slot')+1]==str(slot)
    assert args[args.index('--manifest-sha')+1]==MF
    stat=(p/'stat').read_text().rsplit(') ',1)[1].split()
    return dict(PID=pid,start_ticks=stat[19],process_state=stat[0],argv=args)
def same(q,slot):
    actual=process(q['PID'],slot);assert actual['start_ticks']==q['start_ticks'];return actual
def killstep(steps,kind):
    assert all(s in GPU and re.fullmatch(r'910(?:371|374|376|380)[.]\d+',s)for s in steps)
    if steps:state['signals'].append(call(['scancel','--signal='+kind,*steps]));save()
def running(rows):
    return [s for s in GPU if s not in rows or rows[s][2].split()[0].rstrip('+')not in TERMINAL]
def atomic_stop(folder,identity):
    p=folder/'STOP.json';payload=dict(identity=identity,normal_stop=True)
    if p.exists():assert json.loads(p.read_bytes())==payload;return
    temp=p.with_name('STOP.json.writing-root-'+str(os.getpid()))
    with temp.open('x')as f:json.dump(payload,f,indent=2);f.flush();os.fsync(f.fileno())
    try:os.link(temp,p)
    finally:temp.unlink()

def interrupted(signum,frame):
    raise TimeoutError('Owned cleanup coordinator interrupted by signal '+str(signum))
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)

assert socket.gethostname().split('.')[0]=='a14r2n20'
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==MF
with P.open('x')as f:json.dump(state,f)
try:
    rows=accounts()
    for slot in range(8):
        run='fewshot-20261104-a-s'+str(slot);label='GLM-r444-fewshot-'+run
        assert rows[GPU[slot]][1:4]==[label,'RUNNING','0:0']
        assert rows[CPU[slot]][1:4]==[label+'-keeper','RUNNING','0:0']
        assert not any(r[1]==label+'-client'for r in rows.values()),'Client already submitted'
        w=json.loads((R/'results'/(label+'-watch.json')).read_bytes())
        assert w['actual_GPU']==GPU[slot]and w['CPU_keeper']==CPU[slot]and not w['completed']and not w['actual_client']and not w['forced_cleanup']
        folder=R/'results/GLM-W4-fewshot-quality-r444'/run
        assert not(folder/'MODEL_READY.json').exists()and not(folder/'FAILED.json').exists()
        identity=json.loads((folder/'FRONTEND.json').read_bytes())['identity']
        assert identity['actual_step']==GPU[slot]and identity['run']==run and identity['gpu_start_epoch']==w['original_GPU_start_unix']and identity['source_manifest_SHA256']=='1183b8e6166afaefc319a528e6a0f7ce95a0a22a8b4215adf597c2912866df68'
        for stage in range(2):
            for rank in range(8):
                ready=json.loads((folder/('ready-%d-%d.json'%(stage,rank))).read_bytes())
                assert ready['identity']==identity and ready['Native_SHA256']==NATIVE and ready['loaded']and ready['postload_4GiB']and ready['service_agreement']
        query=call(['scontrol','listpids',CPU[slot]],3);assert query['returncode']==0,query
        candidates=[]
        for line in query['stdout'].splitlines():
            cells=line.split()
            if not cells or not cells[0].isdigit():continue
            try:candidates.append(process(int(cells[0]),slot))
            except (OSError,AssertionError,ValueError):pass
        assert len(candidates)==1,(slot,query,candidates)
        state['slots'].append(dict(slot=slot,GPU=GPU[slot],keeper=CPU[slot],folder=str(folder),identity=identity,controller=candidates[0],listpids=query))
    state['phase']='PREFLIGHT_PASSED';save()
    for row in state['slots']:
        same(row['controller'],row['slot']);frozen.append(row)
        os.kill(row['controller']['PID'],signal.SIGSTOP)
        state['signals'].append(dict(scope='ExactkeepercontrollerPIDonly',PID=row['controller']['PID'],signal='STOP'));save()
    time.sleep(.2)
    for row in frozen:
        assert same(row['controller'],row['slot'])['process_state']in('T','t')
        assert not(Path(row['folder'])/'MODEL_READY.json').exists()
    state['phase']='ALL_KEEPERS_FROZEN';save()
    # Commit before first STOP so interruption cannot implicitly resume admission.
    state['STOP_committed']=True;save()
    for row in frozen:atomic_stop(Path(row['folder']),row['identity'])
    state['phase']='ORIGINAL_PROTOCOL_STOP_SENT';save()
    until=min(time.monotonic()+90,end-45)
    while time.monotonic()<until:
        rows=accounts();state['last_GPU_accounting']={s:rows.get(s)for s in GPU};save()
        if not running(rows):break
        time.sleep(2)
    for kind,seconds in [('TERM',15),('KILL',15)]:
        pending=running(rows)
        if not pending:break
        killstep(pending,kind);until=min(time.monotonic()+seconds,end-12)
        while time.monotonic()<until:
            rows=accounts()
            if not running(rows):break
            time.sleep(1)
    state['phase']='GPU_DRAIN_OR_FAILSTOP_OBSERVED';state['last_GPU_accounting']={s:rows.get(s)for s in GPU};save()
except BaseException as error:
    state.update(error=repr(error),traceback=traceback.format_exc())
    if state['STOP_committed']:
        try:killstep(GPU,'KILL')
        except BaseException as cleanup:state['GPU_failstop_error']=repr(cleanup)
finally:
    for row in frozen:
        try:
            same(row['controller'],row['slot'])
            kind=signal.SIGKILL if state['STOP_committed']else signal.SIGCONT
            os.kill(row['controller']['PID'],kind)
            state['signals'].append(dict(scope='ExactkeepercontrollerPIDonly',PID=row['controller']['PID'],signal='KILL'if state['STOP_committed']else'CONT',reason='No late client launch after STOP'if state['STOP_committed']else'Uncommitted preflight rollback'))
        except ProcessLookupError:pass
        except BaseException as error:state.setdefault('keeper_signal_errors',[]).append(repr(error))
    state['phase']='COORDINATOR_FINISHED_POST_STILL_REQUIRED';state['completed']=True
    try:state['final_accounting']=accounts()
    except BaseException as error:state['final_accounting_error']=repr(error)
    state['end_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(dict(watch=str(P),completed=state['completed'],STOP_committed=state['STOP_committed'],error=state.get('error'),phase=state['phase'],GPU_submitted=0,post_required=True)))
