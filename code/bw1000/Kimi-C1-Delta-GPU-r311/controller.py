"""One bounded isolated TP16 single-layer C1 profile; never repeats the GPU run."""
from pathlib import Path
import os,json,subprocess,hashlib,time,datetime,re,socket
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=Path(__file__).resolve().parent
LABEL='Kimi-C1-Delta-GPU-r311';NODES=['a14r2n08','a14r2n09'];W=R/'results'/(LABEL+'-watch.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert os.environ['SLURM_JOB_ID']=='910407' and socket.gethostname().split('.')[0]=='a14r2n20'
    assert not W.exists();m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());mh=sha(D/'SOURCE_MANIFEST.json')
    assert mh==os.environ['KIMI_R311_MANIFEST_SHA'] and all(sha(D/n)==h for n,h in m['files_SHA256'].items())
    cp=R/'candidates/Kimi-C1-Delta-profile-r309/CPU-build.json';assert sha(cp)==m['CPU_receipt_SHA256']
    c=json.loads(cp.read_bytes());assert c['completed'] and c['passed'] and c['CPU_only'] and not c['Native_dlopen']
    for key in ['native','Main']:assert sha(str(c[key]).replace('/refactor/',str(R)+'/',1))==c[key+'_SHA256']
    source=R/'incoming/Kimi-C1-Delta-profile-r309';assert sha(source/'SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256']
    sm=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(source/n)==h for n,h in sm['files_SHA256'].items())
    env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
    for parent in ['910380','910407']:
        text=subprocess.check_output(['scontrol','show','job',parent,'-o'],env=env,text=True,timeout=25)
        assert 'UserId=aclsylqidf('in text and 'JobState=RUNNING'in text
    q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'GPU_submitted':False,
       'actual_GPU':None,'CPU_keeper':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
       'source_manifest_SHA256':mh,'CPU_receipt_SHA256':sha(cp),'Model_gain':False,'one_attempt':True,
       'GPU_hard_s':600,'measurement_max_s':300,'scope':'Component C1T1 Delta profile only, not fullModel/live32K replay'}
    proc=None;actual=None
    def save():
        q['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=W.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(W)
    def accounting():
        return[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910380','-n','-P',
            '-o','JobID,JobName%120,State,ExitCode,NodeList%120'],env=env,text=True,timeout=25).splitlines()]
    def account():
        rows=[x for x in accounting()if x[1]==LABEL];assert len(rows)<=1
        if rows:
            row=rows[0];assert re.fullmatch(r'910380[.]\d+',row[0])
            nodes=subprocess.check_output(['scontrol','show','hostnames',row[4]],env=env,text=True,timeout=15).splitlines()
            assert set(nodes)==set(NODES) and len(nodes)==2;return row
    def no_foreign():
        for row in accounting():
            if re.fullmatch(r'910380[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:
                nodes=subprocess.check_output(['scontrol','show','hostnames',row[4]],env=env,text=True,timeout=15).splitlines()
                assert not(set(nodes)&set(NODES)),'An active step intersects target nodes'
    def probe(tag):
        args=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2',
          '--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00',
          '--job-name='+LABEL+'-'+tag,'python3',str(D/'resource_probe.py')]
        z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160)
        q.setdefault('probe_attempts',{})[tag]={'returncode':z.returncode,'stdout':z.stdout[-6000:],'stderr':z.stderr[-3000:]};save()
        rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')]
        assert z.returncode==0 and len(rows)==2 and {v['node']for v in rows}==set(NODES) and all(v['CPU_only'] and
          not v['GPU_runtime_initialized'] and v['KFD_empty'] and v['control30040_bindable']for v in rows)
        return rows
    save()
    try:
        assert account()is None;no_foreign();q['fresh_before']=probe('before');no_foreign();save()
        head=next(v['ib0_ipv4']for v in q['fresh_before']if v['node']==NODES[0]);nonce=os.urandom(16).hex()
        export='ALL,GLOBAL_MODEL_MODE_R2354=shca,GLOBAL_OFFSET_R2354=0,WAVE_INSIDE_R2323=Kimi-C1-Delta-GPU-r311/GPU_inside.sh,KIMI_ENGINE_HEAD='+head+',KIMI_ENGINE_NONCE='+nonce+',GLOBAL_PORT_R2354=23040,KIMI_R311_MANIFEST_SHA='+mh
        args=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1',
          '-N2','-n2','--ntasks-per-node=1','-c64','-w',','.join(NODES),'--gres=dcu:8','--mem=49152M',
          '-t','00:10:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J-%t.log')),
          '--export='+export,'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
        with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:
            proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        q.update(GPU_submitted=True,srun_client_PID=proc.pid,GPU_start_UTC_unix=time.time());save();deadline=time.monotonic()+700
        while time.monotonic()<deadline:
            row=account()
            if row:
                assert actual is None or actual==row[0];actual=row[0];q.update(actual_GPU=actual,last_Slurm=row)
                if row[2].split()[0].rstrip('+')in TERMINAL:q['terminal']=row;save();break
            if proc.poll()is not None:
                q.setdefault('srun_exit_seen',time.monotonic());q['srun_exit_code']=proc.poll()
                if time.monotonic()-q['srun_exit_seen']>60:raise RuntimeError('srun exited/accounting unresolved; follow same label, no retry')
            save();time.sleep(5)
        else:raise TimeoutError('Follow same bounded actual GPU to terminal; do not resubmit')
        q['post']=probe('post');q['post_empty']=True
        assert q['terminal'][2:4]==['COMPLETED','0:0'],'GPU failure; preserve cleanup without promotion'
        refs=[];ranks=[]
        for node in range(2):
            p=R/'results'/(LABEL+'-'+actual+'-node'+str(node)+'.json');v=json.loads(p.read_bytes())
            assert v['completed'] and v['normal_close'] and v['owners']==8 and v['actual_GPU']==actual and v['source_manifest_SHA256']==mh
            assert v['C']==v['T']==1 and v['pool']==3 and v['Math']==0 and v['capture']is False
            for owner in v['workers']:
                assert owner['completed'] and owner['normal_close'] and len(owner['samples'])==30 and 'error'not in owner
                for arm in ['OFF-before','ON','OFF-after']:
                    samples=[s for s in owner['samples']if s['arm']==arm];assert len(samples)==10 and sum(not s['warmup']for s in samples)==8
                ranks.append(owner['rank'])
            log=R/'logs'/(LABEL+'-'+actual+'-'+str(node)+'.log');text=log.read_text(errors='replace')
            assert text.count('KIMI_C1_DELTA_R311_NORMAL '+str(node))==1 and 'Using network IBext_v8'in text and 'Using network Socket'not in text
            refs.append({'node':node,'path':str(p),'SHA256':sha(p),'log_SHA256':sha(log)})
        assert sorted(ranks)==list(range(16));q.update(completed=True,state='COMPONENT_PROFILE_COMPLETE_PENDING_ROOT_AUDIT',owners=refs,component_lifecycle_qualified=True);save()
    except BaseException as e:
        q.update(error=repr(e),state='FAILED_BEFORE_GPU'if not q['GPU_submitted']else'FOLLOW_SAME_HANDLE_NO_RETRY',completed='terminal'in q);save()
        if 'terminal'in q and not q.get('post_empty'):
            try:q['error_post']=probe('errorpost');q['post_empty']=True;save()
            except BaseException as cleanup:q['cleanup_error']=repr(cleanup);save()
        raise

if __name__=='__main__':main()
