"""One existing draft-node residency; same bounded Slurm pattern as actual601."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,socket,subprocess,sys,time

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=Path(__file__).resolve().parent
LABEL='Kimi-long-draft-memory-floor-r362';NODE='a02r3n15'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(g):
    for k,v in dict(cards=1,target_cards=0,GPU_max_s=600,CPU_max_s=900,append_prompt=64,queries=1,CPUs=8,mem_MiB=65536).items():assert type(g[k])is int and g[k]==v
    assert g['GPU_parent']==910371 and g['GPU_node']==NODE and g['CPU_parent']==910407
    assert g['execution_selected']is True and g['one_attempt']is True and g['synthetic_cost_only']is True
    assert g['profile']is False and g['prompt_tokens']==32760
    assert re.fullmatch('[0-9a-f]{40}',g['publication_commit'])and re.fullmatch('[a-z0-9-]+',g['run'])
    assert sha(D/'SOURCE_MANIFEST.json')==g['source_manifest_SHA256']
    m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
    assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
    assert sha(R/'incoming/resource_wave_local_r3097_outer.sh')==m['outer_wrapper_SHA256']
    sys.path.insert(0,str(D));import worker
    worker.verify_sources(D,R,g['source_manifest_SHA256'])
    p=R/'candidates/Kimi-long-draft-memory-floor-r362/CPU-result.json';assert sha(p)==g['CPU_receipt_SHA256']
    q=json.loads(p.read_bytes());assert q['completed']and q['passed']and q['GPU']==0 and not q['Native_dlopen']
    assert q['source_manifest_SHA256']==g['source_manifest_SHA256']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execution-go',required=True);g=json.loads(Path(parser.parse_args().execution_go).read_bytes());verify(g)
    assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
    env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
    W=R/'results'/('Kimi-r362-watch-'+g['run']+'.json');assert not W.exists()
    q=dict(completed=False,passed=False,GPU_submitted=False,run=g['run'],CPU_keeper_actual=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
           source_manifest_SHA256=g['source_manifest_SHA256'],publication_commit=g['publication_commit'],target_cards=0,retry=False,reload=False)
    def save():
        q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(W)
    def rows():
        return [x.split('|')for x in subprocess.check_output(['sacct','-S','2026-11-01','-j','910371','-n','-P','-o','JobIDRaw,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()]
    def idle(suffix):
        cmd=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=256M','-w',NODE,'-t','00:01:00','--job-name='+LABEL+'-'+suffix,'python3',str(D/'idle.py')]
        p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=100);out=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')]
        assert p.returncode==0 and len(out)==1 and out[0]['node']==NODE
        return out[0]
    def no_foreign(ignore=()):
        for row in rows():
            if re.fullmatch(r'910371[.]\d+',row[0])and row[0]not in ignore and row[2].split()[0].rstrip('+')not in TERMINAL:
                nodes=subprocess.check_output(['scontrol','show','hostnames',row[4]],env=env,text=True,timeout=15).splitlines()
                assert NODE not in nodes,'Foreign active step on reserved draft node'
    process=None;save()
    try:
        raw=subprocess.check_output(['scontrol','show','job','910371','-o'],env=env,text=True,timeout=25)
        assert 'UserId=aclsylqidf('in raw and 'JobState=RUNNING'in raw
        nodes=re.search(r'(?:^| )NodeList=([^ ]+)',raw).group(1)
        assert NODE in subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines()
        assert not any(row[1]==LABEL for row in rows()),'Existing once-only label; follow it'
        no_foreign();q['fresh_before']=idle('before');assert q['fresh_before']['passed'];save()
        until=time.monotonic()+30
        while True:
            seen=[row for row in rows()if row[0]==q['fresh_before']['actual_step']]
            if seen and seen[0][2:4]==['COMPLETED','0:0']:break
            assert time.monotonic()<until,'Own CPU-before terminal unresolved';time.sleep(1)
        no_foreign({q['fresh_before']['actual_step']});epoch=time.time();q['original_GPU_start_unix']=epoch;save()
        cmd=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c8','--gres=dcu:1','--mem=65536M','-w',NODE,'-t','00:10:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'--export=ALL,WAVE_INSIDE_R2323='+D.name+'/GPU_inside.sh,KIMI_R362_MANIFEST='+g['source_manifest_SHA256']+',KIMI_R362_RUN='+g['run']+',KIMI_R362_DEADLINE='+str(epoch+570),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
        with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:process=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        q.update(GPU_submitted=True,srun_PID=process.pid);save();end=time.monotonic()+660
        while time.monotonic()<end:
            found=[row for row in rows()if row[1]==LABEL];assert len(found)<=1
            if found:
                row=found[0];assert row[4]==NODE and re.fullmatch(r'910371[.]\d+',row[0]);assert q.get('actual_GPU',row[0])==row[0]
                q.update(actual_GPU=row[0],last_Slurm=row);save()
                if row[2].split()[0].rstrip('+')in TERMINAL:q['terminal']=row;break
            time.sleep(10)
        assert 'terminal'in q,'Original handle terminal unresolved; no replay'
        p=R/'results'/('Kimi-r362-pressure-'+g['run']+'.json');until=time.monotonic()+30
        while not p.exists()and time.monotonic()<until:time.sleep(1)
        if p.exists():q.update(report_path=str(p),report_SHA256=sha(p),report=json.loads(p.read_bytes()))
        q['post']=idle('post');q['post_empty']=q['post']['passed']
        report=q.get('report',{});q['passed']=q['terminal'][2:4]==['COMPLETED','0:0']and report.get('passed',False)and q['post_empty']
    except BaseException as error:
        q['error']=repr(error)
        if process is not None and process.poll()is None and q.get('actual_GPU'):
            subprocess.run(['scancel','--signal=TERM',q['actual_GPU']],env=env,timeout=25,check=True)
        raise
    finally:q['completed']=True;save()
    raise SystemExit(0 if q['passed']else 1)


if __name__=='__main__':main()
