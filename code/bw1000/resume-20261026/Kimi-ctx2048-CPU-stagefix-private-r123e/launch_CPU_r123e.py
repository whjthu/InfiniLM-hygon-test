import pathlib,json,subprocess,os,time,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';src=r/'incoming/Kimi-ctx2048-r123e';m=json.loads((src/'CPU-stage-file-manifest-r123e.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in m['files_SHA256'].items():assert sha(src/n)==h,n
receipt=r/'results/Kimi-ctx2048-CPU-r123e-launch.json';assert not receipt.exists(),'CPU launch already recorded; inspect same unique label, do not reinvoke'
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']};env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
parent=subprocess.check_output(['scontrol','show','job','910407','-o'],env=env,text=True,timeout=20);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
spec=next(x.split('=',1)[1]for x in parent.split()if x.startswith('NodeList='));assert 'a14r2n20'in subprocess.check_output(['scontrol','show','hostnames',spec],env=env,text=True,timeout=20).splitlines()
rows=subprocess.check_output(['sacct','-S','2026-10-25','-j','910407','-n','-P','-o','JobIDRaw,JobName%100,State'],env=env,text=True,timeout=20).splitlines();assert not any(x.split('|')[1:2]==['Kimi-ctx2048-HOST-r123e']for x in rows),'existing label must be inspected, never duplicate'
outer=r/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce';assert (r/'containers/dtk26-sandbox-r1').is_dir()
cmd=['srun','--jobid=910407','-N1','-n1','-w','a14r2n20','-c8','--mem=32768M','--gres=none','--exact','--overlap','--cpu-bind=none','--time=00:15:00','--job-name=Kimi-ctx2048-HOST-r123e','--export=ALL,WAVE_INSIDE_R2323=Kimi-ctx2048-r123e/CPU_inside_r123e.sh,GLOBAL_MODEL_MODE_R2354=shca','--output='+str(r/'results/Kimi-ctx2048-CPU-r123e.out'),'--error='+str(r/'results/Kimi-ctx2048-CPU-r123e.err'),'bash',str(outer),'probe']
# Reserve immutable intent BEFORE Popen; an interrupted submit requires same-label sacct, never blind retry.
q={'CPU_only':True,'GPU':False,'created_utc':time.time(),'parent':'910407','node':'a14r2n20','job_name':'Kimi-ctx2048-HOST-r123e','stage_manifest_SHA256':sha(src/'CPU-stage-file-manifest-r123e.json'),'source_pins_checked':True,'status':'intent-before-Popen'}
with receipt.open('xb')as f:f.write((json.dumps(q,indent=2)+'\n').encode())
p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,start_new_session=True);q.update(controller_pid=p.pid,status='srun-client-started-actual-StepID-pending');tmp=receipt.with_suffix('.tmp');tmp.write_bytes((json.dumps(q,indent=2)+'\n').encode());tmp.replace(receipt);print(json.dumps(q))
