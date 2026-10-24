import pathlib,json,hashlib,subprocess,re,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';mp=r/'incoming/GLM-layer0-CPU-manifest-r5756.json';m=json.loads(mp.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
assert not(r/'candidates/glm-layer0-fixture-r5756').exists()
job=subprocess.check_output(['scontrol','show','job','910406','-o'],text=True);assert 'JobState=RUNNING'in job and re.search(r'\bNodeList=a14r2n19\b',job)
live=subprocess.check_output(['scontrol','show','step','910406','-o'],text=True);assert not re.search(r'\bStepId=910406\.\d+\b',live),live
args=['srun','--jobid=910406','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a14r2n19','--gres=none','--mem=4000M','-t','00:05:00','--job-name=glm-layer0-CPU-r5756','--output='+str(r/'logs/glm-layer0-CPU-r5756-%J.log'),'env','CUDA_VISIBLE_DEVICES=','HIP_VISIBLE_DEVICES=','ROCR_VISIBLE_DEVICES=','WAVE_INSIDE_R2323=GLM_layer0_CPU_r5756_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(r/'glm-layer0-CPU-intent-r5756.json').open('x')as f:json.dump({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'freshparent':job,'live':live,'manifest_SHA256':sha(mp)},f)
with(r/'logs/glm-layer0-CPU-r5756-controller.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q={'submitted':True,'PID':p.pid,'scope':'single1TU CPU-only compile/link/extract, no dlopen/GPU/Model','manifest_SHA256':sha(mp)};(r/'glm-layer0-CPU-launch-r5756.json').write_text(json.dumps(q));print(json.dumps(q))
