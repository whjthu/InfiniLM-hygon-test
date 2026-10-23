"""Single offline CPU tokenizer task, never a Model or GPU launch."""
import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');mp=r/'incoming/GLM-I512x32-fixture-CPU-manifest-r5732.json';m=json.loads(mp.read_text())
for n,h in m['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
for n in['glm-I512x32-fixture-r5732-launch.json','glm-I512x32-fixture-r5732-intent.json','results/glm-I512x32-real-inputs-r5732.json','results/glm-I512x32-real-inputs-receipt-r5732.json']:assert not(r/n).exists(),n
# One CPU, noGPU; existing GPU tasks may continue. Exclusive names prevent replay.
args=['srun','--jobid=910407','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=8000M','-t','00:10:00','--job-name=GLM-I512x32-fixture-CPU-r5732','--output='+str(r/'logs/GLM-I512x32-fixture-CPU-r5732-%J.log'),'env','CUDA_VISIBLE_DEVICES=','HIP_VISIBLE_DEVICES=','ROCR_VISIBLE_DEVICES=','WAVE_INSIDE_R2323=GLM_I512x32_fixture_r5732_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(r/'glm-I512x32-fixture-r5732-intent.json').open('x')as f:json.dump({'args':args,'manifest_SHA256':hashlib.sha256(mp.read_bytes()).hexdigest(),'exclusive_pre_Popen':True},f)
with(r/'logs/GLM-I512x32-fixture-CPU-r5732-controller.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'args':args,'manifest_SHA256':hashlib.sha256(mp.read_bytes()).hexdigest(),'Model_or_GPU_work':False};(r/'glm-I512x32-fixture-r5732-launch.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
