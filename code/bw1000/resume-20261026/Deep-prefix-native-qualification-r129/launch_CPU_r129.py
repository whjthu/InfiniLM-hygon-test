"""Run separately after staging. HOST-only step, no GPU submission."""
import pathlib,json,hashlib,subprocess,os,time
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-prefix-native-qualification-r129';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((D/'manifest-r129.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
record=R/'results/Deep-prefix-native-qualification-r129-CPU-launch.json';assert not record.exists()and not(R/'candidates/Deep-prefix-native-qualification-r129').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
parent=subprocess.check_output(['scontrol','show','job','910407','-o'],env=env,text=True,timeout=20);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
args=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c8','-w','a14r2n20','--gres=none','--mem=32768M','--time=00:15:00','--job-name=Deep-prefix-native-HOST-r129','--output='+str(R/'logs/Deep-prefix-native-HOST-r129-%J.log'),'env','WAVE_INSIDE_R2323=Deep-prefix-native-qualification-r129/CPU_inside_r129.sh','bash',str(outer),'probe']
q={'UTC':time.time(),'CPU_only':True,'GPU_requested':False,'manifest_SHA256':sha(D/'manifest-r129.json'),'status':'intent-before-Popen','args':args}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/Deep-prefix-native-HOST-r129-client.log').open('x')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q.update(client_PID=p.pid,status='srun_started_follow_unique_label_no_restart');record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
