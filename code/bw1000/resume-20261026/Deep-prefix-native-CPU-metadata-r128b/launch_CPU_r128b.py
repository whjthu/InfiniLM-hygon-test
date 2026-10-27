import pathlib,json,hashlib,subprocess,os,time
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-prefix-native-CPU-metadata-r128b';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((D/'manifest-r128b.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
assert sha(R/'candidates/Deep-prefix-native-r128/CPU-build.json')==m['original_failed_receipt_SHA256'];assert sha(R/'candidates/Deep-prefix-native-r128/libDeep_prefix_native_r128.so')==m['library_SHA256'];record=R/'results/Deep-prefix-native-r128b-CPU-launch.json';assert not record.exists()and not(R/'candidates/Deep-prefix-native-CPU-metadata-r128b').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
parent=subprocess.check_output(['scontrol','show','job','910407','-o'],env=env,text=True,timeout=20);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
args=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=2048M','--time=00:04:00','--job-name=Deep-prefix-native-metadata-r128b','--output='+str(R/'logs/Deep-prefix-native-metadata-r128b-%J.log'),'env','WAVE_INSIDE_R2323=Deep-prefix-native-CPU-metadata-r128b/CPU_inside_r128b.sh','bash',str(outer),'probe']
q={'UTC':time.time(),'CPU_only':True,'GPU_requested':False,'Native_TUs_recompiled':0,'DSO_relinked':False,'CPU_contract_test_TU_compiled':1,'manifest_SHA256':sha(D/'manifest-r128b.json'),'args':args,'state':'intent-before-Popen'}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/Deep-prefix-native-metadata-r128b-client.log').open('x')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q.update(client_PID=p.pid,state='submitted_follow_unique_label_no_restart');record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
