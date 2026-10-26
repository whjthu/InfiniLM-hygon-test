import pathlib,json,os,sys,ast,subprocess,hashlib
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D));sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((D/'GPU-manifest-final-source.json').read_bytes())
for p,h in m['source_SHA256'].items():assert sha(p)==h,p
for p in D.iterdir():
 if p.suffix in ['.py','.sh']:assert b'\r'not in p.read_bytes(),p
 if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
 if p.suffix=='.sh':assert subprocess.run(['bash','-n',str(p)],capture_output=True).returncode==0,p
code=(D/'kimi_TP16_event_runner_r9220.cpp').read_text();assert 'pthread_setaffinity_np('not in code and 'actual_owner_cpus!=owner_cpus'in code and 'process_cpus.size()<32'in code
binary='/tmp/Kimi-owner-contract-'+os.uname().nodename+'-'+str(os.getpid());built=subprocess.run(['c++','-std=c++17','-pthread',str(D/'owner_CPU_contract.cpp'),'-o',binary],text=True,capture_output=True,timeout=30);assert built.returncode==0,built.stderr;run_owner=subprocess.run([binary],capture_output=True,text=True);assert run_owner.returncode==0,run_owner.stderr;owner_contract=json.loads(run_owner.stdout);assert owner_contract['passed']and owner_contract['actual_original_header']
import run_GPU_r9220e,torch
assert not torch.cuda.is_initialized()and len(os.sched_getaffinity(0))>=32
for p,h in m['pinned_libraries'].items():assert sha(p)==h,p
z=subprocess.run(['ldd','-r',m['library']],text=True,capture_output=True,timeout=60);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr and 'undefined symbol:'not in z.stdout+z.stderr
cuda_lines=[line for line in z.stdout.splitlines()if 'libcudart.so.12 =>'in line];assert len(cuda_lines)==1;cuda_path=cuda_lines[0].split('=>',1)[1].strip().split()[0];cuda_provider={cuda_path:sha(cuda_path)}
print(json.dumps({'CUDAEvent_provider_SHA256':cuda_provider,'passed':True,'CPU_only':True,'GPU':False,'dlopen':False,'host':os.uname().nodename,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'affinity':len(os.sched_getaffinity(0)),'source_manifest_SHA256':sha(D/'GPU-manifest-final-source.json'),'all_text_LF_AST_bash_n':True,'CPU_owner_fullpool_before_firstSubmit_auto4_contract':owner_contract}),flush=True)
