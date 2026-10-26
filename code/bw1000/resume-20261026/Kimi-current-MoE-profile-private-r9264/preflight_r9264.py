import pathlib,json,os,sys,ast,subprocess,hashlib
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D));sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((D/'GPU-manifest-source.json').read_bytes())
for p,h in m['source_SHA256'].items():assert sha(p)==h,p
for p in D.iterdir():
 if p.suffix in ['.py','.sh']:assert b'\r'not in p.read_bytes(),p
 if p.suffix=='.py':ast.parse(p.read_bytes())
 if p.suffix=='.sh':assert subprocess.run(['bash','-n',str(p)],capture_output=True).returncode==0,p
import run_GPU_r9264,torch
assert not torch.cuda.is_initialized() and len(os.sched_getaffinity(0))>=32
for p,h in m['pinned_libraries'].items():assert sha(p)==h,p
z=subprocess.run(['ldd','-r',m['library']],text=True,capture_output=True,timeout=60);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr and 'undefined symbol:'not in z.stdout+z.stderr,z.stdout+z.stderr
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'dlopen':False,'host':os.uname().nodename,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'affinity':len(os.sched_getaffinity(0)),'source_manifest_SHA256':sha(D/'GPU-manifest-source.json'),'all_text_LF_AST_bash_n':True,'Runtime_auto_owner_width':4}),flush=True)
