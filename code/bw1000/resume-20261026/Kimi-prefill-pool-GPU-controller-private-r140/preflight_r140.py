import pathlib,json,os,ast,sys,subprocess,hashlib,importlib.util
D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((D/'GPU-manifest-source.json').read_bytes());pin=json.loads((D/'controller-source-pins.json').read_bytes())
for rel,h in pin['files_SHA256'].items():assert sha(D/rel)==h,rel
for p in D.iterdir():
 if p.suffix in ['.py','.sh']:assert b'\r\n'not in p.read_bytes()
 if p.suffix=='.py':ast.parse(p.read_bytes())
 if p.suffix=='.sh':assert subprocess.run(['bash','-n',str(p)],capture_output=True).returncode==0
for p,h in m['pinned_libraries'].items():assert sha(p)==h,p
assert sha(m['native'])==m['native_SHA256']and sha(m['library'])==m['library_SHA256'];assert len(os.sched_getaffinity(0))>=32
sys.path.insert(0,str(D));import mapped_layer_weights_r9125,Kimi_full_MoE_input_r8669s
for mod in [mapped_layer_weights_r9125,Kimi_full_MoE_input_r8669s]:assert pathlib.Path(mod.__file__).resolve().parent==D
spec=importlib.util.find_spec('torch');assert spec and spec.origin and 'torch'not in sys.modules
z=subprocess.run(['ldd','-r',m['library']],text=True,capture_output=True,timeout=60);assert z.returncode==0 and'not found'not in z.stdout+z.stderr and'undefined symbol:'not in z.stdout+z.stderr and m['helper_expected_path']in z.stdout+z.stderr
defs=subprocess.check_output(['nm','-D','--defined-only',m['library']],text=True,timeout=30);assert all(any(line.split()[-1]==name and line.split()[-2]=='T'for line in defs.splitlines()if len(line.split())>=3)for name in ['modern_kimi_prefill_pool_run_r132','modern_kimi_prefill_pool_result_r132'])
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'dlopen':False,'Torch_imported':False,'host':os.uname().nodename,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'affinity':len(os.sched_getaffinity(0)),'source_manifest_SHA256':sha(D/'GPU-manifest-source.json'),'all_text_LF_AST_bash_n':True,'actual466pins':True}),flush=True)
