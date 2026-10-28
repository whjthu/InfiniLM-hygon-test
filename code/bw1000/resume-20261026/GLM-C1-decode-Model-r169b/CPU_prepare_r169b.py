"""CPU-only hash/AST/scheduler qualification; no Torch/Native imports."""
import ast,pathlib,json,hashlib,os,subprocess,sys,datetime
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/GLM-C1-decode-Model-r169b';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);B.mkdir(exist_ok=False);m=json.loads((D/'GPU-manifest.json').read_bytes())
for n,h in m['source_SHA256'].items():
 p=D/n;assert sha(p)==h,n
 if n.endswith('.py'):ast.parse(p.read_text())
for n,h in m['existing_remote_SHA256'].items():assert sha(R/n)==h,n
p=subprocess.run([sys.executable,'-B',str(D/'CPU_runtime_contract_r169b.py')],capture_output=True,text=True,timeout=60);assert p.returncode==0,(p.stdout,p.stderr);contract=json.loads(p.stdout);assert contract['passed']and contract['CPU_only']and not contract['GPU']and not contract['model']
q={'completed':True,'CPU_only':True,'GPU':False,'Torch_imported':False,'Native_dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'manifest_SHA256':sha(D/'GPU-manifest.json'),'contract':contract,'Native_DSO_SHA256':m['Model_SHA256'],'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()};(B/'CPU-prepare.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
