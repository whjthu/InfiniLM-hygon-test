import ast,importlib,importlib.util,json,pathlib,os,socket,hashlib,sys
sys.path.append('/refactor/incoming');src=pathlib.Path(__file__).parent;tree=ast.parse((src/'GPU_fixture.py').read_text());names=set()
for x in ast.walk(tree):
 if isinstance(x,ast.Import):names.update(a.name for a in x.names)
 elif isinstance(x,ast.ImportFrom)and x.module:names.add(x.module)
rows=[]
for name in sorted(names):
 spec=importlib.util.find_spec(name);assert spec is not None,name;module=importlib.import_module(name);p=getattr(module,'__file__',None);rows.append({'module':name,'origin':str(spec.origin),'actual_file':p,'SHA256':hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()if p and pathlib.Path(p).is_file()else None})
import torch
assert not torch.cuda.is_initialized();q={'CPU_only':True,'GPU':False,'GPU_entry_executed':False,'AST_all_imports_discovered_and_safe_imported':True,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'modules':rows,'Torch_GPU_uninitialized':True};(src/'CPU-entry-import-closure-r9182b.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
