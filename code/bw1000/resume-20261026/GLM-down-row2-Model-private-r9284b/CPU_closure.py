import pathlib,json,hashlib,os,ast,subprocess,importlib,importlib.util,traceback
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'affinity':sorted(os.sched_getaffinity(0))}
try:
 assert len(q['affinity'])>=32
 qualified=r/'candidates/GLM-down-row2-Model-r9284b/CPU-build.json'
 if os.environ.get('SLURM_PROCID')=='0':subprocess.run(['python','-B',str(inc/'native_CPU_qualify_r9284b.py')],check=True)
 else:
  import time
  for _ in range(120):
   if qualified.exists():break
   time.sleep(.5)
  assert qualified.exists(),'rank0nativeCPUqualifiedreceiptmissing'
 pins=json.loads((inc/'source-pins.json').read_bytes())
 for n,h in pins['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in pins['existing_remote_SHA256'].items():assert sha(r/n)==h,n
 modules=['glm_arrival_probe_r9284b','arrival_runner_r9258','case_plan_r9284b','preflight_r9284b','preengine_r9284b','glm_crossrequest_control_r9059','glm_crossrequest_resident_base_r9059','quality_metrics_r9132b'];q['import_paths']={}
 for name in modules:
  mo=importlib.import_module(name);fp=pathlib.Path(mo.__file__);assert fp.parent==inc,(name,fp);q['import_paths'][name]={'path':str(fp),'SHA256':sha(fp)}
 import torch
 assert not torch.cuda.is_initialized();q['torch_version']=torch.__version__
 from preflight_r9284b import validate
 q['native_metadata']=validate()['model_library_metadata'];assert q['native_metadata']['library_SHA256']==json.loads((r/'candidates/GLM-down-row2-Model-r9284b/CPU-build.json').read_bytes())['library_SHA256'];ld=subprocess.run(['ldd',q['native_metadata']['library_actual_path']],text=True,capture_output=True,timeout=30);assert ld.returncode==0 and 'not found'not in ld.stdout+ld.stderr;q['ldd']=ld.stdout+ld.stderr
 entry=ast.parse((inc/'model_entry_r9284b.py').read_text());q['entry_imports']={}
 for node in entry.body:
  names=[x.name for x in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)else[]
  for name in names:
   found=importlib.util.find_spec(name);assert found,name;q['entry_imports'][name]=found.origin
 from case_plan_r9284b import load
 q['case_plan']=load();from glm_crossrequest_resident_base_r9059 import FLOOR
 assert FLOOR==512*1024**2;q['floor_bytes']=FLOOR
 program=r/'incoming/GLM-prefill-Model-r9241/program-r9241.json';q['program_SHA256']=sha(program);pr=json.loads(program.read_bytes());assert len(pr['service_cases'])==64 and pr['max_context']==640;assert pr['service_cases'][:32]==json.loads((inc/'corpus-reference.json').read_bytes())['requests'];q['distinct_requests']=32
 cfg=json.loads(pathlib.Path('/old/models/ZhipuAI-GLM-5.3/config.json').read_bytes());q['actual_sparse_counts']=[sum(x=='sparse'for x in cfg['mlp_layer_types'][lo:hi])for lo,hi in [(0,38),(38,78)]];assert q['actual_sparse_counts']==[35,40]
 assert sha(inc/'arrival_runner_r9258.py')=='389856df6f5a00fd8b89207f26af39e60a3a8d9d112824cbec9f18cb9c5f11a9'
 z=subprocess.run(['python','-B',str(inc/'CPU_selector.py')],capture_output=True,text=True,timeout=20);assert z.returncode==0,z.stderr;q['selector_CPU']=json.loads(z.stdout)
 for f in inc.glob('*.py'):ast.parse(f.read_text())
 for f in inc.glob('*.sh'):assert b'\r'not in f.read_bytes();subprocess.run(['bash','-n',str(f)],check=True)
 q['completed']=True
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
out=inc/('CPU-result-'+q['node']+'.json');out.write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','actual_step','node','program_SHA256','error']}));assert q['completed']
