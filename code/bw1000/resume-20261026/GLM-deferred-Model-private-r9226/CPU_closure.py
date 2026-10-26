import pathlib,json,hashlib,os,ast,subprocess,importlib,importlib.util,traceback
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'affinity':sorted(os.sched_getaffinity(0))}
try:
 assert len(q['affinity'])>=32
 pins=json.loads((inc/'source-pins.json').read_bytes())
 for n,h in pins['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in pins['existing_remote_SHA256'].items():assert sha(r/n)==h,n
 modules=['glm_ragged_probe_r9226','preflight_r9183','preengine_r9183','glm_crossrequest_control_r9059','glm_crossrequest_resident_base_r9059','quality_metrics_r9132b','schedule_r9132b','decode_wave_r9225','case_plan_r9226']
 q['import_paths']={}
 for name in modules:
  m=importlib.import_module(name);fp=pathlib.Path(m.__file__);assert fp.parent==inc,(name,fp);q['import_paths'][name]={'path':str(fp),'SHA256':sha(fp)}
 import torch
 assert not torch.cuda.is_initialized();q['torch_version']=torch.__version__
 from preflight_r9183 import validate
 q['native_metadata']=validate()['model_library_metadata'];assert q['native_metadata']['library_SHA256']=='a549b9b3951b51b52d3f8a96f0fe585da6a92cb88e77054970fdbef3ad483078'
 ld=subprocess.run(['ldd',q['native_metadata']['library_actual_path']],text=True,capture_output=True,timeout=30);assert ld.returncode==0 and 'not found'not in ld.stdout+ld.stderr;q['ldd']=ld.stdout+ld.stderr
 entry=ast.parse((inc/'model_entry_r9226.py').read_text());q['entry_imports']={}
 for node in entry.body:
  names=[x.name for x in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)else[]
  for name in names:
   found=importlib.util.find_spec(name);assert found,name;q['entry_imports'][name]=found.origin
 from case_plan_r9226 import load
 q['case_plan']=load();z=subprocess.run(['python','-B',str(inc/'CPU_selector.py')],capture_output=True,text=True,timeout=20);assert z.returncode==0,z.stderr;q['selector_CPU']=json.loads(z.stdout)
 # Reuse exact frontend loop test; no nativeRuntime or GPU invocation.
 z=subprocess.run(['python','-B',str(inc/'CPU_actual_loop.py')],capture_output=True,text=True,timeout=30);assert z.returncode==0,z.stderr;q['loop_CPU']=json.loads(z.stdout)
 from glm_crossrequest_resident_base_r9059 import FLOOR
 assert FLOOR==512*1024**2;q['floor_bytes']=FLOOR
 program=r/'results/GLM-B32-tokenized-r9183.json';q['program_SHA256']=sha(program);pr=json.loads(program.read_bytes());assert len(pr['service_cases'])==32 and len({x['id']for x in pr['service_cases']})==32 and pr['max_context']==640;q['distinct_requests']=32
 for f in inc.glob('*.py'):ast.parse(f.read_text())
 for f in inc.glob('*.sh'):assert b'\r'not in f.read_bytes();subprocess.run(['bash','-n',str(f)],check=True)
 q['completed']=True
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
out=inc/('CPU-result-'+q['node']+'.json');out.write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','actual_step','node','program_SHA256','error']}));assert q['completed']
