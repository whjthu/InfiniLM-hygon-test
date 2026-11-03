"""Changed typed-entry only: reuse actual913 input; no encode/Native/Runtime."""
from pathlib import Path
import ctypes as C,hashlib,json,os,subprocess,sys,time,traceback
from types import SimpleNamespace
import entry,binding_proof,stage_context
D=Path(__file__).resolve().parent
B=Path('/refactor/candidates/Deep-layer2-3-boundary-diagnostic-r447-entry-CPU-v2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 B.mkdir(exist_ok=False);start=time.monotonic()
 q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],Native_dlopen=False,Runtime_created=False,GPU=False,weights_payload_opened=False,actual_input_encode_replayed=False,old_suites_replayed=False,stages=[])
 def save():
  t=B/'result.tmp';t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(B/'result.json')
 def audit(event,args):
  if event=='ctypes.dlopen'and args and args[0]is not None:raise RuntimeError('No Native/SDK dlopen in changed-entry CPU')
  if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('No Model payload in changed-entry CPU')
 def guard(stage):
  paths=[x.split()[-1]for x in Path('/proc/self/maps').read_text().splitlines()if len(x.split())>=6]
  found=[p for p in paths if p.startswith('/refactor/')or any(s in p for s in ['libDeep_','libinfini','_infinilm.cpython'])]
  q['stages'].append(dict(stage=stage,project_maps=found,visibility={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}));save();assert not found
 save()
 try:
  assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  mf=os.environ['DEEP447_ENTRY_CPU_SOURCE_SHA256'];m=entry.verify_tree(D,mf)
  cfg=json.loads((D/'config.template.json').read_bytes());pin=json.loads((D/'BINDING_PIN.json').read_bytes());plan=json.loads((D/'CPU_PLAN.json').read_bytes())
  entry.configure(cfg,pin,require_ready=False)
  for p,h in plan['inherited_files_SHA256'].items():assert sha(p)==h,p
  original=json.loads(Path(plan['CPU913_receipt']).read_bytes())
  assert original['passed']and original['completed']and original['actual_CPU']=='910407.913'and original['actual_input_processor']
  assert original['actual_input']['prompt_count']==8489 and original['actual_input']['prompt_IDs_SHA256']==plan['prompt_IDs_SHA256']
  q['inherited_actual_CPU913_input']=dict(receipt=plan['CPU913_receipt'],SHA256=sha(plan['CPU913_receipt']),actual_input=original['actual_input'])
  parent=Path(cfg['parent_root']);old=Path(cfg['diagnostic_root']);boundary=Path(cfg['boundary_root'])
  entry.verify_tree(boundary,pin['source_manifest_SHA256'])
  entry.verify_tree(old,'cf570e363016fac276037184ffdc3bdcd03f3bc3b8896ddae1c1031e0d3b552d')
  sys.path[:0]=[str(boundary),str(old),str(parent),cfg['base_provider_root']]
  sys.addaudithook(audit);guard('before_derivation')
  previous=Path('/refactor/candidates/Deep-layer2-3-boundary-diagnostic-r447-entry-CPU-v1/result.json')
  assert sha(previous)=='188c6cc2e7f5ca3b5ac540d97819f6d4562c18fdf5ec9be8edaafb701693cc72'
  completed=json.loads(previous.read_bytes());assert completed['completed']and not completed['passed']and completed['actual_CPU']=='910407.924'and completed['actual_Native_derivation']
  context_root=Path(cfg['context_package_root']);assert sha(context_root/'CONTEXT_DERIVATION.json')=='f510393b69f198c035048f600637317419a6ef9f06947b58307e8c9b369030f0'
  proof=json.loads((context_root/'CONTEXT_DERIVATION.json').read_bytes());assert proof==completed['context_derivation']
  assert proof['Native_SHA256']==pin['library_SHA256']and proof['Native_CPU_receipt_SHA256']==pin['CPU_receipt_SHA256']and sha(pin['library'])==pin['library_SHA256']
  for name,record in proof['modules'].items():assert sha(context_root/'diagnostic_deep_context'/name)==record['private_SHA256']
  q['reused_Native_derivation_CPU924']=dict(receipt=str(previous),SHA256=sha(previous),original_overall_passed=False,successful_phase='actual_Native_derivation and exact Context staging')
  q['context_derivation']=proof
  sys.path.insert(0,str(context_root))
  loader=entry.module('diagnostic_deep_context.checkpoint_loader',context_root/'diagnostic_deep_context/checkpoint_loader.py')
  assert loader.NATIVE_SHA==pin['library_SHA256']
  class Table:
   def __init__(self):self.values={}
   def __getattr__(self,name):return self.values.setdefault(name,SimpleNamespace())
  table=Table();loader.declare(table)
  model_api=entry.module('model_api',old/'model_api.py');model_api.bind(table)
  boundary_api=entry.module('boundary_api',boundary/'boundary_api.py');boundary_api.Capture(table,None)
  runner=entry.module('run_teacher',boundary/'run_teacher.py')
  assert runner.Capture is boundary_api.Capture
  names={line.split()[-1]for line in subprocess.check_output(['nm','-D','--defined-only',pin['library']],text=True,timeout=25).splitlines()if len(line.split())>=3}
  assert set(table.values)<=names and set(binding_proof.CAPTURE_NAMES)<=names
  HIGH=0x123456789abcdef0;seen=[];keep=[]
  for name in ['modern_native_engine_reset','modern_native_engine_destroy']:
   def callback(ctx,name=name):assert ctx==HIGH;seen.append((name,ctx));return 0
   f=C.CFUNCTYPE(C.c_int,C.c_void_p)(callback);keep.append(f);table.values[name]=f
  loader.declare(table)
  assert table.modern_native_engine_reset(HIGH)==0 and table.modern_native_engine_destroy(HIGH)==0 and len(seen)==2
  assert table.modern_native_engine_forward.argtypes[0]is C.c_void_p
  # New API metadata/copy uses real libffi with high addresses; no DSO call.
  data=(C.c_uint64*15)();raw=C.create_string_buffer(32);seen_new=[]
  @C.CFUNCTYPE(C.c_int,C.c_int,C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t)
  def info(owner,index,address,count):
   assert(owner,index,count)==(7,6,15)and C.cast(address,C.c_void_p).value==C.addressof(data);seen_new.append('info');address[0]=83;return 0
  @C.CFUNCTYPE(C.c_int,C.c_int,C.c_size_t,C.c_void_p,C.c_size_t)
  def copy(owner,index,address,count):
   assert(owner,index,count)==(7,6,32)and address==C.addressof(raw);seen_new.append('copy');return 0
  table.values['deep_boundary_info_r447']=info;table.values['deep_boundary_copy_r447']=copy
  boundary_api.Capture(table,None)
  assert table.deep_boundary_info_r447(7,6,data,15)==0 and data[0]==83
  assert table.deep_boundary_copy_r447(7,6,raw,32)==0 and seen_new==['info','copy']
  assert all(getattr(table,n).argtypes is not None for n in binding_proof.MODE_NAMES+binding_proof.CAPTURE_NAMES)
  q['actual_typed_entry']=dict(symbols=len(table.values),new_CABI=7,actual_DSO_nm_checked=True,high64_ctx=hex(HIGH),high64_CPUbyte_pointers=True,Native_leaf='Real libffi callbacks only; actual Native execute=false',actual_runner_file=runner.__file__,actual_capture_file=boundary_api.__file__)
  cfg=dict(cfg,context_package_root=str(context_root))
  (B/'bound-config.json').write_text(json.dumps(cfg,indent=2)+'\n');(B/'BINDING_PIN.json').write_bytes((D/'BINDING_PIN.json').read_bytes())
  cmd=[sys.executable,'-B',str(D/'entry.py'),'--config',str(B/'bound-config.json'),'--config-sha',sha(B/'bound-config.json'),'--pin',str(B/'BINDING_PIN.json'),'--pin-sha',sha(B/'BINDING_PIN.json'),'--source-sha',mf]
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=30);q['actual_CLI']=dict(argv=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
  assert result.returncode==0 and json.loads(result.stdout)==dict(source_schema=True,CPU_READY=True,Native_dlopen=False,Model_READY=False,GPU=False)
  guard('typed_and_actual_CLI_complete')
  q.update(passed=True,source_manifest_SHA256=mf,binding_pin_SHA256=sha(B/'BINDING_PIN.json'),bound_config=dict(path=str(B/'bound-config.json'),SHA256=sha(B/'bound-config.json')),bound_pin=dict(path=str(B/'BINDING_PIN.json'),SHA256=sha(B/'BINDING_PIN.json')),context_proof=dict(path=str(context_root/'CONTEXT_DERIVATION.json'),SHA256=sha(context_root/'CONTEXT_DERIVATION.json')))
 except BaseException as error:q.update(passed=False,error=repr(error),traceback=traceback.format_exc())
 finally:q.update(completed=True,wall_s=time.monotonic()-start);save();print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','error','wall_s']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
