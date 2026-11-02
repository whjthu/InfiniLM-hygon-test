"""Changed real admission/context/loader glue; Native/device/weights are leaves."""
from pathlib import Path
import ast,copy,ctypes,hashlib,json,os,tempfile,types
from unittest.mock import patch
D=Path(__file__).resolve().parent;O=D/'owner'
def run(torch):
 from owned_context_r404 import OwnedContext
 from model_admission_r404 import constructor_snapshot,NATIVE_SHA
 source=ast.parse((O/'model_entry_r290.py').read_bytes());worker=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
 creates=[n for n in ast.walk(worker)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_glm_model_create'];assert len(creates)==1
 withnode=next(n for n in worker.body if isinstance(n,ast.With)and isinstance(n.items[0].context_expr,ast.Name)and n.items[0].context_expr.id=='_owned_r404')
 body=withnode.body
 hook=next(i for i,n in enumerate(body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='constructor_snapshot')
 gate=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.slice,ast.Constant)and t.slice.value=='memory_after_create'for t in n.targets))
 load=next(i for i,n in enumerate(body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Subscript)and isinstance(n.value.func.slice,ast.Constant)and n.value.func.slice.value=='load')
 assert hook<gate<load
 gate_if=next(i for i in range(gate,len(body))if isinstance(body[i],ast.If)and 'early.item()'in ast.unparse(body[i].test))
 guard_fn=ast.FunctionDef(name='original_preweight_guard',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='ctx',annotation=None,type_comment=None),ast.arg(arg='report',annotation=None,type_comment=None)],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=copy.deepcopy(body[gate:gate_if+1]),decorator_list=[],returns=None,type_comment=None)
 guardtree=ast.Module(body=[guard_fn],type_ignores=[]);ast.fix_missing_locations(guardtree);guardcode=compile(guardtree,'actual-r404-original8GiB-admission','exec')
 baseline=Path('/refactor/incoming/modern_glm_full_model_probe_r2.py');assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='a9b643a21e5ff5eaea021d768aa0342a155948fd2f962157c08a683647d06a05'
 text=baseline.read_text();raw=text[text.index(' keys=set('):text.index(' def control(x):')]
 loadercode=compile('def actual_raw_loader(ctx,report):\n'+raw+'\n return report\n','original-pinned-raw-loader-recording-leaves','exec')
 model=Path('/old/models/ZhipuAI-GLM-5.3');mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];key=next(k for k in sorted(mapping)if k.startswith('model.layers.38.'))
 class CheckpointLeaf:
  def __enter__(self):return self
  def __exit__(self,*a):return False
  def get_tensor(self,name):assert name==key;return torch.ones(1,dtype=torch.bfloat16)
 class Function:
  def __init__(self,fn):self.fn=fn
  def __call__(self,*a):return self.fn(*a)
 cases=[]
 with tempfile.TemporaryDirectory(prefix='r404-samectx-')as temp:
  r=Path(temp);(r/'results').mkdir()
  for phase in ['normal','census_reject','floor_reject','load_failure','finish_failure','request_failure','cleanup_failure','primary_plus_cleanup_failure']:
   events=[];ctx=object();report={};lib=types.SimpleNamespace();calls={'create':1,'destroy':0,'shutdown':0,'load':0,'finish':0};events.append('create')
   def check(rc):
    if rc:raise RuntimeError('recording Native RC '+str(rc))
   def destroy(c):assert c is ctx;calls['destroy']+=1;events.append('destroy');return 1 if phase in['cleanup_failure','primary_plus_cleanup_failure']else 0
   def shutdown():calls['shutdown']+=1;events.append('shutdown');return 0
   lib.modern_glm_model_destroy=destroy;lib.modern_glm_model_shutdown=shutdown
   layer=dict(layer=38,tensors=[dict(category=k,shape=[1],bytes=2,unique_start_address=True)for k in ['latent','rotary','projected_key']])
   census=dict(layer_begin=38,layer_end=78,rows=1,capacity=32768,loaded=False,cache_layers=[dict(layer=l,tensors=copy.deepcopy(layer['tensors']))for l in range(38,78)],cache_category_bytes=dict(latent=80,rotary=80,projected_key=80),cache_unique_start_bytes=240)
   if phase=='census_reject':census['cache_category_bytes']['projected_value']=2
   def getc(c):assert c is ctx;events.append('census');return json.dumps(census).encode()
   lib.modern_glm_constructor_census_r4158=Function(getc)
   lib.modern_glm_model_keys=lambda c:json.dumps([key]).encode()
   def loadc(c,desc):
    assert c is ctx;values=json.loads(desc);assert len(values)==1 and values[0]['name']==key and values[0]['pointer']>0
    calls['load']+=1;events.append('load');return 1 if phase in ['load_failure','primary_plus_cleanup_failure']else 0
   def finish(c):assert c is ctx;calls['finish']+=1;events.append('finish');return 1 if phase=='finish_failure'else 0
   lib.modern_glm_model_load_batch=loadc;lib.modern_glm_model_finish=finish
   def control(c,n,t,stream):assert c is ctx;t.mul_(16);events.append('WORLD_admission');return 0
   lib.modern_glm_model_control=control
   lp=r/(phase+'-life.json');snapshot=r/(phase+'-snapshot.json')
   env=dict(GLM_V_ON_DEMAND_R403='1',GLM_RUN_R5040='CPU404-'+phase,GLM_SOURCE_MANIFEST_SHA=os.environ['GLM_SOURCE_MANIFEST_SHA'])
   with patch.dict(os.environ,env),patch.object(torch.cuda,'mem_get_info',lambda *a:((7 if phase=='floor_reject'else 12)*1024**3,64*1024**3)):
    lease=OwnedContext(lib,ctx,check,1,0,path=lp)
    ns=dict(torch=torch,lib=lib,check=check,ptr=lambda t:t,stream=lambda:0,rank=0,stage=1,CARDS=16,tag='CPU404-'+phase,r=r,_owned_r404=lease,verify_owner_affinity=lambda *a:{'recording_only':True},service_stats=lambda:{},json=json,model=model,collections=__import__('collections'),safe_open=lambda *a,**k:CheckpointLeaf(),time=__import__('time'),os=os)
    exec(guardcode,ns);exec(loadercode,ns)
    try:
     with lease:
      constructor_snapshot(lib,ctx,1,0,report,snapshot,provider_reader=lambda _:dict(library_SHA256=NATIVE_SHA,recording_provider_leaf=True))
      assert snapshot.exists()and hashlib.sha256(snapshot.read_bytes()).hexdigest()==report['constructor_snapshot_SHA256'];events.append('actual_snapshot_read')
      ns['original_preweight_guard'](ctx,report)
      if not report.get('capacity_rejected_before_disk_loading'):
       ns['actual_raw_loader'](ctx,report)
       events.append('request_same_ctx')
       if phase=='request_failure':raise RuntimeError('recording request leaf failure')
    except RuntimeError:pass
   state=json.loads(lp.read_bytes());assert calls['create']==calls['destroy']==calls['shutdown']==1
   assert state['run']==env['GLM_RUN_R5040']and state['source_manifest_SHA256']==env['GLM_SOURCE_MANIFEST_SHA']
   if snapshot.exists():
    produced=json.loads(snapshot.read_bytes());assert produced['run']==state['run']and produced['actual_step']==state['actual_step']and produced['source_manifest_SHA256']==state['source_manifest_SHA256']
   assert state['destroy_attempts']==state['shutdown_attempts']==1 and state['completed']
   if phase in ['census_reject','floor_reject']:assert calls['load']==calls['finish']==0
   if phase=='normal':assert calls['load']==calls['finish']==1 and state['normal']and events.index('actual_snapshot_read')<events.index('WORLD_admission')<events.index('load')<events.index('finish')<events.index('request_same_ctx')<events.index('destroy')
   if phase not in ['normal','floor_reject']:assert not state['normal']
   if phase=='primary_plus_cleanup_failure':assert state['primary_error']and state['cleanup_errors']
   cases.append(dict(name=phase,passed=True,events=events,counts=calls,primary_preserved=state['primary_error'],cleanup_errors=state['cleanup_errors'],actual_lifecycle_file=True,Native_leaves_recording_only=True))
 return dict(passed=True,cases=cases,original_worker_create_calls=1,actual_OwnedContext=True,actual_constructor_snapshot=True,actual_pinned_loader_body=True,original_admission_AST=True,same_ctx=True,no_actual_checkpoint_or_Native=True,raw_loader_fixture='One synthetic CPU tensor behind explicit safe_open leaf, original actual index name; not checkpoint math/geometry qualification',actual_GPU=False)
