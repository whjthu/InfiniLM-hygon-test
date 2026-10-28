"""Pure local CPU source and queue/admission contract. Never launches Slurm/Native."""
import ast,hashlib,importlib.util,json,pathlib,sys,tempfile,types,subprocess
D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for p in D.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
m=json.loads((D/'service-source-manifest-r194.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
model=json.loads((D/'runtime/model-manifest-r9183.json').read_bytes());prefix='/refactor/incoming/Kimi-agent-service-Model-r194/'
for n,h in model['source_SHA256'].items():assert n.startswith(prefix)and sha(D/n.removeprefix(prefix))==h,n
assert m['GPU_authorized']is False and not m['model_or_tools_qualified']
assert model['native_DSO_SHA256']==m['Native_SHA256']=='ba6f9a01e459c5b57839677160cd916cba29312ee8355c939b898bc591ac7d81'
assert model['native_receipt_SHA256']=='37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d'
assert model['topology']=={'tp_nodes':2,'pp':2,'logical_capacity':1}
spec=json.loads((D/'client/client-launch-r194.json').read_bytes())
for path,key in [('client/client_r194.py','SHA256'),('client/client_p0_r143.py','p0_SHA256'),('client/client_regular_agent.py','regular_SHA256'),('client/client_long_output.py','long_output_SHA256'),('runtime/ordinary_token_counter_r179.py','counter_SHA256')]:assert sha(D/path)==spec[key]
assert spec['regular_SHA256']=='51218ed03190ea87b63cb06cf6c5883aae5d327a7db878109fd34c5ce4bde194'and spec['long_output_SHA256']=='20b9bbbc506072890df78d591fc0f22e654173f3ce4744ebf09d35ec8ce3daa9'
checks.append('all_AST_bytepins_actualNative542_profiles_frozen_clients')
sys.path.insert(0,str(D/'runtime'))
from long_input_r179 import KimiLongBridge
import native_engine_bridge_r143 as bridge
reasons=types.SimpleNamespace(ERROR='error',TIMEOUT='timeout',CANCELED='canceled');sp=types.SimpleNamespace(max_tokens=8192,temperature=0.,stop=None)
e=KimiLongBridge(None,32768,reasons,encode_chat=lambda msgs,*a:('CPU only',list(msgs)),max_pending=8)
a=e.add_chat_request(messages=[1]*17408,sampling_params=sp,request_id='CPU-A');b=e.add_chat_request(messages=[2]*8192,sampling_params=sp,request_id='CPU-B')
assert e.take(0)is a and e.take(0)is b and a.max_new_tokens==b.max_new_tokens==8192
assert a._events is not b._events and a.cancel_event is not b.cancel_event and b.metadata['active_requests_at_accept']==1
e.add_aborted_req(a,reasons.CANCELED);assert a.cancel_event.is_set()and not b.cancel_event.is_set()and not a.generated_token_ids
checks.append('FIFO_overlapping_admission_private_cancel_fullprompt_budgets')
# Verify route constants are set before constructing private server, without importing actual server/Torch.
names=['infinilm','infinilm.server','infinilm.server.inference_server','fastapi'];saved={n:sys.modules.get(n)for n in names};mods={n:types.ModuleType(n)for n in names};mods['infinilm'].__path__=[];mods['infinilm.server'].__path__=[];mods['infinilm'].server=mods['infinilm.server'];mods['infinilm.server'].inference_server=mods['infinilm.server.inference_server'];route=mods['infinilm.server.inference_server']
class StubServer:
 def __init__(self,**kwargs):
  assert(route.DEFAULT_STREAM_TIMEOUT,route.DEFAULT_REQUEST_TIMEOUT)==(1800.,3600.)
  assert kwargs['max_tokens']==4096 and kwargs['max_cache_len']==32768 and kwargs['max_batch_size']==1
  self._build_sampling_params=lambda data:data
 def _register_routes(self,app):pass
route.InferenceServer=StubServer;mods['fastapi'].FastAPI=lambda **kwargs:types.SimpleNamespace(**kwargs)
try:
 sys.modules.update(mods);server,app=bridge.existing_server_app(e,'CPU-path-only','kimi_k3');assert server.engine is e and e.validation_timeouts_s=={'no_actual_progress':1800.,'absolute_request_including_queue':3600.}
finally:
 for n,old in saved.items():
  if old is None:sys.modules.pop(n,None)
  else:sys.modules[n]=old
routes=ast.parse((D/'overlay/python/infinilm/server/inference_server.py').read_bytes());calls=[n for n in ast.walk(routes)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='stream_request'];assert len(calls)==2
for call in calls:
 args={k.arg:k.value for k in call.keywords};assert args['timeout'].id=='DEFAULT_STREAM_TIMEOUT'and args['request_timeout'].id=='DEFAULT_REQUEST_TIMEOUT'
checks.append('real_route_constants1800_3600_and_unchanged_absolute_deadline')
target=(D/'runtime/target_model.py').read_text();assert 'modern_kimi_engine_create_batch_r9122('not in target and 'modern_kimi_engine_create_long_agent_r179('in target
protocol=(D/'runtime/target_protocol.py').read_text();assert 'len(prompt) + cap <= 32768'in protocol and 'input_commit_progress' in protocol
idle=(D/'launcher/idle_service_r194.py').read_text();assert 'SO_REUSEADDR'in idle and "states.count('0A')"in idle and "states.count('06')"in idle
client=(D/'client/CPU_client_inside_r194.sh').read_text();assert "CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''"in client and '--timeout 3600'in client
assert 'torch'not in sys.modules and 'ctypes'not in sys.modules
checks.append('explicit_long_factory_no_old_entry_reuseaddr_separate_portstates_CPUclient_masks')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
supervisor=load('CPU_supervisor',D/'launcher/resident_p0_supervisor_r146.py')
with tempfile.TemporaryDirectory(prefix='Kimi189CPU-')as t:
 root=pathlib.Path(t);(root/'results').mkdir();(root/'logs').mkdir();calls=[];rows=[None]
 s=supervisor.ResidentP0Supervisor(root=root,source=D,run='CPU',version='r194',env={},account=lambda name:rows[0],gpu_started=0,clock=lambda:10,popen=lambda args,**kw:(calls.append(args)or types.SimpleNamespace(pid=123)))
 s.tick('910380.CPU');assert not calls
 ident={'actual_step':'910380.CPU','native_SHA256':supervisor.NATIVE_SHA,'cards':32,'TP':16,'PP':2,'max_context_tokens':32768,'protocol_commit':'41a013adf042e18e579689b46b6e9f11bf1fb77b','validation_timeouts_s':{'no_actual_progress':1800.0,'absolute_request_including_queue':3600.0}}
 s.identity.write_text(json.dumps(ident));s.ready.write_text(json.dumps({'ready':True,'service_node':'a14r2n08','actual_step':'910380.CPU','URL_on_service_node':'http://127.0.0.1:43844/v1'}))
 s.tick('910380.CPU');s.tick('910380.CPU');assert len(calls)==1 and '--gres=none'in calls[0]and not any('--gres=dcu' in x for x in calls[0])
 assert any('CPU_client_inside_r194.sh'in x for x in calls[0])and calls[0][-3:]==['bash',str(root/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 s.output.mkdir();(s.output/'result.json').write_text(json.dumps({'identity_SHA256':s.state['identity_SHA256'],'passed':True,'completed':True}));rows[0]=['910380.CPU-client',s.name,'COMPLETED','0:0'];s.tick('910380.CPU');assert s.state['stop_written']and s.state['client_P0_passed']and len(calls)==1
checks.append('unique_CPUclient_outer_container_no_retry_exact_identity_stop')
unchanged={'runtime/parameter_loader_r9143.py':'e17c21d8df2e1d180ab3d91d2a8425349accd598a927b10294308c9897b75675','runtime/batch_policy_r9128.py':'518bef657469fcb837fe6b425b770c1405c824afca454b5635529d0706d581c3','runtime/xtml_codec_r143.py':'7618b171dd40d58548d6707e40fdb88c1666fda8bd23d7107d6e8f7ad070a4d7'}
for n,h in unchanged.items():assert sha(D/n)==h,n
newtree=ast.parse((D/'runtime/native_engine_bridge_r143.py').read_bytes())
getstream=lambda tree:next(n for c in tree.body if isinstance(c,ast.ClassDef)and c.name=='NativeEngineBridge'for n in c.body if isinstance(n,ast.AsyncFunctionDef)and n.name=='stream_request')
assert hashlib.sha256(ast.dump(getstream(newtree),include_attributes=False).encode()).hexdigest()=='162f653b87c0b126b18b96a42185b405c7c81896af5e01feb7ec80abcafa6275'
checks.append('original_parameter_loader_greedy_codec_and_actual_deadline_body_unchanged')

# Read the actual pinned helper constructor boundary, without importing sockets/Torch or creating StageControl.
helper=D/'runtime/stage_control_r2386.py'
assert sha(helper)=='269cc1a5097e0264052d1c929a18ac0bbfe20ade0b44947455dd21bf9e6b804d'
ht=ast.parse(helper.read_bytes());cls=next(n for n in ht.body if isinstance(n,ast.ClassDef)and n.name=='StageControl');ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
bound=next(n for n in ctor.body if isinstance(n,ast.Assert)and isinstance(n.test,ast.Compare)and any(isinstance(x,ast.Name)and x.id=='timeout_seconds'for x in ast.walk(n.test)))
assert ast.dump(bound.test,include_attributes=False)==ast.dump(ast.parse('1<=timeout_seconds<=7200',mode='eval').body,include_attributes=False)
predicate=compile(ast.Expression(bound.test),'actual_pinned_StageControl_bound','eval')
assert eval(predicate,{'timeout_seconds':7200})is True and eval(predicate,{'timeout_seconds':8500})is False
target_tree=ast.parse((D/'runtime/target_model.py').read_bytes())
assignment=next(n for n in target_tree.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='STAGE_CONTROL_WAIT_SECONDS_R194'for x in n.targets));assert ast.literal_eval(assignment.value)==7200
guard=next(n for n in target_tree.body if isinstance(n,ast.Assert)and any(isinstance(x,ast.Name)and x.id=='STAGE_CONTROL_WAIT_SECONDS_R194'for x in ast.walk(n.test)))
call=next(n for n in ast.walk(target_tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='StageControl')
assert assignment.lineno<guard.lineno<call.lineno and call.args[-1].id=='STAGE_CONTROL_WAIT_SECONDS_R194'
assert eval(compile(ast.Expression(guard.test),'pre_constructor_guard','eval'),{'STAGE_CONTROL_WAIT_SECONDS_R194':7200})is True
assert 'STAGE_CONTROL_WAIT_SECONDS_R194'not in (D/'runtime/target_inside.sh').read_text()
assert 'KIMI_RUNTIME_INNER_S_R9225=8500'in (D/'runtime/target_inside.sh').read_text()
assert bridge.VALIDATION_STREAM_TIMEOUT_S==1800. and bridge.VALIDATION_REQUEST_TIMEOUT_S==3600.
assert all(b'\r'not in p.read_bytes()for p in D.rglob('*.sh'))
checks.append('sameSHA_actual_StageControl_constructor_7200_allowed_8500_rejected_guard_before_call_no_socket_allshLF')

z=subprocess.run([sys.executable,'-B',str(D/'CPU_runtime_contract_r194.py')],capture_output=True,text=True,timeout=40);assert z.returncode==0,z.stderr
checks.append('actual_true_input_chunk_fixtures_full8K17K24K_budgets4096_8192_reset_error_cancel')
out={'passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'Model_loaded':False,'Slurm_called':False,'checks':checks,'fixture_is_model_qualification':False,'fullModel_peak_and_numeric_quality':'UNKNOWN','source_manifest_SHA256':sha(D/'service-source-manifest-r194.json')}
(D/'CPU-local-contract-r194.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
