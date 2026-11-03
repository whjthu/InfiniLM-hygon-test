"""Actual official GLM checkpoint CPU processor, no Native Model or GPU.

Optional Native root isolation follows successful CPU673; encoder is real.
"""
from pathlib import Path
import asyncio,copy,hashlib,importlib.machinery,json,os,sys,time,types
D=Path(__file__).resolve().parent;R=Path('/refactor');P=R/'incoming/GLM-NoV-16card-W4-pair-r414/owner';F=D.parent/'fixtures-r1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda b:hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()
def main():
 out=R/'candidates/GLM-NoV-W4-quality-encoding-r418-CPU-v1';out.mkdir(exist_ok=False);report=out/'encoding.json';started=time.monotonic();q=dict(completed=False,passed=False,GPU=0,Model=0,Native_dlopen=False,Native_constructed=False,weights_loaded=False,compile=False,actual_service_executed=False,actual_checkpoint_CPU_processor=False,optional_Native_import_isolation=True,actual_root_or_ABI_qualified=False,cases=[],private_prompt_files_stay_R_L=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
 def save():t=report.with_suffix('.tmp');t.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');t.replace(report)
 save()
 try:
  assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert sha(D/'SOURCE_MANIFEST.json')==os.environ['GLM_R418_CPU_SOURCE_SHA256']
  for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
  assert sha(P/'SOURCE_MANIFEST.json')=='5e944fb676f15022f1387d936cc9286c00953f0ca6884ef79b12979b4bca8ac4'
  for n,h in json.loads((P/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(P/n)==h,n
  assert sha(F/'MANIFEST.json')=='c172ddf32b3e414066bcf91d2b0766ed89be763041e8a5ec5f873e7c2383c9df'
  for n,h in json.loads((F/'MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(F/n)==h,n
  dataset=json.loads((F/'gsm8k-new64.json').read_bytes());plan=json.loads((F/'requests-and-pairs.json').read_bytes());assert len(dataset['cases'])==len(plan['requests'])==64 and plan['engine_requests']==len(plan['comparison_schedule'])==128
  assert {c['id']for c in dataset['cases']}=={r['case_id']for r in plan['requests']};assert len({c['id']for c in dataset['cases']})==64
  U=R/'candidates/latest-family-service-r270/GLM/python/infinilm';install=R/'candidates/latest-family-service-r270/GLM/installation.json';assert sha(install)=='bf9bdef51cfcab59cdc72c59b7f371fbfb822153dde42eee1f5ce5e1942e2b15'
  for n,h in json.loads(install.read_bytes())['source_files_SHA256'].items():assert sha(U/n)==h,n
  class ForbiddenNative(types.ModuleType):
   def __getattr__(self,name):raise RuntimeError('CPU encoding forbids Native root API:'+name)
  root=types.ModuleType('infinilm');root.__path__=[str(U)];root.__package__='infinilm';sys.modules['infinilm']=root
  core=ForbiddenNative('infinilm._core');core.__file__='FORBIDDEN_NATIVE';core.__path__=[];core.__spec__=importlib.machinery.ModuleSpec(core.__name__,None);root._core=core;sys.modules[core.__name__]=core
  llm=types.ModuleType('infinilm.llm');llm.__path__=[str(U/'llm')];llm.__package__='infinilm.llm';sys.modules[llm.__name__]=llm
  for suffix,name in [('scheduler','SchedulerOutput'),('static_scheduler','StaticSchedulerOutput')]:
   mod=types.ModuleType('infinilm.llm.'+suffix);setattr(mod,name,type(name,(),{}));sys.modules[mod.__name__]=mod
  class ForbiddenEngine:
   def __init__(self,*a,**k):raise RuntimeError('CPU encoding forbids Async Model construction')
  llm.AsyncLLMEngine=ForbiddenEngine
  def audit(event,args):
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weight/model-binary payload read forbidden')
  sys.addaudithook(audit)
  import torch
  torch.set_num_threads(1)
  def forbidden(*a,**k):raise RuntimeError('GPU API forbidden in checkpoint CPU encoding')
  for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,forbidden)
  torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
  from infinilm.llm.sampling_params import SamplingParams
  from infinilm.llm.request import FinishReason
  llm.SamplingParams=SamplingParams;llm.FinishReason=FinishReason;sys.path.insert(0,str(P))
  from prepared_family_entry_r268 import prepare_app,template_prefix_resolver
  from infinilm.server.agent_protocol import prepare_agent_request
  model=Path('/old/models/ZhipuAI-GLM-5.3');prepared=prepare_app('GLM',str(model));engine,server=prepared['engine'],prepared['server']
  identity=prepared['identity'];assert identity['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'and identity['Native_loaded_by_factory']is identity['weights_opened_by_factory']is False
  template=prepared['tokenizer'].get_chat_template();template_sha=digest(template.encode());assert template_sha=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
  prefix_for=template_prefix_resolver(prepared['processor']);case_by_id={c['id']:c for c in dataset['cases']}
  q.update(source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),fixture_manifest_SHA256=sha(F/'MANIFEST.json'),requests_file_SHA256=sha(F/'requests-and-pairs.json'),template_SHA256=template_sha,prepared_entry_identity=identity,checkpoint_metadata_SHA256={n:sha(model/n)for n in ['config.json','generation_config.json','model.safetensors.index.json']},tokenizer_files_SHA256={p.name:sha(p)for p in model.iterdir()if p.is_file()and p.name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','chat_template.jinja']},actual_checkpoint_CPU_processor=True)
  for ordinal,row in enumerate(plan['requests']):
   body=copy.deepcopy(row['body']);unchanged=copy.deepcopy(body);assert digest(canonical(body))==row['compact_body_SHA256'];assert body['messages'][1]['content']==case_by_id[row['case_id']]['question']and body['max_tokens']==4096 and body['temperature']==0 and body['stream']is False
   data,schemas,choice=prepare_agent_request(copy.deepcopy(body),'glm5');data.update(_agent_schemas=schemas,_agent_choice=choice);params=server._build_sampling_params(data);assert params.max_tokens==4096
   request=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id='CPU418-'+str(ordinal),request_data=data,add_generation_prompt=True,chat_template_kwargs=data.get('chat_template_kwargs')or{})
   assert engine.take(0)is request and request._admission_metadata_ready and request.max_new_tokens==4096
   ids=list(request.prompt_token_ids);prompt=request.prompt;id_sha=digest(json.dumps(ids,separators=(',',':')).encode());h=request.metadata['history_render_audit'];assert h['qualified']and h['prompt_tokens']==len(ids)and request.metadata['Native_prompt_token_ids_SHA256']==id_sha and 0<len(ids)+4096<=32768 and body==unchanged
   reverse_prompt,reverse_ids=engine.encode_chat(data['messages'],data,True,data.get('chat_template_kwargs')or{});assert reverse_prompt==prompt and list(reverse_ids)==ids
   prefix=prefix_for(data,data.get('chat_template_kwargs')or{});assert prompt.endswith(prefix)and 0<len(prefix.encode())<=4096
   private=out/('prompt-%03d-private.json'%ordinal);private.write_text(json.dumps(dict(case_id=row['case_id'],request_body_SHA256=row['compact_body_SHA256'],prompt=prompt,prompt_token_ids=ids),ensure_ascii=False)+'\n')
   engine.record_started(request);engine.record_finished(request,0,True);request.finish(FinishReason.CANCELED)
   async def retire():
    async for token in engine.stream_request(request,timeout=1,request_timeout=2):pass
   asyncio.run(retire());assert request.request_id not in engine._requests
   q['cases'].append(dict(case_id=row['case_id'],source_index=case_by_id[row['case_id']]['source_index'],request_body_SHA256=row['compact_body_SHA256'],prompt_tokens=len(ids),ordinary_budget=4096,input_plus_output=len(ids)+4096,prompt_bytes=len(prompt.encode()),prompt_SHA256=digest(prompt.encode()),prompt_IDs_SHA256=id_sha,prefix=prefix,prefix_SHA256=digest(prefix.encode()),prefix_bytes=len(prefix.encode()),private_path=str(private),private_SHA256=sha(private),reverse_consumer_prompt_and_IDs_equal=True,body_unchanged=True,history_render_audit=h,CPU_queue_retired=True,Native_reset_or_generation_executed=False));save()
  encoded={c['case_id']:c for c in q['cases']}
  for a in plan['comparison_schedule']:assert a['case_id']in encoded and a['compact_body_SHA256']==encoded[a['case_id']]['request_body_SHA256']and a['window']in(2,4)
  assert all(sum(a['case_id']==c and a['window']==w for a in plan['comparison_schedule'])==1 for c in encoded for w in(2,4));assert not engine._requests;engine.stop()
  lengths=[c['prompt_tokens']for c in q['cases']];q.update(passed=True,planned_logical_requests=128,unique_actual_encodings=64,matching_body_pairs=64,prompt_tokens_min=min(lengths),prompt_tokens_max=max(lengths),prompt_tokens_sum=sum(lengths),all_input_plus4096_within32768=True,all_actual_reverse_consumer_equal=True)
 except BaseException as error:
  import traceback;q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  q.update(completed=True,elapsed_s=time.monotonic()-started)
  if 'torch'in sys.modules:
   q['Torch_CUDA_initialized']=sys.modules['torch'].cuda.is_initialized()
   if q['Torch_CUDA_initialized']:q.update(passed=False,unexpected_GPU_initialization=True)
  maps=Path('/proc/self/maps').read_text();q['unexpected_Native_library_maps']=[line for line in maps.splitlines()if any(n in line for n in ['libGLM_V_on_demand_Model_r403.so','libGLM_PP4_registry_r184b.so','_infinilm.cpython'])]
  if q['unexpected_Native_library_maps']:q['passed']=False
  save();print(json.dumps(dict(path=str(report),SHA256=sha(report),passed=q['passed'],actual_CPU=q['actual_CPU'],elapsed_s=q['elapsed_s'])))
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
