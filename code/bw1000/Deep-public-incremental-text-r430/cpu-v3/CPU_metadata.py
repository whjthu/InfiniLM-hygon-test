"""Import/admission metadata closure only; original863 nine functional cases not rerun."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,sys,time,traceback
from types import SimpleNamespace
D=Path(__file__).resolve().parent;SOURCE=D.parent;R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=R/'candidates/Deep-public-incremental-text-r430-CPU-metadata-v3';out.mkdir(exist_ok=False);report=out/'result.json';started=time.monotonic()
 q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=False,Native_dlopen=None,weight_open_attempts=[],nine_functional_cases_replayed=False,stages=[])
 def save():
  t=report.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2)+'\n');t.replace(report)
 def observe(stage):
  torch=sys.modules.get('torch');initialized=None if torch is None else bool(torch.cuda.is_initialized())
  paths=sorted({line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if len(line.split())>=6 and line.split()[-1].startswith('/')})
  markers=['libDeep_','libGLM_','libKimi_','_infinilm.cpython','libinfinirt.so','libinfinilm_runtime.so','libinfiniops.so','libinfinicore.so','libinfiniccl.so']
  project=[p for p in paths if p.startswith('/refactor/')or any(name in p for name in markers)]
  q['stages'].append(dict(stage=stage,framework_Torch_imported=torch is not None,framework_modules={n:getattr(sys.modules.get(n),'__file__',None)for n in ['torch','transformers','tokenizers','numpy']},hidden_GPU_env={k:os.environ[k]for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']},torch_cuda_Python_initialized=initialized,all_project_map_matches=project,project_python_modules={n:getattr(m,'__file__',None)for n,m in list(sys.modules.items())if n.startswith(('public_deep','infinilm'))}));save()
  if project:q['Native_dlopen']=True;raise RuntimeError('Project map present; preserve and stop without broad allowlist')
  if initialized:raise RuntimeError('Unexpected framework GPU initialized flag')
 save();engine=None;req=None
 try:
  assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  assert sha(D/'SOURCE_MANIFEST.json')==os.environ['R430_CPU_SOURCE_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
  for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
  assert sha(SOURCE/'SOURCE_MANIFEST.json')==m['production_source_manifest_SHA256']
  for name,h in json.loads((SOURCE/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(SOURCE/name)==h,name
  old=SOURCE/'cpu-v2';assert sha(old/'SOURCE_MANIFEST.json')==m['CPU_v2_source_manifest_SHA256']
  for name,h in json.loads((old/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(old/name)==h,name
  plan=json.loads((SOURCE/'PLAN.json').read_bytes());fixture=R/'results/Deep430-original-two-outputs-private.json';assert sha(fixture)==plan['private_fixture_SHA256'];seq=json.loads(fixture.read_bytes())['sequences'][0]
  model=Path(plan['tokenizer_path'])
  for name,h in plan['tokenizer_assets_SHA256'].items():assert sha(model/name)==h,name
  parent=R/'incoming/public-Deep-native-provider-r427/source-v3';public=R/'incoming/public-GLM-native-provider-r425/public-python'
  assert sha(parent/'SOURCE_MANIFEST.json')==plan['qualified_427_source_SHA256']and sha(public.parent/'SOURCE_MANIFEST.json')==plan['qualified_public_source_SHA256']
  def audit(event,args):
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):
    q['weight_open_attempts'].append(os.fsdecode(args[0]));raise RuntimeError('Weight payload forbidden')
  sys.addaudithook(audit);sys.path[:0]=[str(SOURCE),str(parent)];observe('before_imports')
  spec=importlib.util.spec_from_file_location('original_430_CPU_v2_methods',old/'CPU_entry.py');actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
  Parser,prepare,history=actual.pure_public(public);observe('actual_public_parser_imports')
  processor,methods=actual.actual_processor(public,model,history);observe('actual_Deep_processor_and_tokenizer')
  from public_deep_text.prepared_factory import prepare_engine
  reasons=SimpleNamespace(TIMEOUT='timeout',ERROR='error',CANCELED='canceled',EOS_TOKEN='eos_token',LENGTH='length')
  engine=prepare_engine('Deep',str(model),processor=processor,finish_reasons=reasons)['engine'];engine.start();observe('actual_private_prepared_factory')
  body=copy.deepcopy(seq['body']);data,schemas,choice=prepare(body,parser='deepseek_v4');data['_agent_schemas']=schemas;data['_agent_choice']=choice
  server=actual.public_consumer(public,Parser,engine);server._build_sampling_params(data);params=server._build_sampling_params(data)
  req=engine.add_chat_request(messages=data.get('messages',[]),sampling_params=params,request_id='CPU-metadata-only',request_data=data,add_generation_prompt=bool(data.get('add_generation_prompt',True)),chat_template_kwargs=data.get('chat_template_kwargs')or{})
  assert engine.take(timeout=1)is req;digest=hashlib.sha256(json.dumps(list(req.prompt_token_ids),separators=(',',':')).encode()).hexdigest();assert digest==seq['original_prompt_IDs_SHA256']
  observe('actual_one_admission_no_emit_or_stream');assert not q['weight_open_attempts']
  q.update(source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),production_source_manifest_SHA256=sha(SOURCE/'SOURCE_MANIFEST.json'),CPU_v2_source_manifest_SHA256=sha(old/'SOURCE_MANIFEST.json'),actual_prompt_tokens=len(req.prompt_token_ids),actual_prompt_IDs_SHA256=digest,Native_dlopen=False,Model_READY=False,actual_checkpoint_processor_method_sources=methods,Native_owner_calls=0,decoder_prefix_or_parser_case_replays=0,scope='Framework import can load Torch; recorded import/maps/initialized Python flag only. No inference, Native owner, output sequence replay or device qualification.')
  # passed becomes true only after every metadata guard above. No future error
  # can leave it true: except forces false and the finally preserves cleanup.
  q['passed']=True
 except BaseException as error:q.update(passed=False,error=repr(error),traceback=traceback.format_exc());raise
 finally:
  try:
   if req is not None:
    req.mark_abort('canceled');q['metadata_only_unowned_request_abort_intent']=True;q['no_forged_Native_finish']=not req.is_finished()
   if engine is not None:engine.stop()
  except BaseException as cleanup:q.update(passed=False,cleanup_error=repr(cleanup));raise
  finally:q.update(completed=True,wall_s=time.monotonic()-started);save()
if __name__=='__main__':main()
