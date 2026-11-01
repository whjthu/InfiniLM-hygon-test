"""Official current prepared ingress only; never construct a Native model."""
import pathlib,json,hashlib,copy,os,sys,time
R=pathlib.Path('/refactor');D=R/'incoming/GLM-prefill-W2-W4-owner-r307';HERE=pathlib.Path(__file__).parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(D/'SOURCE_MANIFEST.json')=='3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'
parent=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in parent['files_SHA256'].items())
assert all(os.environ[k]=='-1' for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
card=json.loads((HERE/'FIXTURE_CARD.json').read_bytes());step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('GLM349-official-encoding-'+step);assert not out.exists();out.mkdir()
report={'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'weights_loaded':False,'private_Native_dlopen':False,'actual_CPU':step,'fixtures':[],'raw_ID_sequences_stay_private':True};start=time.monotonic()
try:
 from prepared_family_entry_r268 import prepare_app
 prepared=prepare_app('GLM','/old/models/ZhipuAI-GLM-5.3');assert prepared['identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'
 assert prepared['identity']['Native_loaded_by_factory'] is prepared['identity']['weights_opened_by_factory'] is False
 from infinilm.server.agent_protocol import prepare_agent_request
 engine,server=prepared['engine'],prepared['server']
 for entry in card['fixtures']:
  path=pathlib.Path(entry['request_external_path'].replace('/public/home/aclsylqidf/hygon-inference-refactor-20261014','/refactor',1));raw=path.read_bytes();assert sha(path)==entry['request_SHA256'];body=json.loads(raw);unchanged=copy.deepcopy(body)
  data,schemas,choice=prepare_agent_request(copy.deepcopy(body),server.tool_call_parser)
  params=server._build_sampling_params(data);assert params.max_tokens==4096
  request=engine.add_chat_request(messages=data['messages'],sampling_params=params,request_id='CPU349-'+entry['id'],request_data=data,add_generation_prompt=True,chat_template_kwargs=data.get('chat_template_kwargs') or {})
  assert body==unchanged and request._admission_metadata_ready and request.max_new_tokens==4096
  ids=list(request.prompt_token_ids);digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest();h=request.metadata['history_render_audit'];assert h['qualified'] and h['prompt_tokens']==len(ids) and request.metadata['Native_prompt_token_ids_SHA256']==digest
  if entry['id']=='anchor-calculate':assert len(ids)==8900 and digest=='c0c4f97b19cf58227e9ef8b60c17520cb2b44e8337c1c1805d96b9aab7ee16fe'
  assert len(ids)+4096<=32768
  private=out/(entry['id']+'-prompt-IDs.json');private.write_text(json.dumps({'fixture':entry['id'],'request_SHA256':entry['request_SHA256'],'prompt_token_ids':ids,'prompt_SHA256':hashlib.sha256(request.prompt.encode()).hexdigest()},ensure_ascii=False))
  report['fixtures'].append({'id':entry['id'],'request_SHA256':entry['request_SHA256'],'prompt_tokens':len(ids),'prompt_IDs_SHA256':digest,'prompt_render_SHA256':hashlib.sha256(request.prompt.encode()).hexdigest(),'private_IDs_path':str(private),'private_IDs_file_SHA256':sha(private),'input_unchanged':True,'history_render_audit':h})
 engine.stop();report.update(passed=True,prepared_entry_identity=prepared['identity'],official_processor_class=prepared['identity']['processor_class'],current_server_SHA256=prepared['identity']['server_SHA256'],actual_prepared_ingress=True)
except BaseException as error:
 import traceback
 report.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
finally:
 report.update(completed=True,seconds=time.monotonic()-start)
 maps=pathlib.Path('/proc/self/maps').read_text();assert 'GLM-PP4-registry-repair-r184b' not in maps
 if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
 (out/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)
