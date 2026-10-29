"""Finite CPU replay of two genuine GLM request histories; no Model or source export."""
import pathlib,json,hashlib,copy,sys,importlib.util,os,datetime
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);os.environ.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0')
R=pathlib.Path('/refactor');D=R/'incoming/GLM-regular-service-r204';out=R/'results/GLM-actual-history-CPU-r220.json';assert not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
PINS={'overlay/python/infinilm/server/agent_protocol.py': 'c91ab84a5ddc800602dcae0319ad5c02ad3b01c1936a2bd57bce90c2ad15abc3', 'api_request_constraints_r203.py': '7e8b3476b1e89c2c5e57b491ee3c0027466451981377f5ab89cf97a30c5e263e'}
def load(name,rel):
 p=D/rel;assert sha(p)==PINS[rel];s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'CPU_only':True,'GPU':False,'Native_imported':False,'weights_read':False,'source_or_payload_exported':False,'cases':[],'source_SHA256':PINS}
try:
 protocol=load('r220_actual_protocol','overlay/python/infinilm/server/agent_protocol.py');constraints=load('r220_actual_constraints','api_request_constraints_r203.py')
 from transformers import PreTrainedTokenizerFast
 tok=PreTrainedTokenizerFast.from_pretrained('/old/models/ZhipuAI-GLM-5.3',local_files_only=True);assert hashlib.sha256(tok.get_chat_template().encode()).hexdigest()=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c';enc=constraints.make_encode_chat(tok)
 watch=json.loads((R/'GLM-regular-service-r204-watch.json').read_bytes());assert watch['run']=='GLM-regular-service-r204-b34db8d1eb674914'
 for case in ['HTTP','SSE']:
  folder=R/'results'/('GLM-regular-live-r204-'+watch['run'])/case;req=folder/'request-005.request.json';decoded=folder/'request-005.decoded.json';raw=req.read_bytes();body=json.loads(raw);prepared,schemas,choice=protocol.prepare_agent_request(body,'glm5');prepared['_agent_choice']=choice;kwargs=prepared.get('chat_template_kwargs')or{};assert kwargs.get('clear_thinking')is False
  before=copy.deepcopy(prepared);prompt,ids=enc(prepared['messages'],prepared,prepared.get('add_generation_prompt',True),kwargs);assert before==prepared
  markers=copy.deepcopy(prepared['messages']);fields=[]
  for i,m in enumerate(markers):
   rc=m.get('reasoning_content')
   if isinstance(rc,str)and rc:
    marker='R220_RC_'+str(i)+'_97ad5f';assert marker not in prompt;fields.append((marker,rc));m['reasoning_content']=marker
  assert len(fields)>=1
  marked,_=enc(markers,prepared,prepared.get('add_generation_prompt',True),kwargs);counts=[marked.count(m)for m,_ in fields];positions=[marked.index(m)if m in marked else-1 for m,_ in fields];restored=marked;trimmed=marked
  for m,rc in fields:restored=restored.replace(m,rc);trimmed=trimmed.replace(m,rc.strip())
  actual=json.loads(decoded.read_bytes());usage=actual['usage'];entry={'case':case,'request_SHA256':hashlib.sha256(raw).hexdigest(),'decoded_SHA256':sha(decoded),'actual_reported_prompt_tokens':usage['prompt_tokens'],'replayed_prompt_tokens':len(ids),'actual_prompt_token_count_matches':len(ids)==usage['prompt_tokens'],'actual_prompt_SHA256':hashlib.sha256(prompt.encode()).hexdigest(),'replayed_prompt_ids_SHA256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'historical_reasoning_fields':len(fields),'one_marker_contribution_each':all(n==1 for n in counts),'contribution_order_preserved':positions==sorted(positions),'actual_render_reconstructed_with_raw_RC':restored==prompt,'actual_render_reconstructed_with_stripped_RC':trimmed==prompt,'caller_unchanged':before==prepared,'budget':prepared['max_tokens'],'full_prompt_plus_requested_budget_fits':len(ids)+prepared['max_tokens']<=32768,'source_constraints_and_protocol_unchanged':True};q['cases'].append(entry)
  assert entry['actual_prompt_token_count_matches']and entry['one_marker_contribution_each']and entry['contribution_order_preserved']and (entry['actual_render_reconstructed_with_raw_RC']or entry['actual_render_reconstructed_with_stripped_RC'])and entry['full_prompt_plus_requested_budget_fits']
 assert 'torch'not in sys.modules and '_infinilm'not in sys.modules;q['completed']=True
except BaseException as e:q.update(error_class=type(e).__name__,error_SHA256=hashlib.sha256(str(e).encode()).hexdigest())
q['actual_step']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');q['scope']='Finite exact-source CPU tokenizer replay of genuine saved HTTP/SSE request005 with controlled RC contribution markers; original token count checked against decoded usage. No new Native execution/hash instrumentation or all-history proof; full current matrix/normal cleanup still required.'
with out.open('xb')as f:f.write((json.dumps(q,indent=2)+'\n').encode())
print(json.dumps(q));assert q['completed']
