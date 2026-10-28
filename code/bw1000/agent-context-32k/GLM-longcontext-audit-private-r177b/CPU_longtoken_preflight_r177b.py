"""Official-tokenizer long-context workload preparation; zero Native/GPU/model outputs."""
import pathlib,os,sys,json,hashlib,importlib.util,datetime
assert os.environ.get('USE_TORCH')=='0'and os.environ.get('USE_TF')=='0'
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');sys.path.insert(0,str(D));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((D/'source-manifest-r177b.json').read_bytes())
for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
spec=importlib.util.spec_from_file_location('longctx_protocol_r177b',D/'agent_protocol.py');protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
from transformers import PreTrainedTokenizerFast
from wire_and_prefill_r177b import admit,encode,decode,prefill_chunks
model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');cfg=json.loads((model/'config.json').read_bytes());tok=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True)
assert hashlib.sha256(tok.get_chat_template().encode()).hexdigest()=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];O=R/'results'/('GLM-longtoken-r177b-'+step);O.mkdir(exist_ok=False)
TOOLS=[{'type':'function','function':{'name':'read_file','description':'Read a complete JSON document.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}},{'type':'function','function':{'name':'calculate','description':'Add integer amounts.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}},{'type':'function','function':{'name':'write_file','description':'Write a JSON report.','parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content'],'additionalProperties':False}}}]
def make_messages(records):
 messages=[{'role':'system','content':'Audit all eight complete tool documents. Preserve every source record; calculate totals and cite document and record IDs.'},{'role':'user','content':'Read the eight ledger JSON documents and use their complete contents to prepare an audit report, then verify the total against the documented controls.'}]
 for doc in range(8):
  rows=[{'record_id':'ledger-'+str(doc)+'-'+str(j),'department':['engineering','finance','operations','research'][j%4],'period':'2026-'+str(1+j%12).zfill(2),'revenue':10000+doc*113+j*37,'cost':4000+doc*29+j*19,'control_code':'GLM-LONG-RECORD-'+str(doc*100000+j),'note':'Reconcile revenue less cost against the signed department ledger; retain this row in the audit evidence.'}for j in range(records)]
  call='CPU_SYNTHETIC_HISTORY_CALL_'+str(doc);messages.extend([{'role':'assistant','content':'','reasoning_content':'I will read the next complete ledger document before deriving its controls.','tool_calls':[{'id':call,'type':'function','function':{'name':'read_file','arguments':json.dumps({'path':'ledger-'+str(doc)+'.json'})}}]},{'role':'tool','tool_call_id':call,'content':json.dumps({'document_id':'ledger-'+str(doc),'records':rows,'record_count':len(rows)},separators=(',',':'))}])
 messages.append({'role':'user','content':'Use all eight returned documents without dropping rows. Summarize the totals and the reconciliation controls, with source IDs.'});return messages
def render(messages):
 body={'messages':messages,'tools':TOOLS,'tool_choice':'none','temperature':0,'reasoning_effort':'low','chat_template_kwargs':{'clear_thinking':False}};prepared,schemas,choice=protocol.prepare_agent_request(body,'glm5');assert len(prepared['messages'])==len(messages)
 for original,normalized in zip(messages,prepared['messages']):
  assert all(normalized.get(k)==original.get(k)for k in ['role','content','reasoning_content','tool_call_id'])
  assert len(normalized.get('tool_calls',[]))==len(original.get('tool_calls',[]))
  for a,b in zip(original.get('tool_calls',[]),normalized.get('tool_calls',[])):
   assert a['id']==b['id']and a['type']==b['type']and a['function']['name']==b['function']['name']and json.loads(a['function']['arguments'])==b['function']['arguments']
 text=tok.apply_chat_template(prepared['messages'],tokenize=False,add_generation_prompt=True,**prepared['chat_template_kwargs']);return body,tok.encode(text,add_special_tokens=False)
cases=[]
for name,context,floor,output,expect in [('input8K-output4K',32768,8192,4096,True),('input16K-output4K',32768,16384,4096,True),('input16K-thinking8K',32768,16384,8192,True),('overbudget32K-no-crop',32768,30000,4096,False),('extension64K-input32K-output4K',65536,32768,4096,True),('extension64K-input48K-thinking8K',65536,49152,8192,True)]:
 records=1
 while True:
  body,ids=render(make_messages(records))
  if len(ids)>=floor:break
  records+=2;assert records<1024
 body['max_tokens']=output;inputfile=O/(name+'.request.json');inputfile.write_text(json.dumps(body,ensure_ascii=False,indent=2));admitted=True
 try:admit(ids,output,context)
 except ValueError:admitted=False
 assert admitted==expect
 chunks=prefill_chunks(ids,512);assert all(len(q['ids'])<=512 for q in chunks)
 if admitted:assert decode(encode(1,ids,output,context),1,context)==(ids,output)
 cases.append({'name':name,'context':context,'prompt_tokens':len(ids),'output_budget':output,'total_tokens':len(ids)+output,'admitted_by_budget':admitted,'config_supports_declared_context':cfg['max_position_embeddings']>=context,'tool_result_messages':8,'matching_toolCallID_pairs':8,'records_per_document':records,'input_file':str(inputfile),'input_SHA256':sha(inputfile),'input_bytes':inputfile.stat().st_size,'prefill_chunk':512,'prefill_calls':len(chunks),'first_generated_token_only_after_complete_prefill':True,'no_crop_drop_or_model_input_padding':True,'source':'synthetic complete ledger tool-history input, genuine official tokenizer count; NOT actual model-produced tool history','actual_Native_or_model_acceptance':False})
assert not any(n.startswith(('_infinilm','infinicore'))for n in sys.modules)
out={'completed':True,'CPU_only':True,'GPU_initialized':False,'Native':False,'weights_loaded':False,'actual_model_output_generated':False,'actual_tool_execution':False,'actual_step':step,'manifest_SHA256':sha(D/'source-manifest-r177b.json'),'tokenizer_template_SHA256':hashlib.sha256(tok.get_chat_template().encode()).hexdigest(),'model_config_SHA256':sha(model/'config.json'),'model_max_position_embeddings':cfg['max_position_embeddings'],'full_indexer_layers_by_PP2_stage':[sum(x=='full'for x in cfg['indexer_types'][a:b])for a,b in [(0,38),(38,78)]],'cases':cases,'longcontext_service_PASS':False,'normal_agent_PASS':False,'remaining_gaps':['Actual same-Native cache/constructor/GPU memory and long-prefix indexer/TopK support','Actual8-step tool calls and long tool returns generated/executed throughHTTP/SSE','Quality/cap/longgen/EOS/cancel/recovery/C2 isolation at32K','64K capacity is extension only; no allocation or GPU proof']}
dest=O/'CPU-preflight.json';dest.write_text(json.dumps(out,indent=2));print(json.dumps({'completed':True,'CPU_only':True,'path':str(dest),'SHA256':sha(dest),'model_max_position_embeddings':out['model_max_position_embeddings'],'full_indexer_layers_by_PP2_stage':out['full_indexer_layers_by_PP2_stage'],'cases':cases,'actual_model_acceptance':False}))
