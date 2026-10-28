"""Offline1203 evidence only. Never imports Torch/Native, never sends HTTP."""
import ast,asyncio,collections,copy,enum,hashlib,importlib.util,json,os,pathlib,re,sys,types
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;OLD=R/'incoming/Deep-agent-regular32K-r182';RESULT=R/'results/Deep-agent-regular32K-r182-910374.1203.json';CP=R/'candidates/Deep-agent-regular32K-r182/CPU-preflight.json';ROOT=RESULT.with_suffix('');ALLOWED={'read_file','calculate','write_file'}
sha=lambda b:hashlib.sha256(b).hexdigest()
def packed(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def file_sha(p):return sha(pathlib.Path(p).read_bytes())
def no_backend():assert not any(x in sys.modules for x in ['torch','transformers','_infinilm']), 'Forbidden backend import'
def load(name,path,package=False):
 spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)]if package else None);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);no_backend();return m
def method(path,cls,name,namespace):
 tree=ast.parse(path.read_bytes());c=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==cls);f=copy.deepcopy(next(x for x in c.body if isinstance(x,ast.FunctionDef)and x.name==name));f.decorator_list=[];f.returns=None
 for a in [*f.args.posonlyargs,*f.args.args,*f.args.kwonlyargs]:a.annotation=None
 if f.args.vararg:f.args.vararg.annotation=None
 if f.args.kwarg:f.args.kwarg.annotation=None
 t=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]));exec(compile(t,str(path),'exec'),namespace);return namespace[name],sha(ast.dump(f).encode())
def safe_error(exc):
 text=str(exc);known=['Model did not satisfy tool_choice','Multiple calls with parallel_tool_calls=false','Truncated tool call','Tool call inside reasoning','Unknown tool call','Malformed structured tool call','Malformed or unknown tool call','Invalid format: missing EOS token','Invalid thinking format: missing </think>']
 return {'class':type(exc).__name__ if type(exc).__name__in ['AgentProtocolError','ValueError','AssertionError','RuntimeError','KeyError']else'OTHER','static_message':next((x for x in known if text==x),'OTHER'),'message_SHA256':sha(text.encode())}
def names(calls):return [c.get('function',{}).get('name')if c.get('function',{}).get('name')in ALLOWED else'OTHER'for c in calls]
def normalize(value):
 if isinstance(value,dict):return {k:normalize(v)for k,v in value.items()if k not in ['id','index']}
 if isinstance(value,list):return [normalize(v)for v in value]
 return value
def parse_run(Parser,collect,schemas,choice,parallel,thinking,chunks,split_final=False):
 p=Parser(schemas,thinking=thinking,choice=choice,parallel=parallel,parser='deepseek_v4');events=[];err=None
 try:
  chunks=list(chunks)
  if split_final:
   for text in chunks:events.extend(p.feed(text))
   events.extend(p.feed('',final=True))
  else:
   for i,text in enumerate(chunks):events.extend(p.feed(text,final=i==len(chunks)-1))
 except Exception as e:err=safe_error(e)
 value=normalize(collect(events));calls=normalize(p.calls);payload=[packed(c)for c in calls];counts=collections.Counter(payload)
 summary={'call_count':len(calls),'call_names':names(calls),'same_payload_duplicate_count':sum(v-1 for v in counts.values()),'calls_SHA256':sha(packed(calls)),'message_SHA256':sha(packed(value)),'error':err,'remaining_buffer_chars':len(p.buffer),'parser_mode_is_tool':p.mode=='tool'}
 return summary,{'message':value,'calls':calls,'events':events,'error':err}
def summary_equal(a,b):return all(a[k]==b[k]for k in ['call_count','call_names','same_payload_duplicate_count','calls_SHA256','message_SHA256','error'])
def pin_sources():
 mf=OLD/'manifest-r182.json';assert file_sha(mf)=='2fbc2b355d0c6342f33c163da3542077d1a7160ca9268c3f2cd29092e5534f22';m=json.loads(mf.read_bytes())
 for p,h in m['source_SHA256'].items():assert file_sha(OLD/p)==h
 assert file_sha(CP)=='0bed44bae0bbb1775eed220f29f5798ac65cd1c4c9b1ce2ab0a89af2ed00c2f5';cpu=json.loads(CP.read_bytes());assert cpu['completed']and not cpu['Torch_imported']
 for p,h in cpu['overlay_files_SHA256'].items():assert file_sha(p)==h
 return m,cpu
def modules():
 # Namespace shells avoid infinilm.__init__ and llm/native imports. Parser bytes unchanged.
 pkg=types.ModuleType('r195_exact');pkg.__path__=[];sys.modules[pkg.__name__]=pkg
 protocol=load('r195_exact.agent_protocol',OLD/'overlay/python/infinilm/server/agent_protocol.py')
 parser=load('r195_exact.tool_parsers',OLD/'overlay/python/infinilm/server/tool_parsers/__init__.py',True)
 bridge=load('r195_bridge',OLD/'native_engine_bridge_r143c.py');wire=load('r195_wire',OLD/'wire_client_r182.py');prep=load('r195_prepare',OLD/'chat_prepare_r173.py')
 return protocol,parser,bridge,wire,prep
def exact_tokenizer(cpu):
 from tokenizers import Tokenizer
 model=pathlib.Path(cpu['model_directory']);tp=model/'tokenizer.json';assert file_sha(tp)==cpu['profiles']['tokenizer_json_SHA256'];backend=Tokenizer.from_file(str(tp))
 # Execute ONLY the installed HF fast-tokenizer _decode method AST. This is
 # the real server method, not an alternative decoding/cleanup policy.
 spec=importlib.util.find_spec('transformers');assert spec and spec.origin;source=pathlib.Path(spec.origin).with_name('tokenization_utils_fast.py');fn,astsha=method(source,'PreTrainedTokenizerFast','_decode',{})
 class TokenizerAdapter:
  def __init__(self):self._tokenizer=backend;self.clean_up_tokenization_spaces=False
  def encode(self,text,add_special_tokens=False):assert add_special_tokens is False;return self._tokenizer.encode(text,add_special_tokens=False).ids
  def decode(self,ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
   assert skip_special_tokens is False and clean_up_tokenization_spaces is False
   return fn(self,list(ids),skip_special_tokens=False,clean_up_tokenization_spaces=False)
 tok=TokenizerAdapter();assert tok.decode([1])=='<｜end▁of▁sentence｜>';no_backend()
 return tok,{'tokenizer_json_SHA256':file_sha(tp),'installed_HF_fast_source_SHA256':file_sha(source),'executed_exact_HF_decode_AST_SHA256':astsha,'HF_module_imported':False,'skip_special_tokens':False,'clean_up_tokenization_spaces':False}
def enum_reason(cpu):
 suffix='/python/infinilm/llm/request.py';paths=[pathlib.Path(p)for p in cpu['baseline_files_SHA256']if p.endswith(suffix)];assert len(paths)==1;p=paths[0];assert file_sha(p)==cpu['baseline_files_SHA256'][str(p)]
 t=ast.parse(p.read_bytes());c=next(x for x in t.body if isinstance(x,ast.ClassDef)and x.name=='FinishReason');ns={'Enum':enum.Enum};exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),str(p),'exec'),ns);return ns['FinishReason'],file_sha(p)
def wire_records(protocol,prepare,tok):
 paths=[]
 for directory in [ROOT/'regular'/x for x in ['HTTP','SSE','thinking','negative']]+[ROOT/'long-output']:
  if directory.is_dir():paths.extend(sorted(directory.glob('request-*.request.json')))
 rows=[]
 for path in paths:
  raw=path.read_bytes();body=json.loads(raw);row={'path':path,'body':body,'SHA256':sha(raw),'wire_path':path.with_name(path.name.removesuffix('.request.json')+'.wire')}
  try:
   data,schemas,choice=protocol.prepare_agent_request(body,'deepseek_v4');data['_agent_schemas']=schemas;data['_agent_choice']=choice;prompt,ids=prepare.prepare(data['messages'],data,bool(data.get('add_generation_prompt',True)),data['chat_template_kwargs'],tok,32768,data.get('max_tokens',4096));row.update(data=data,schemas=schemas,choice=choice,prompt=prompt,prompt_ids=ids)
  except Exception as e:row['preparation_error']=safe_error(e)
  if row['wire_path'].is_file():
   wr=row['wire_path'].read_bytes();row['wire_SHA256']=sha(wr);row['wire']=wr
   row['wire_request_ids']=set()
   for line in wr.decode(errors='replace').splitlines():
    try:
     q=json.loads(line[6:]if line.startswith('data: ')else line)
     if isinstance(q,dict)and isinstance(q.get('id'),str):row['wire_request_ids'].add(q['id'])
    except Exception:pass
  rows.append(row)
 return rows
async def bridge_replay(bridge,tok,reasons,record,prompt):
 ids=record['generated_token_ids'];req=bridge.NativeRequest(record['request_id'],prompt,[],record['max_new_tokens'],(1,),reasons)
 for value in ids:req.emit_token(value)
 req.finish(reasons(record['transport_finish_reason']));engine=bridge.NativeEngineBridge(tok,32768,reasons,eos_token_ids=(1,),default_max_tokens=4096);outputs=[]
 async for output in engine.stream_request(req,timeout=1800,request_timeout=7200):outputs.append(output)
 assert len(outputs)==1 and outputs[0].finished
 return outputs[0].token_text,req.visible_token_ids
def main():
 no_backend();assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 own=json.loads((D/'manifest-r195.json').read_bytes())
 for p,h in own['source_SHA256'].items():assert file_sha(D/p)==h
 m,cpu=pin_sources();protocol,parser,bridge,wire,prep=modules();tok,tp=exact_tokenizer(cpu);reasons,rs=enum_reason(cpu)
 server=OLD/'overlay/python/infinilm/server/inference_server.py';fns={'ToolCallParser':parser.ToolCallParser,'FinishReason':reasons};agent_parser,aps=method(server,'InferenceServer','_agent_parser',fns);convert,cs=method(server,'InferenceServer','_convert_finish_reason',fns)
 # Verify actual route operations before reproducing their tiny glue below.
 st=ast.parse(server.read_bytes());sc=next(x for x in st.body if isinstance(x,ast.ClassDef)and x.name=='InferenceServer');dump=ast.dump(sc);assert "attr='strip'"in dump and "attr='feed'"in dump
 raw=RESULT.read_bytes();snapshot=json.loads(raw);assert snapshot['actual_step']=='910374.1203'and snapshot['Native_SHA256']=='232fb7e2566f7d1961fa433d9e45660bfe760ec726676dcb7f1f406d857c801f';rows=wire_records(protocol,prep,tok)
 tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'local-only'/('Deep-offline-replay-r195-'+tag);out.mkdir(mode=0o700,parents=True,exist_ok=False);private=[]
 report={'completed':False,'CPU_only':True,'Torch_imported':False,'Native_loaded':False,'weights_opened':False,'Model_generated':False,'input_SHA256':sha(raw),'input_changed_since_snapshot':False,'r182_manifest_SHA256':file_sha(OLD/'manifest-r182.json'),'CPU_prepare_receipt_SHA256':file_sha(CP),'tokenizer':tp,'FinishReason_source_SHA256':rs,'agent_parser_AST_SHA256':aps,'finish_reason_AST_SHA256':cs,'source_manifest_SHA256':file_sha(D/'manifest-r195.json'),'actual_bridge_visible_text_frames_per_request':1,'per_token_comparison_is_actual_transport':False,'requests':[],'wire_source_SHA256':[{'request_SHA256':x['SHA256'],'wire_SHA256':x.get('wire_SHA256')}for x in rows]}
 for i,rec in enumerate(snapshot['requests']):
  row={'ordinal':i+1,'prompt_tokens':rec['prompt_tokens'],'max_new_tokens':rec['max_new_tokens'],'actual_generated_tokens':rec.get('actual_generated_tokens'),'normal_reset_before':bool(rec.get('normal_reset_before')),'normal_reset_after':bool(rec.get('normal_reset_after'))};report['requests'].append(row)
  if 'generated_token_ids'not in rec:row['incomplete_record']=True;continue
  ids=rec['generated_token_ids'];assert len(ids)==rec['actual_generated_tokens']and all(type(x)is int and x>=0 for x in ids);row['generated_ids_SHA256']=sha(packed(ids))
  candidates=[x for x in rows if rec['request_id']in x.get('wire_request_ids',set())]
  basis='exact_wire_response_request_id'
  if not candidates:
   candidates=[x for x in rows if 'data'in x and x['body'].get('user')==rec['admission'].get('diagnostic_user')and len(x['prompt_ids'])==rec['prompt_tokens']and x['body'].get('max_tokens',4096)==rec['max_new_tokens']and x['data']['chat_template_kwargs']==rec['admission']['chat_template_kwargs']];basis='unique_user_prompt_count_budget_full_template_kwargs'
  row['mapping_candidate_count']=len(candidates);row['mapping_exact_response_id']=basis=='exact_wire_response_request_id'
  if len(candidates)!=1:row['mapping_ambiguous_or_missing']=True;continue
  w=candidates[0];data=w['data'];prompt=w['prompt'];schemas=w['schemas'];choice=w['choice'];thinking=prompt.rstrip().endswith(('<think>','<|open|>think<|sep|>'));parallel=data.get('parallel_tool_calls',True)
  converted=convert(types.SimpleNamespace(),reasons(rec['transport_finish_reason']));assert converted in ['stop','length','error','timeout','canceled']
  row.update(request_source_SHA256=w['SHA256'],wire_SHA256=w.get('wire_SHA256'),prepared_prompt_SHA256=sha(prompt.encode()),prepared_prompt_ids_SHA256=sha(packed(w['prompt_ids'])),prompt_count_matches=len(w['prompt_ids'])==rec['prompt_tokens'],named_choice=choice if choice in ALLOWED else'OTHER',parallel_enabled=parallel,thinking=thinking,schema_tool_names=sorted(x if x in ALLOWED else'OTHER'for x in schemas),actual_native_EOS=rec['native_finish_reason']=='eos_token',actual_native_length=rec['native_finish_reason']=='length',server_finish_is_stop=converted=='stop',server_finish_is_length=converted=='length')
  text,visible=asyncio.run(bridge_replay(bridge,tok,reasons,rec,prompt));assert visible==[x for x in ids if x!=1];row.update(decoded_text_SHA256=sha(text.encode()),decoded_chars=len(text),EOS_token_count=ids.count(1),EOS_is_last=bool(ids and ids[-1]==1),visible_ids_SHA256=sha(packed(visible)))
  req=types.SimpleNamespace(prompt=prompt);p=agent_parser(types.SimpleNamespace(tool_call_parser='deepseek_v4'),data,req);assert p.choice==choice and p.parallel==parallel and p.mode==('reasoning_content'if thinking else'content')
  http,httpfull=parse_run(parser.ToolCallParser,protocol.collect_message,schemas,choice,parallel,thinking,[text.strip()]);sse,ssefull=parse_run(parser.ToolCallParser,protocol.collect_message,schemas,choice,parallel,thinking,[text],True)
  relaxed,relaxedfull=parse_run(parser.ToolCallParser,protocol.collect_message,schemas,'auto',True,thinking,[text.strip()]);row.update(actual_HTTP_replay=http,actual_SSE_replay=sse,HTTP_SSE_equivalent=summary_equal(http,sse),relaxed_cardinality_diagnostic=relaxed)
  # Prefixes use exact HF decode for every actual visible-token prefix. This
  # contrast is not the buffered actual transport and never invents a delta.
  prefix_errors=collections.Counter();previous='';deltas=[];monotonic=True;first_call=None
  for j in range(1,len(visible)+1):
   value=tok.decode(visible[:j],skip_special_tokens=False,clean_up_tokenization_spaces=False);s,_=parse_run(parser.ToolCallParser,protocol.collect_message,schemas,choice,parallel,thinking,[value])
   if s['error']:prefix_errors[s['error']['static_message']]+=1
   if s['call_count']and first_call is None:first_call=j
   if value.startswith(previous):deltas.append(value[len(previous):])
   else:monotonic=False
   previous=value
  row.update(per_token_prefix_count=len(visible),prefix_first_accepted_call_token=first_call,prefix_final_text_equivalent=previous==text,prefix_decode_append_monotonic=monotonic,prefix_error_counts=dict(prefix_errors))
  if monotonic:
   inc,incfull=parse_run(parser.ToolCallParser,protocol.collect_message,schemas,choice,parallel,thinking,deltas,True);row.update(token_delta_replay=inc,token_delta_vs_actual_SSE_equivalent=summary_equal(inc,sse))
  char,chfull=parse_run(parser.ToolCallParser,protocol.collect_message,schemas,choice,parallel,thinking,list(text),True);row.update(character_partition_replay=char,character_partition_vs_actual_SSE_equivalent=summary_equal(char,sse))
  markers=['<think>','</think>','<｜DSML｜tool_calls>','</｜DSML｜tool_calls>','<｜end▁of▁sentence｜>'];row['control_symbols']=[{'count':text.count(x),'at_head':text.startswith(x),'at_tail':text.endswith(x)}for x in markers];row['control_symbols_SHA256']=sha(packed(markers));rawnames=re.findall(r'<｜DSML｜invoke\s+name="([^"<>]+)"',text);row['raw_invoke_count']=len(rawnames);row['raw_invoke_names']=[x if x in ALLOWED else'OTHER'for x in rawnames]
  # Public preprocessing filters Deep named schemas; test constraint encoding
  # independently without changing the original request or generating tokens.
  def contrast(body):
   d,ss,cc=protocol.prepare_agent_request(body,'deepseek_v4');pr,ii=prep.prepare(d['messages'],d,True,d['chat_template_kwargs'],tok,32768,d.get('max_tokens',4096));return pr,ss
  opposite=copy.deepcopy(w['body']);opposite['parallel_tool_calls']=not parallel;other,_=contrast(opposite);auto=copy.deepcopy(w['body']);auto['tool_choice']='auto';ap,ass=contrast(auto)
  row['prompt_constraints']={'parallel_flag_changes_prompt':other!=prompt,'named_vs_auto_changes_prompt':ap!=prompt,'auto_schema_tool_count':len(ass),'actual_schema_tool_count':len(schemas),'explicit_choice_or_parallel_forwarded_to_official_encoder':False,'named_schema_filtering_is_encoded':choice in ALLOWED and list(schemas)==[choice],'official_template_invoke_examples':prep.encoder.TOOLS_TEMPLATE.count('invoke name=')}
  if w.get('wire'):
   try:
    if w['body'].get('stream'):message,finish,usage=wire.sse_message(w['wire'])
    else:
     q=json.loads(w['wire']);message=q['choices'][0]['message'];finish=q['choices'][0]['finish_reason']
    row['wire_decoded_success']=True;row['wire_message_SHA256']=sha(packed(normalize(message)));row['wire_matches_replay_message']=sha(packed(normalize(message)))==(sse if w['body'].get('stream')else http)['message_SHA256']
   except Exception as e:row['wire_decoded_success']=False;row['wire_decode_error']=safe_error(e)
  private.append({'ordinal':i+1,'prompt':prompt,'text':text,'actual_HTTP':httpfull,'actual_SSE':ssefull,'relaxed':relaxedfull,'character_partition':chfull})
 report.update(completed=True,input_changed_since_snapshot=file_sha(RESULT)!=sha(raw),Torch_imported='torch'in sys.modules,Transformers_imported='transformers'in sys.modules,Native_loaded='_infinilm'in sys.modules,normal_CPU_exit=True);no_backend()
 (out/'private-full-replay.json').write_bytes(packed(private));os.chmod(out/'private-full-replay.json',0o600);(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,separators=(',',':')))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  print(json.dumps({'completed':False,'CPU_only':True,'Torch_imported':'torch'in sys.modules,'Native_loaded':'_infinilm'in sys.modules,'error':safe_error(e)},separators=(',',':')));raise SystemExit(1)
