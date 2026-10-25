"""CPU real tokenizer + actual localhost HTTP/SSE replay of prior Model1033 IDs."""
import pathlib,json,hashlib,os,time,signal
signal.alarm(140)
from transformers import PreTrainedTokenizerFast
from incremental_text_r9118 import IncrementalTextDecoder
from service_transport_replay_r9118 import run_local_service_benchmark
r=pathlib.Path('/refactor');source=r/'results/deep-quality-bench-r9101-910374.1033.json';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert sha(source)=='8490d51432f3401b6f30d68376ff5868a525f28444377be618f2b3bb015e1d1c';q=json.loads(source.read_bytes());records={v['id']:v for c in q['service_native_cohorts']for v in c['records']};assert set(records)==set(map(str,range(8)))
model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');tok=PreTrainedTokenizerFast.from_pretrained(str(model),local_files_only=True);assert tok.eos_token_id==1
native_trace=[]
def adapter(items,emit):
 for i,item in enumerate(items):
  rec=records[item['id']];ids=rec['generated_ids'];dec=IncrementalTextDecoder(tok,eos_ids=[1]);useful=[]
  for j,token in enumerate(ids):
   eos=token==1;piece,count=dec.push(token,eos=eos,final=j==len(ids)-1)
   if count:useful.append(token)
   emit(i,token,piece,eos=eos)
   if eos:break
  tail=dec.finish();assert not tail,'finalcap/EOS must already flush'
  expected=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);assert dec.emitted==expected
  native_trace.append({'id':item['id'],'expected':expected,'decoder_output':dec.emitted,'useful_token_ids':useful,'stop_reason':rec['stop_reason']})
prompts=[v['prompt']for v in q['online_service']['requests']];result=run_local_service_benchmark(adapter,prompts,max_batch=4,max_new_tokens=512,request_rate=100000,deadline_seconds=30);byid={c['id']:c for c in result['requests']};checks=[]
for n in native_trace:
 c=byid[n['id']];ids=[e['token_id']for e in c['events']];checks.append({'id':n['id'],'completed':c['status']=='completed','text_match':c['text']==n['expected'],'IDs_match':ids==n['useful_token_ids'],'native_stop_reason':n['stop_reason'],'client_stop_reason':c['stop_reason'],'expected_chars':len(n['expected']),'actual_chars':len(c['text']),'expected':n['expected'],'client':c['text']})
assert len(checks)==8 and all(c['completed']and c['text_match']and c['IDs_match']for c in checks)
unicode_cases=[]
for text in ['hello world\nnext line after newline','中文测试，逐字输出。你好！','日本語テスト 한국어 테스트','🙂🚀 café é 中文 emoji 👨‍👩‍👧‍👦',' leading  spaces\n\n trailing  ','right-to-left العربية עברית','literal replacement � text']:
 ids=tok.encode(text,add_special_tokens=False);expected=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 for eos in [False,True]:
  dec=IncrementalTextDecoder(tok,eos_ids=[1]);parts=[]
  for j,t in enumerate(ids):piece,_=dec.push(t,final=not eos and j==len(ids)-1);parts.append(piece)
  if eos:piece,count=dec.push(1,eos=True);assert not count;parts.append(piece)
  parts.append(dec.finish());assert ''.join(parts)==expected;unicode_cases.append({'input':text,'with_EOS':eos,'tokens':len(ids),'match':True,'output':expected})
# Non-committed work never enters either visible token count or decoded output.
dec=IncrementalTextDecoder(tok,eos_ids=[1],pad_ids=[0]);assert dec.push(42,'draft')==('',False)and dec.push(42,'replay')==('',False)and dec.push(0,'pad')==('',False);assert dec.finish()==''
report={'passed':True,'CPU_only':True,'Model_loaded':False,'GPU_initialized':False,'actual_HTTP_SSE':True,'not_Model_performance':True,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'original_Model_producer_SHA256':sha(source),'tokenizer_files_SHA256':{n:sha(model/n)for n in ['tokenizer.json','tokenizer_config.json']},'actual_replay_checks':checks,'actual_replay_passed':len(checks),'unicode_cases':unicode_cases,'noncommitted_exclusion_pass':True,'service':result,'decoder_contract':'cleanup=False; emitted prefix invariant; end=max(printed,computed); partialU+FFFD buffered; whitespace/CJK/newline stable boundary; EOS/cap final flush; textonly EOS tail not counted'}
out=r/'results'/('Kimi-serving-CPU-replay-r9118-'+report['producer']+'.json');assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'passed':True,'path':str(out),'SHA256':sha(out),'producer':report['producer'],'replay8of8':True,'unicode14of14':True,'CPU_only':True,'not_Model_performance':True}))
