"""CPU actual HTTP tests of optional resident adapter; never calls native/GPU."""
import pathlib,json,hashlib,os,time,urllib.request,urllib.error,concurrent.futures,socket,struct,threading,signal
signal.alarm(140)
from transformers import PreTrainedTokenizerFast
from resident_api_r9118b import ResidentAPI
r=pathlib.Path('/refactor');src=r/'results/deep-quality-bench-r9101-910374.1033.json';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert sha(src)=='8490d51432f3401b6f30d68376ff5868a525f28444377be618f2b3bb015e1d1c';q=json.loads(src.read_bytes());records=[v for c in q['service_native_cohorts']for v in c['records']];tok=PreTrainedTokenizerFast.from_pretrained('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8',local_files_only=True);api=ResidentAPI(tok,encode_messages=lambda m:tok.encode(m[0]['content'],add_special_tokens=False),eos_ids=[1],queue_capacity=1,max_tokens=64);pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
def body(rid,**extra):
 return dict(model=api.model,request_id=rid,messages=[{'role':'user','content':'CPU replay input'}],max_tokens=64,temperature=0,stream=True,**extra)
def client(data):
 req=urllib.request.Request(api.url,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'});z={'status':None,'text':'','token_ids':[],'done':False,'finish':None}
 try:
  with urllib.request.urlopen(req,timeout=10)as resp:
   z['status']=resp.status
   for raw in resp:
    if not raw.startswith(b'data: '):continue
    raw=raw[6:].strip()
    if raw==b'[DONE]':z['done']=True;break
    e=json.loads(raw)
    if 'error'in e:z['error']=e['error'];continue
    c=e['choices'][0];z['text']+=c.get('delta',{}).get('content','')
    if c.get('finish_reason'):z['finish']=c['finish_reason']
    if 'token_id'in e and not e.get('text_only'):z['token_ids'].append(e['token_id'])
 except urllib.error.HTTPError as e:z['status']=e.code;z['error']=e.read().decode()
 return z
def awaitqueue():
 end=time.monotonic()+5
 while api.jobs.qsize()==0:
  assert time.monotonic()<end;time.sleep(.005)
checks={};details=[]
try:
 invalid=body('badtemp');invalid['temperature']=.5;checks['nongreedy_400']=client(invalid)['status']==400
 invalid=body('badlength');invalid['messages'][0]['content']='abcdef '*600;checks['prompt_overflow_400']=client(invalid)['status']==400
 invalid=body('badfield');invalid['tools']=[{}];checks['unknown_fields_400']=client(invalid)['status']==400
 chosen=[next(v for v in records if v['stop_reason']=='EOS'),next(v for v in records if v['stop_reason']=='max_new_tokens')]
 for j,rec in enumerate(chosen):
  rid='replay'+str(j);future=pool.submit(client,body(rid));awaitqueue()
  if j==0:checks['full_queue_503']=client(body('overflow'))['status']==503
  request=api.next_request(timeout=1);assert request['id']==rid;case=api.to_case(request,mode='adaptive',draft='dflash2',K=7);assert case['schema']==1 and case['cap']==64 and case['K']==7
  bad=pool.submit(api.next_request,0)
  try:bad.result(timeout=2);checks['owner_thread_enforced']=False
  except RuntimeError:checks['owner_thread_enforced']=True
  api.emit_tokens(rid,[42],kind='draft');api.emit_tokens(rid,[42],kind='replay');api.emit_tokens(rid,[0],kind='pad')
  ids=rec['generated_ids'];api.emit_tokens(rid,ids,final=rec['stop_reason']=='max_new_tokens');api.finish_request(rid,'stop'if rec['stop_reason']=='EOS'else'length');out=future.result(timeout=10);expected=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);assert out['status']==200 and out['done']and out['text']==expected and out['token_ids']==[t for t in ids if t!=1];details.append({'id':rid,'stop':out['finish'],'text_match':True,'token_ids_match':True,'text':out['text'],'useful_tokens':len(out['token_ids'])})
  checks['EOS_tail'if j==0 else'cap_flush']=True
 # True socket reset after admission, then owner emits/finishes safely at boundary.
 data=json.dumps(body('disconnect')).encode();sock=socket.create_connection(('127.0.0.1',api.server.server_port),timeout=3);sock.sendall(('POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '+str(len(data))+'\r\nConnection: close\r\n\r\n').encode()+data);assert b'200'in sock.recv(2048);sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));sock.close();awaitqueue();req=api.next_request(timeout=1);assert req['id']=='disconnect';api.emit_tokens(req['id'],[671,850]);api.finish_request(req['id'],'cancelled');end=time.monotonic()+2
 while not api.records['disconnect']['cancelled'].is_set()and time.monotonic()<end:time.sleep(.01)
 checks['disconnect_observed']=api.records['disconnect']['cancelled'].is_set();checks['active_cleared_at_boundary']=api.active is None
 assert all(checks.values()),checks
finally:api.close();pool.shutdown(wait=True)
report={'passed':True,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'Model_called':False,'GPU_initialized':False,'actual_localhost_HTTP_SSE':True,'not_Model_performance':True,'checks':checks,'valid_outputs':details,'r9116_native_hook_available':False,'integration_status':'OPTIONAL_ADAPTER_ONLY; do not call case-end results streaming; new stable emitter/leader sink needed after frozen r9116','source_SHA256':{n:sha(pathlib.Path(__file__).parent/n)for n in ['resident_api_r9118b.py','incremental_text_r9118.py']}}
out=r/'results'/('Kimi-serving-API-CPU-r9118-'+report['producer']+'.json');assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'passed':True,'producer':report['producer'],'path':str(out),'SHA256':sha(out),'checks':checks,'CPU_only':True,'not_Model_performance':True}))
