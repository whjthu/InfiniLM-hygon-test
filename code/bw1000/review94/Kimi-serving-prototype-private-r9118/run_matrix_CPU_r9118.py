import pathlib,json,os,time,concurrent.futures,socket,threading,runpy,hashlib,signal
signal.alarm(145)
from transformers import PreTrainedTokenizerFast
from resident_api_r9118b import ResidentAPI
from client_load_r9118 import run_load,total_cards
from committed_event_wire_r9118 import CommittedEventSender,recv_frame
root=pathlib.Path('/refactor');runpy.run_path(str(pathlib.Path(__file__).with_name('run_API_CPU_r9118b.py')),run_name='__main__');tok=PreTrainedTokenizerFast.from_pretrained('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8',local_files_only=True);api=ResidentAPI(tok,encode_messages=lambda m:tok.encode(m[0]['content'],add_special_tokens=False),eos_ids=[1],queue_capacity=8,max_tokens=64);prompts=[{'id':str(i),'prompt':'CPU transport test '+str(i)}for i in range(8)];results=[];pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
try:
 for C in [1,2,4,8,16,32]:
  scenario={'name':'cpu-C'+str(C),'kind':'closed_loop','concurrency':C,'requests':max(2,C)};future=pool.submit(run_load,[api.url],scenario,prompts,33,4,20261026,15)
  while not future.done():
   req=api.next_request(timeout=.01)
   if req is not None:api.emit_tokens(req['id'],[671,850,14738,12855],final=True);api.finish_request(req['id'],'length')
  z=future.result();assert z['metrics']['arrivals']==max(2,C)and z['metrics']['unfinished']==0;assert all('server_queue_seconds'in r for r in z['requests']if r['status']=='completed');results.append(z)
finally:api.close();pool.shutdown(wait=True)
listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);frames=[]
def receiver():
 conn,_=listener.accept()
 with conn:
  for _ in range(3):frames.append(recv_frame(conn))
thread=threading.Thread(target=receiver);thread.start();sender=CommittedEventSender('127.0.0.1',listener.getsockname()[1],'CPU-nonce');assert sender.event('cpu-wire',[671,850],'verified');assert sender.finished('cpu-wire','length');sender.close();thread.join(timeout=3);listener.close();assert len(frames)==3 and frames[0]=={'type':'hello','nonce':'CPU-nonce','schema':1}and frames[1]['seq']==0 and frames[2]['seq']==1;off=CommittedEventSender();assert off.event('cpu-off',[671],'AR')and off.sock is None
report={'passed':True,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU_initialized':False,'Model_called':False,'metrics_NOT_Model_performance':True,'client_C_matrix':[1,2,4,8,16,32],'results':results,'event_wire_pass':True,'event_frames':frames,'DP_card_count_test':total_cards({'replicas':[{'target_cards':32,'draft_cards':1},{'target_cards':32,'draft_cards':2}]})==67,'native_B1_serial':True,'native_stream_hook_currently_available':False};assert report['DP_card_count_test'];p=root/'results'/('Kimi-serving-matrix-CPU-r9118-'+report['producer']+'.json');assert not p.exists();p.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'passed':True,'producer':report['producer'],'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'C':[1,2,4,8,16,32],'CPU_only':True,'not_Model_performance':True}))
