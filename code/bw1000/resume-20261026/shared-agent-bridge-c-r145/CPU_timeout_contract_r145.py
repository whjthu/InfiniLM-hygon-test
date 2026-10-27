"""Real bridge C timers with fake owner. No HTTP/Native/model qualification."""
import asyncio,enum,hashlib,importlib.util,json,os,pathlib,threading,time,types
D=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('bridge_c',D/'native_engine_bridge_r143c.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
class Reasons(enum.Enum):
 EOS_TOKEN='eos_token';STOP_STRING='stop_string';STOP='stop';LENGTH='length';CANCELED='canceled';TIMEOUT='timeout';ERROR='error'
class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):return 'fixture'
 def encode(self,text,**kwargs):return [1,2,3]
 def decode(self,ids,**kwargs):return ''.join(chr(x)for x in ids)
async def check():
 passed=[];e=b.NativeEngineBridge(Tokenizer(),256,Reasons)
 def new(name):
  q=e.add_chat_request(messages=[],sampling_params=types.SimpleNamespace(temperature=0.,max_tokens=128,stop=None),request_id=name);assert e.take()is q;return q
 q=new('idle');items=[x async for x in e.stream_request(q,timeout=.02,request_timeout=1)]
 assert items[-1].finish_reason is Reasons.TIMEOUT and not q.is_finished()and q.finish_reason is None and q.abort_reason is Reasons.TIMEOUT
 e.add_aborted_req(q,Reasons.CANCELED);assert q.abort_reason is Reasons.TIMEOUT and not q.is_finished();q.metadata['original_reset_completed']=True;q.finish(Reasons.EOS_TOKEN);assert q.finish_reason is Reasons.TIMEOUT;passed+=['idle_timeout_not_overwritten_by_finally','finish_only_after_owner_reset']
 q=new('continuous');stop=threading.Event()
 def pump():
  while not stop.is_set():q.emit_token(ord('T'));time.sleep(.001)
 t=threading.Thread(target=pump);t.start();start=time.monotonic()
 try:items=await asyncio.wait_for(_collect(e.stream_request(q,timeout=1,request_timeout=.04)),timeout=.3)
 finally:stop.set();t.join()
 assert time.monotonic()-start<.25 and items[-1].finish_reason is Reasons.TIMEOUT and q.abort_reason is Reasons.TIMEOUT and q.finish_reason is None and not q.is_finished()and not items[-1].token_text
 q.metadata['original_reset_completed']=True;q.finish(Reasons.CANCELED);assert q.finish_reason is Reasons.TIMEOUT;passed.append('absolute_deadline_with_continuous_progress_and_no_partial_tool_text')
 q=new('first_error');e.add_aborted_req(q,Reasons.ERROR);e.add_aborted_req(q,Reasons.TIMEOUT);e.add_aborted_req(q,Reasons.CANCELED);assert q.abort_reason is Reasons.ERROR and not q.is_finished();q.finish(Reasons.EOS_TOKEN);assert q.finish_reason is Reasons.ERROR;passed.append('first_ERROR_preserved')
 q=new('first_timeout');e.add_aborted_req(q,Reasons.TIMEOUT);q.fail('later error');assert q.finish_reason is Reasons.TIMEOUT and q.metadata['native_error']=='later error';passed.append('first_TIMEOUT_preserved_native_error_still_recorded')
 e.stop();assert e.take()is None
 return passed
async def _collect(iterator):return [x async for x in iterator]
if __name__=='__main__':
 checks=asyncio.run(check());out={'passed':True,'CPU_fake_owner_only':True,'HTTP':False,'Native':False,'model':False,'checks':checks,'bridge_SHA256':hashlib.sha256((D/'native_engine_bridge_r143c.py').read_bytes()).hexdigest()};target=pathlib.Path(os.environ.get('BRIDGE_CPU_OUTPUT_R145',str(D)));target.mkdir(parents=True,exist_ok=True);(target/'CPU-timeout-contract-r145.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
