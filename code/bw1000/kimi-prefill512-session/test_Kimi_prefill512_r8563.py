"""Actual consumers: eligibility/restoration plus real TCP delayed first ACK.

This verifies control contracts only. No GPU/Engine/model performance claim.
"""
import json,threading,time,socketserver
from pathlib import Path
from kimi_prefill512_scope_r8563 import eligible_job,ScopedProvider,MODE,ELIGIBLE
from online_kimi_provider_r8517 import KimiLoadedAdapter
from online_broker_r8455 import read_frame,send_frame
from kimi_prefill512_live_r8563 import emit_fixed_arrivals,digest
from kimi_prefill512_protocol_r8563 import service_windows
import os

def job(lens,pasts=None,kind='prefill'):
 return dict(kind=kind,pasts=[0]*len(lens)if pasts is None else pasts,inputs=[[4]*n for n in lens])
cases=[(job([512]),True),(job([512]*4),True),(job([512]*16),True),
 (job([1024]),False),(job([512,521]),False),(job([512]*17),False),
 (job([512],[1]),False),(job([1],[512],'decode'),False),(job([]),False)]
for value,expected in cases:assert eligible_job(value)==expected
old_forward=KimiLoadedAdapter.forward;before=dict(os.environ)
try:
 os.environ[MODE]='1';os.environ[ELIGIBLE]='0'
 def fake(self,value):
  assert os.environ[ELIGIBLE]==('1'if eligible_job(value)else'0')
  if value.get('force_error'):raise RuntimeError('fixture')
  return [5]
 KimiLoadedAdapter.forward=fake;a=object.__new__(ScopedProvider)
 for value,_ in cases:
  assert a.forward(value)==[5]and os.environ[ELIGIBLE]=='0'
 value=job([512]);value['force_error']=True
 try:a.forward(value);raise AssertionError('missingfail')
 except RuntimeError:pass
 assert os.environ[ELIGIBLE]=='0'
finally:
 KimiLoadedAdapter.forward=old_forward
 for k in [MODE,ELIGIBLE]:
  if k in before:os.environ[k]=before[k]
  else:os.environ.pop(k,None)

lock=threading.Lock();arrivals=[];ack=[];nonce='CPU-fixture-actual-sockets-8563'
class Handler(socketserver.StreamRequestHandler):
 def handle(self):
  hello=read_frame(self.rfile);assert hello['op']=='hello'and hello['nonce']==nonce
  send_frame(self.request,dict(ok=True))
  req=read_frame(self.rfile);stamp=time.perf_counter();assert req['op']=='arrive'
  with lock:key=len(arrivals);arrivals.append(stamp)
  if key==0:time.sleep(.2) # ACK backlog must not delay other actual senders.
  send_frame(self.request,dict(ok=True,accepted=True,request=key,arrival=stamp))
  with lock:ack.append(time.perf_counter())
class Server(socketserver.ThreadingTCPServer):daemon_threads=True;allow_reuse_address=False
server=Server(('127.0.0.1',0),Handler);t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
try:
 inputs=[[i]*512 for i in range(8)];records=[]
 plan=dict(offsets_s=[i*.001 for i in range(8)],prompt_SHA256=[digest(p)for p in inputs])
 emit_fixed_arrivals((server.server_address,nonce),inputs,plan,records)
 assert len(records)==8 and [v['producer_index']for v in records]==list(range(8))
 assert max(arrivals)-min(arrivals)<.15 # first ACK deliberately0.2s later.
 assert max(v['send_t']for v in records)<min(arrivals)+.15
 assert max(ack)-min(arrivals)>=.2
finally:server.shutdown();server.server_close();t.join(2)
events=[dict(kind='token',t=1.,index=0,eos=True)for _ in range(8)]
zero=service_windows(events);assert zero['all_decode_empty']and zero['service_postfirst_empty']and zero['all_decode_tok_s']is None
events=[dict(kind='token',t=1.+i*.01,index=0,eos=True)for i in range(8)]
empty=service_windows(events);assert empty['all_decode_empty']and empty['service_postfirst_tok_s']is None
events=[dict(kind='token',t=1.,index=0,eos=False)for _ in range(8)]+[dict(kind='token',t=2.,index=1,eos=False)]
normal=service_windows(events);assert not normal['all_decode_empty']and normal['all_decode_tok_s']==1.
protocol=Path('kimi_prefill512_protocol_r8563.py').read_text();assert 'distinct_prompt_count=len({digest(p)for p in inputs})'in protocol
out=dict(CPU_only=True,GPU=False,eligibility_cases=len(cases),restores_after_success_and_failure=True,
 report_window_cases=3,zero_or_no_after_tokens_return_None_and_empty=True,prompt_diversity_report_explicit=True,
 actual_TCP_streams=8,first_ACK_delay_s=.2,actual_ingress_span_s=max(arrivals)-min(arrivals),
 max_send_lateness_s=max(v['lateness_s']for v in records),independent_release_not_blocked_by_previous_ACK=True,
 scope='Actual source consumers and real sockets; no Engine or performance qualification')
p=Path('Kimi-prefill512-CPU-contract-r8563.json');p.write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n');print(json.dumps(out))
