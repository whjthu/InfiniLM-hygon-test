"""Incremental byte/SSE consumer. Feed with actual socket receive timestamps."""
import codecs,json,math
from token_clock import PREFIX
class ClientClock:
 def __init__(self,start_ns):
  self.start=start_ns;self.decode=codecs.getincrementaldecoder('utf8')();self.text='';self.tokens=[];self.text_progress=[];self.frames=[];self.terminal=None;self.http_status=None;self.first_semantic_ns=None
 def begin_response(self,status):
  self.http_status=int(status)
 def feed(self,raw,received_ns,final=False):
  if self.http_status is not None and self.http_status!=200:return
  if received_ns<self.start:raise ValueError('Receive clock precedes request')
  self.text+=self.decode.decode(raw,final=final);self.text=self.text.replace('\r\n','\n')
  while '\n\n'in self.text:
   block,self.text=self.text.split('\n\n',1)
   for line in block.split('\n'):
    if line.startswith(PREFIX):
     q=json.loads(line[len(PREFIX):])
     if q.get('terminal'):self.terminal=q
     elif q.get('text_progress'):self.text_progress.append(q)
     else:
      if type(q['ordinal'])is not int or q['ordinal']!=len(self.tokens):raise ValueError('Missing/duplicate token notice')
      if self.tokens and received_ns<self.tokens[-1]['received_ns']:raise ValueError('Nonmonotonic receive clock')
      self.tokens.append(dict(q,received_ns=received_ns))
   data='\n'.join(x[5:].lstrip()for x in block.split('\n')if x.startswith('data:'))
   if data:
    self.frames.append(dict(data=data,received_ns=received_ns))
    if data!='[DONE]':
     q=json.loads(data)
     semantic=any(any(c.get('delta',{}).get(k)for k in ['content','reasoning_content','tool_calls'])for c in q.get('choices',[]))
     if semantic and self.first_semantic_ns is None:self.first_semantic_ns=received_ns
  if final and self.text.strip():raise ValueError('Truncated SSE frame')
 def summary(self):
  ticks=[q['received_ns']for q in self.tokens if q['visible']and not q['eos']];itls=[(b-a)/1e9 for a,b in zip(ticks,ticks[1:])]
  pct=lambda p:sorted(itls)[max(0,math.ceil(len(itls)*p)-1)]if itls else None
  complete=self.terminal is not None and self.terminal['emitted']==len(self.tokens)and not self.terminal.get('observer_overflow',False)
  return dict(HTTP_status=self.http_status,rejected=self.http_status is not None and self.http_status!=200,finish_reason=self.terminal.get('finish_reason')if self.terminal else None,capped=self.terminal.get('finish_reason')=='length'if self.terminal else None,token_notice_count=len(self.tokens),special_tokens=sum(t['special']for t in self.tokens),EOS_tokens=sum(t['eos']for t in self.tokens),empty_text_progress=sum(t['empty_delta']for t in self.text_progress),unstable_UTF8_progress=sum(t['unstable_UTF8_tail']for t in self.text_progress),visible_nonEOS=len(ticks),token_ID_receipt_TTFT_s=(ticks[0]-self.start)/1e9 if ticks else None,semantic_TTFT_s=(self.first_semantic_ns-self.start)/1e9 if self.first_semantic_ns else None,decode_tok_s=(len(ticks)-1)*1e9/(ticks[-1]-ticks[0])if len(ticks)>1 and ticks[-1]>ticks[0]else None,ITL_p50_s=pct(.5),ITL_p95_s=pct(.95),ITL_p99_s=pct(.99),clock_coverage_complete=complete,complete=complete and bool(self.terminal.get('owned_finished'))if self.terminal else False,owned_finished=self.terminal.get('owned_finished')if self.terminal else None,chunk_is_token=False,zero_interval_is_coalesced=True,quality='NOT_SCORED',missing_or_rejected=not complete or not bool(self.terminal.get('owned_finished'))if self.terminal else True)


def collect_response(response,start_ns,save_read=None):
 """Consume an already-open HTTP response; never create a request here.

 read1 returns actual available socket bytes. Every marker in that read gets
 exactly one shared receive stamp; network/HOST clocks are never subtracted.
 """
 import time
 clock=ClientClock(start_ns);clock.begin_response(response.status)
 if response.status!=200:return clock
 while True:
  raw=response.read1(65536);stamp=time.monotonic_ns()
  if save_read is not None:save_read(raw,stamp)
  clock.feed(raw,stamp,final=not raw)
  if not raw:break
 return clock
