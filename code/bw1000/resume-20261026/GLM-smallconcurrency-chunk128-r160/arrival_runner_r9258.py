"""Finite1..32 maxlive1/8/32 M16x2 native arrival emulation. No HTTP, row recycling, native build or GPU initialization."""
import time, math
from decode_wave_r9225 import decode_wave_r9225
OPS={'WAIT':0,'PREFIX':1,'ANCHOR':2,'DECODE':3,'DONE':4}
MAGIC=4179133
WIRE=6+4*32

def validate_requests(requests, arrivals):
 if not 1<=len(requests)<=32 or len(arrivals)!=len(requests) or len({r['id']for r in requests})!=len(requests):raise ValueError('bounded distinct requests')
 if any(type(a)not in(int,float)or not math.isfinite(a)or a<0 for a in arrivals):raise ValueError('arrival')
 for r in requests:
  if not r['input_ids']or any(type(t)is not int or not 0<=t<154880 for t in r['input_ids'])or type(r['max_new_tokens'])is not int or r['max_new_tokens']<1 or len(r['input_ids'])+r['max_new_tokens']>640:raise ValueError('uncropped input/cap')

class Leader:
 def __init__(self,requests,arrivals,chunked,eos,max_live=32):
  validate_requests(requests,arrivals)
  if type(chunked)is not bool:raise TypeError('strictbool policy')
  if type(max_live)is not int or max_live not in (1,8,32):raise ValueError('supported admission bound')
  self.max_live=max_live;self.requests=requests;self.arrivals=arrivals;self.chunked=chunked;self.eos=set(eos);self.seq=0;self.group=[];self.prefer_decode=False;self.peak_live=0
  self.rows=[{'id':r['id'],'row':i,'arrival':float(arrivals[i]),'admitted':None,'service_start':None,'status':'queued','prefix_offset':0,'past':0,'generated_ids':[],'events':[]}for i,r in enumerate(requests)]
 def live(self):return sum(r['status']in('prefill','ready')for r in self.rows)
 def next(self,now,budget_seconds):
  for i in sorted(range(len(self.requests)),key=lambda i:(self.arrivals[i],i)):
   r=self.rows[i]
   if r['status']=='queued'and r['arrival']<=now and self.live()<self.max_live:r.update(status='prefill',admitted=now)
  self.peak_live=max(self.peak_live,self.live());assert self.live()<=self.max_live
  self.seq+=1;cmd={'seq':self.seq,'op':'DONE','time_ms':int(now*1000),'wait_ms':0,'items':[],'reason':'complete'}
  if now>=budget_seconds:cmd.update(reason='budget',wait_ms=1);return cmd
  if all(r['status']in('EOS','cap')for r in self.rows):return cmd
  ready=[i for i,r in enumerate(self.rows)if r['status']=='ready']
  if self.chunked and self.prefer_decode and ready:return self.decode(cmd,ready)
  if not self.group:self.group=[i for i,r in enumerate(self.rows)if r['status']=='prefill'][:8]
  if self.group:
   prefix=[i for i in self.group if self.rows[i]['prefix_offset']<len(self.requests[i]['input_ids'])-1]
   if prefix:
    cmd['op']='PREFIX';left=4096
    for i in prefix:
     r=self.rows[i];remaining=len(self.requests[i]['input_ids'])-1-r['prefix_offset'];take=min(remaining,128 if self.chunked else remaining,left)
     if take:cmd['items'].append((i,r['past'],take,0));left-=take
    return cmd
   cmd['op']='ANCHOR';cmd['items']=[(i,self.rows[i]['past'],1,0)for i in self.group];return cmd
  if ready:return self.decode(cmd,ready)
  future=[r['arrival']for r in self.rows if r['status']=='queued'and r['arrival']>now]
  if not future:raise RuntimeError('no progress at bounded resident32')
  cmd.update(op='WAIT',wait_ms=max(1,math.ceil((min(future)-now)*1000)));return cmd
 def decode(self,cmd,ready):
  cmd['op']='DECODE';cmd['items']=[(i,self.rows[i]['past'],1,self.rows[i]['generated_ids'][-1])for i in ready[:32]];return cmd
 def started(self,cmd,now):
  for i,_,_,_ in cmd['items']:
   r=self.rows[i];assert r['admitted']is not None and now>=r['arrival']
   if r['service_start']is None:r['service_start']=now
 def consumed(self,cmd,now):
  for i,past,width,_ in cmd['items']:
   r=self.rows[i];assert r['past']==past and r['admitted']is not None and now>=r['arrival']
   assert r['service_start']is not None
   if cmd['op']in('PREFIX','ANCHOR'):assert r['status']=='prefill'
   else:assert r['status']=='ready'
   r['past']+=width
   if cmd['op']=='PREFIX':r['prefix_offset']+=width
  self.prefer_decode=cmd['op']in('PREFIX','ANCHOR')
 def tokens(self,cmd,values,stamp):
  assert cmd['op']in('ANCHOR','DECODE')and len(values)==len(cmd['items'])
  for (i,_,_,_),token in zip(cmd['items'],values):
   r=self.rows[i];req=self.requests[i];assert type(token)is int and 0<=token<154880
   if cmd['op']=='ANCHOR':assert not r['generated_ids']and r['past']==len(req['input_ids'])
   else:assert r['generated_ids']and r['status']=='ready'
   r['generated_ids'].append(token)
   if token not in self.eos:r['events'].append({'time':stamp,'visible':True,'kind':'committed','EOS':False,'pad':False,'draft':False,'replay':False,'token_id':token,'phase':'prefill'if cmd['op']=='ANCHOR'else'decode'})
   if token in self.eos:r.update(status='EOS',stop_reason='EOS',end_time=stamp)
   elif len(r['generated_ids'])>=req['max_new_tokens']:r.update(status='cap',stop_reason='max_new_tokens',end_time=stamp)
   else:r['status']='ready'
   assert r['past']==len(req['input_ids'])+len(r['generated_ids'])-1
  if cmd['op']=='ANCHOR':self.group=[]
  if cmd['op']=='DECODE':self.prefer_decode=False
 def report(self,elapsed,reason):
  result=[]
  for r in self.rows:
   q=dict(r);q['frontend_state']=q['status'];q['status']='completed'if q['status']in('EOS','cap')else'not_arrived'if q['arrival']>elapsed else'unfinished';q['dispatch_queue_wait_seconds']=None if q['service_start']is None else q['service_start']-q['arrival'];q['queue_wait_seconds']=None if q['admitted']is None else q['admitted']-q['arrival'];q['TTFT_seconds']=q['events'][0]['time']-q['arrival']if q['events']else None;result.append(q)
  return {'requests':result,'elapsed_seconds':elapsed,'end_reason':reason,'all_requests_completed':all(r['status']=='completed'for r in result),'scheduled_requests':len(self.requests),'arrivals':sum(r['arrival']<=elapsed for r in result),'not_arrived':sum(r['status']=='not_arrived'for r in result),'unfinished':sum(r['status']=='unfinished'for r in result),'global_reset_allowed':all(r['status']=='completed'for r in result),'max_admitted_live':self.max_live,'peak_admitted_live':self.peak_live,'native_row_lifetime_distinct':len(self.requests),'row_recycling':False,'chunkcap':128 if self.chunked else None,'native_emulation_not_HTTP':True,'quality_evaluated':False}

def encode_command(cmd):
 v=[0]*WIRE
 if cmd is None:return v
 if cmd['op']not in OPS or len(cmd['items'])>32:raise ValueError('command')
 v[:6]=[MAGIC,cmd['seq'],OPS[cmd['op']],cmd['time_ms'],cmd['wait_ms'],len(cmd['items'])]
 for j,item in enumerate(cmd['items']):v[6+j*4:10+j*4]=item
 if any(type(x)is not int or not 0<=x<2**24 for x in v):raise ValueError('exactFP32 integer range')
 return v

def decode_command(v,expected_seq):
 if len(v)!=WIRE or any(not math.isfinite(float(x))or int(x)!=x for x in v):raise ValueError('wire dtype/shape')
 v=list(map(int,v))
 if any(not 0<=x<2**24 for x in v):raise ValueError('wire exactinteger bounds')
 magic,seq,op,ms,wait,n=v[:6]
 if magic!=MAGIC or seq!=expected_seq or op not in OPS.values()or not 0<=n<=32 or any(v[6+4*n:])or(op in(0,4)and n)or(op==0 and wait<1)or(op==4 and wait not in(0,1))or(op not in(0,4)and wait):raise ValueError('wire owner/sequence')
 return {'seq':seq,'op':next(k for k,z in OPS.items()if z==op),'time_ms':ms,'wait_ms':wait,'reason':'budget'if op==OPS['DONE']and wait==1 else'complete','items':[tuple(v[6+4*j:10+4*j])for j in range(n)]}

def payload(cmd,requests):
 op=cmd['op'];items=cmd['items']
 if op not in('PREFIX','ANCHOR','DECODE'):raise ValueError('notforward')
 if not items or len(items)>(32 if op=='DECODE'else 8)or len({x[0]for x in items})!=len(items):raise ValueError('pureboundedbatch')
 rows=[];tokens=[];pasts=[]
 for i,past,n,token in items:
  if not 0<=i<32 or not 0<n or not 0<=past<640 or past+n>640:raise ValueError('row/past/count')
  ids=requests[i]['input_ids']
  if op=='PREFIX':
   if past+n>len(ids)-1:raise ValueError('prefixonlynotready')
   vals=ids[past:past+n]
  elif op=='ANCHOR':
   if n!=1 or past!=len(ids)-1:raise ValueError('completepromptanchor')
   vals=[ids[-1]]
  else:
   if n!=1 or past<len(ids)or not 0<=token<154880:raise ValueError('readydecodeonly')
   vals=[token]
  rows.append(i);tokens.append(vals);pasts.append(past)
 if sum(map(len,tokens))>4096:raise ValueError('packedbudget')
 return rows,tokens,pasts

def run_episode(requests,arrivals,chunked,eos,is_emitter,command_exchange,token_exchange,forward,clock=time.perf_counter,wait=time.sleep,budget_seconds=5000,deferred_feedback=False,max_live=32):
 validate_requests(requests,arrivals)
 if type(chunked)is not bool:raise TypeError('strictbool policy')
 state=Leader(requests,arrivals,chunked,eos,max_live)if is_emitter else None;start=clock();seq=0;trace=[];last_ms=-1
 while True:
  now=clock()-start;outgoing=state.next(now,budget_seconds)if state else None;seq+=1;cmd=command_exchange(outgoing,seq);assert cmd['time_ms']>=last_ms;last_ms=cmd['time_ms'];op=cmd['op'];trace.append({'seq':seq,'op':op,'items':cmd['items'],'emitter_time_ms':cmd['time_ms']})
  if op=='DONE':
   return (state.report(clock()-start,outgoing['reason'])if state else None),trace
  if op=='WAIT':wait(cmd['wait_ms']/1000);continue
  rows,values,pasts=payload(cmd,requests)
  if state:state.started(cmd,clock()-start)
  if op=='DECODE':
   # Same current command/state and rows; at most two independent M16 micros.
   chunks=[list(range(i,min(i+16,len(rows))))for i in range(0,len(rows),16)]
   def one(indices):
    rr=[rows[j]for j in indices];tt=[values[j]for j in indices];pp=[pasts[j]for j in indices]
    return forward(rr,tt,pp,'arrival.DECODE',False)
   for indices,logits,wall,tick,ended in decode_wave_r9225(chunks,one,deferred_feedback):
    micro=dict(cmd,items=[cmd['items'][j]for j in indices])
    if state:state.consumed(micro,clock()-start)
    selected=token_exchange(micro,logits if is_emitter else None)
    if state:state.tokens(micro,selected,clock()-start)
    del logits
  else:
   result=forward(rows,values,pasts,'arrival.'+op,True);logits=result[0]
   if state:state.consumed(cmd,clock()-start)
   if op!='PREFIX':
    selected=token_exchange(cmd,logits if is_emitter else None)
    if state:state.tokens(cmd,selected,clock()-start)
   del logits,result

def torch_world_callbacks(torch,control,is_emitter,device='cuda'):
 """Only existing WORLD integer-FP32 SUM; no newcommunicator/socket/stream."""
 def command_exchange(cmd,seq):
  a=torch.tensor(encode_command(cmd),device=device,dtype=torch.float32);control(a);return decode_command(a.cpu().tolist(),seq)
 def token_exchange(cmd,logits):
  n=len(cmd['items']);a=torch.zeros(3+16,device=device,dtype=torch.float32)
  if is_emitter:a[:3]=torch.tensor([MAGIC,cmd['seq'],n],device=device,dtype=torch.float32);a[3:3+n].copy_(logits.argmax(-1).float())
  control(a);v=a.cpu().tolist()
  if v[:3]!=[MAGIC,cmd['seq'],n]or any(v[3+n:])or any(int(t)!=t or not 0<=int(t)<154880 for t in v[3:3+n]):raise ValueError('token wire')
  return list(map(int,v[3:3+n]))
 return command_exchange,token_exchange
