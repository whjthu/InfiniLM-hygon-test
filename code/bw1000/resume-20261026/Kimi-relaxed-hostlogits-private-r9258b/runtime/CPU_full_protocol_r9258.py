import pathlib,tempfile,threading,time,socket,os,json,sys,types,ctypes as C,traceback
import numpy as np
PP=2;NODES=2*PP;EMITTER=NODES-2
os.environ["KIMI_TRUE_DRAFT_BATCH_R9142"]="1"
os.environ['INFINILM_KIMI_BATCH_I192_R9125']='1'
from batch_frames_r9142 import receive_bodies
from batch_wire_r9128 import listen,recv,send
from draft_context_r9128 import ContextStore
import target_protocol as protocol
from target_protocol import run
from host_observer_r9183 import HostObserver,TimedControl,Reporter
class CPUReporter(Reporter):
 def full(self,*a,**kw):pass
OriginalRequest=protocol.Request
protocol.Request=lambda logical,prompt,cap,epoch,quality=None:OriginalRequest(logical,prompt,min(cap,8 if logical<=16 else 16 if logical<=24 else 24 if logical<=28 else 32 if logical<=30 else 40 if logical==31 else 48),epoch,quality)
class Hub:
 def __init__(self):self.cv=threading.Condition();self.data={}
 def collect(self,key,node,value):
  with self.cv:
   d=self.data.setdefault(key,{});assert node not in d;d[node]=value;self.cv.notify_all();assert self.cv.wait_for(lambda:len(d)==NODES,120),(key,list(d));return[d[i]for i in range(NODES)]
class Fn:
 def __init__(self,fun):self.fun=fun
 def __call__(self,*a):return self.fun(*a)
def v(p,typ,n):return np.ctypeslib.as_array(C.cast(p,C.POINTER(typ)),shape=(n,))
class Fake:
 def __init__(self,node):
  self.node=node;self.owner_armed=False;self.owner_clears=0;self.records=[];self.arm_on=False;self.fast=[0,0];self.replays=0;self.partial_nextchecks=0;self.prefl=0
  names={'capabilities':lambda ctx:json.dumps({'logical_capacity':16,'implemented_max_requests':16,'implemented_max_flatten_tokens':64,'pool_rows':33,'native_batch':True,'flags':3}).encode(),'reset':self.reset,'mode':self.mode,'rows':self.rows,'prefill':self.prefill,'forward':self.forward,'commit':self.commit,'replay':self.replay,'stats':self.stats,'taps_begin':self.arm,'taps_count':self.count,'taps_info':self.info,'taps_copy':self.copy}
  for name,fun in names.items():setattr(self,'modern_kimi_batch_'+name+'_r9122',Fn(fun))
  self.hist=[0]*65;self.modern_kimi_I192_T_counts_r9129=Fn(self.Thist)
  self.modern_kimi_batch_taps_capabilities_r9122=Fn(lambda:0x01200500);self.modern_kimi_I192_counts_r8669y=Fn(self.I192);self.reset(0,0)
 def reset(self,ctx,state):
  self.state_on=bool(state);self.states={i:{'a':i,'s':i+16,'p':0,'phase':0}for i in range(1,17)};self.ledger=[0]*33;self.prefix={i:[]for i in range(1,17)};self.order=[];self.counts=[0]*8;return 0
 def mode(self,ctx,state):self.state_on=bool(state);self.owner_armed=False;self.owner_clears+=1;return 0
 def rows(self,ctx,n,ids,out):
  z=v(out,C.c_int32,n*4).reshape(n,4)
  for j,key in enumerate(v(ids,C.c_int32,n)):
   a=self.states[int(key)];z[j]=[a['a'],a['s'],a['p'],a['phase']]
  return 0
 def arm(self,ctx,epoch,on,n,ids,past,width):self.epoch=int(epoch);self.arm_on=bool(on);self.meta=([int(x)for x in v(ids,C.c_int32,n)],[int(x)for x in v(past,C.c_int32,n)],[int(x)for x in v(width,C.c_int32,n)])if on else([],[],[]);self.records=[];return 0
 def count(self,ctx,p):C.cast(p,C.POINTER(C.c_size_t))[0]=len(self.records);return 0
 def info(self,ctx,index,p,n):view=self.records[index];v(p,C.c_int64,n)[:]=view[0];return 0
 def copy(self,ctx,index,p,n):raw=self.records[index][1];assert len(raw)==n;C.memmove(p,raw,n);return 0
 def output(self,logical,past,width,flat,logits,out,flags,prefill=False):
  T=sum(width);n=len(logical)if prefill else T;v(logits,C.c_uint16,n*128)[:]=0;selected=[];offset=0
  for N in width:
   row=flat[offset:offset+N];selected.extend([min(int(row[-1])+1,127)]if prefill else[min(int(x)+1,127)for x in row]);offset+=N
  if not prefill:
   mat=v(logits,C.c_uint16,n*128).reshape(n,128);mat[:]=0
   for j,x in enumerate(selected):
    mat[j,x]=0x3f80
    if j+1<T and flat[j+1]!=x and flat[j+1]<120:mat[j,flat[j+1]]=0x3f70
  if self.node==EMITTER:v(out,C.c_int64,n)[:]=selected
  v(flags,C.c_int32,1)[0]=int(self.node>=EMITTER)+2*int(self.node==EMITTER)
  if 8<=T<=64 and os.environ.get('INFINILM_KIMI_I192_GROUPED_R8669Y')=='1'and os.environ.get('INFINILM_KIMI_MOE_EXECUTION_R4665')=='1':self.hist[T]+=1
  if T==16 and os.environ.get('INFINILM_KIMI_I192_GROUPED_R8669Y')=='1'and os.environ.get('INFINILM_KIMI_MOE_EXECUTION_R4665')=='1':self.fast[0]+=1;self.fast[1]+=16
  if self.arm_on:
   assert self.meta==(logical,past,width);packed=np.concatenate([np.full((N,7168),0x3f00+key,np.uint16)for key,N in zip(logical,width)])
   for layer in ({0:[19,37],2:[66,78,90]}if PP==2 else{0:[19],2:[37],4:[66],6:[78,90]}).get(self.node,[]):self.records.append(([self.epoch,layer,T,7168,len(logical)]+[x for triple in zip(logical,past,width)for x in triple],packed.tobytes()))
  return 0
 def prefill(self,ctx,T,n,ids,off,tokens,pos,logits,out,flags):
  assert not self.owner_armed,'ownercache survivesEngineReset; explicitClear required beforeprefill'
  logical=list(map(int,v(ids,C.c_int32,n)));offset=list(map(int,v(off,C.c_int32,n+1)));width=[b-a for a,b in zip(offset,offset[1:])];flat=list(map(int,v(tokens,C.c_int64,T)))
  for i,key in enumerate(logical):
   a=self.states[key];assert a['phase']==0;a['p']=width[i];a['phase']=1;self.ledger[a['a']]=width[i];self.prefix[a['a']]=flat[offset[i]:offset[i+1]]
  self.prefl+=1;return self.output(logical,[0]*n,width,flat,logits,out,flags,True)
 def forward(self,ctx,T,n,flags_,read,write,ids,past,total,off,tokens,pos,logits,out,flags):
  assert flags_==3 and T<=64;logical=list(map(int,v(ids,C.c_int32,n)));pa=list(map(int,v(past,C.c_int32,n)));offset=list(map(int,v(off,C.c_int32,n+1)));width=[b-a for a,b in zip(offset,offset[1:])];flat=list(map(int,v(tokens,C.c_int64,T)));self.order=logical
  for i,key in enumerate(logical):
   a=self.states[key];assert a['phase']==1 and a['p']==pa[i]and self.ledger[a['a']]==pa[i]and int(v(read,C.c_int32,n)[i])==a['a']and int(v(write,C.c_int32,n)[i])==a['s'];assert 1<=width[i]<=8 and pa[i]+width[i]<=1024
   if a.pop('after_partial',False):self.partial_nextchecks+=1
   a.update(phase=2,width=width[i],ids=flat[offset[i]:offset[i+1]]);self.prefix[a['s']]=self.prefix[a['a']]+a['ids'];self.ledger[a['s']]=pa[i]+width[i]
  if self.state_on:self.owner_armed=True;self.counts[0]+=1;self.counts[1]+=len(logical)*35;self.counts[7]+=len(logical)
  return self.output(logical,pa,width,flat,logits,out,flags)
 def commit(self,ctx,n,ids,accepted,continuation):
  logical=list(map(int,v(ids,C.c_int32,n)));assert logical==self.order
  for key,A,cont in zip(logical,v(accepted,C.c_size_t,n),v(continuation,C.c_int32,n)):
   a=self.states[key];assert a['phase']==2 and A<a['width'];a['retained']=int(A)+1
   if not cont:a['phase']=4
   elif a['retained']==a['width']:a['p']+=a['width'];a['a'],a['s']=a['s'],a['a'];a['phase']=1
   else:a['phase']=3
  return 0
 def replay(self,ctx,state,logits,out,flags):
  assert bool(state)==self.state_on;ids=[key for key in self.order if self.states[key]['phase']==3];assert ids;past=[];width=[];flat=[]
  for key in ids:
   a=self.states[key];past.append(a['p']);width.append(a['retained']);flat+=a['ids'][:a['retained']];assert self.ledger[a['s']]==a['p']+a['width'];self.prefix[a['s']]=self.prefix[a['a']]+a['ids'][:a['retained']];self.ledger[a['s']]=a['p']+a['retained'];a['p']+=a['retained'];a['a'],a['s']=a['s'],a['a'];a['phase']=1;a['after_partial']=True
  self.replays+=1;self.counts[2]+=len(ids)*35
  if state:v(flags,C.c_int32,1)[0]=0;return 0
  return self.output(ids,past,width,flat,logits,out,flags)
 def stats(self,ctx,p,n):assert n==64;v(p,C.c_uint64,n)[:]=self.counts*8;return 0
 def Thist(self,device,p,n):assert n==65;v(p,C.c_uint64,n)[:]=self.hist;return 0
 def I192(self,device,p,n):v(p,C.c_uint64,n)[:]=self.fast;return 0
class Tok:
 all_special_ids=[127];added_tokens_decoder={126:types.SimpleNamespace(special=True)};special_tokens_map={'eos_token':'eos'}
 def __len__(self):return 128
 def apply_chat_template(self,messages,**kw):return[106,107,108,109,110]if'robe'in messages[0]['content']else[1,2,3]
 def decode(self,ids,**kw):return'#### '+('3'if ids[0]>100 else'18')
sys.modules['transformers']=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tok()))
def port():
 with socket.socket()as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
p1,p2=port(),port();os.environ.update(KIMI_ENGINE_HEAD='127.0.0.1',KIMI_FEATURE_PORT_R9046=str(p1),KIMI_DRAFT_HOST_R9046='127.0.0.1',KIMI_DRAFT_PORT_R9046=str(p2),KIMI_DRAFT_NONCE_R9046='batch-cpu-r9128');server=listen('127.0.0.1',p2);errors=[];store=ContextStore();draft_info={'queries':0,'retired':0,'max_live_contexts':0,'closed':False,'actual_shared_store':True};perkey={}
def drafter():
 try:
  conn,_=server.accept();h,body=recv(conn);assert h['nonce']=='batch-cpu-r9128';send(conn,{'ready':True,'request_keys':True,'block_size':8,'true_torch_batch':True,'chunked_features':True,'supports_projected_cache':True})
  while True:
   m,body=recv(conn);cmd=m['cmd'];reply={'seq':m['seq']}
   if cmd=='close':assert not store.values;reply['closed']=True;send(conn,reply);draft_info['closed']=True;break
   if cmd=='begin_cohort':store.begin(m['epoch']);reply.update(begun=True,epoch=m['epoch'],projected_cache=m.get('projected_cache',False));send(conn,reply);continue
   if cmd=='retire':
    old={key:id(value[1])for key,value in store.values.items()if key!=(m['epoch'],m['request'])};removed=store.retire(m['epoch'],m['request']);assert old=={key:id(value[1])for key,value in store.values.items()};draft_info['retired']+=1;reply.update(retired=True,removed=removed,epoch=m['epoch'],request=m['request']);send(conn,reply);continue
   assert cmd=='query_batch' and not body;items=m['requests'];store.validate_many(m['epoch'],items);bodies=receive_bodies(conn,m);parts=[];rows=[]
   for item,body in zip(items,bodies):
    marker=np.frombuffer(body,'<u2').reshape(item['rows'],35840);assert np.all(marker[:,::7168]==0x3f00+item['request']);parts.append(body)
   joined=store.append_many(m['epoch'],items,parts,lambda a,b:a+b)
   for item,context in zip(items,joined):
    assert len(context)==(item['past']+item['rows'])*35840*2;key=(m['epoch'],item['request']);perkey[key]=perkey.get(key,0)+1;props=[min(item['anchor']+j+1,127)for j in range(item['K'])]
    if item['request']%2 and perkey[key]%3==1 and props[0]<120:props[0]+=3
    rows.append({'request':item['request'],'context_tokens':item['past']+item['rows'],'proposals':props})
   draft_info['queries']+=1;draft_info.setdefault('actual_batch_counts',[]).append(len(items));draft_info['max_live_contexts']=max(draft_info['max_live_contexts'],len(store.values));reply.update(epoch=m['epoch'],requests=rows,actual_model_batch_C=len(items),draft_batch_query_s=.001,append_batch_H2D_s=.0001);send(conn,reply)
  conn.close();server.close()
 except BaseException as e:errors.append(('draft',repr(e),traceback.format_exc()))
t=threading.Thread(target=drafter,daemon=True);t.start();hub=Hub();reports={};live_reports={};models={};begin=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='kimi-batch-r9128-')as temp:
 root=pathlib.Path(temp);(root/'incoming').mkdir();(root/'results/cpuqueue').mkdir(parents=True);os.environ['KIMI_BATCH_CASE_QUEUE_R9128']='cpuqueue'
 for p in pathlib.Path(__file__).with_name('queue').glob('q*.json'):(root/'results/cpuqueue'/p.name).write_bytes(p.read_bytes())
 # No preseededstop: exercise real bounded resident queue, CPU operator adds a latecase after idleheartbeat.
 latecase=json.loads(next(pathlib.Path(__file__).with_name('queue').glob('q*.json')).read_bytes());latecase.update(id='CPU-late-fixed-C1-after-idle',C=1,W=8,strategy='state',adaptive_width=False,cap=8);latecase['requests']=latecase['requests'][:1]
 def append_after_idle():
  until=time.monotonic()+30
  while 'resident_idle_heartbeat'not in live_reports.get(0,{}):
   assert time.monotonic()<until;time.sleep(.01)
  (root/'results/cpuqueue/q0990-late.json').write_text(json.dumps(latecase));(root/'results/cpuqueue/q9999-stop.json').write_text(json.dumps({'action':'stop'}));draft_info['resident_append_after_idle_then_stop']=True
 appender=threading.Thread(target=append_after_idle,daemon=True);appender.start()
 (root/'results/cpuqueue/q0098-invalid.json').write_text(json.dumps({'schema':1,'action':'case','id':'reject-C32W8-T256','C':32,'W':8,'strategy':'state','I192':True,'cap':32}))
 (root/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').write_text(json.dumps({'eos_token_ids':[127]}))
 def worker(node):
  try:
   lib=Fake(node);models[node]=lib;report={'node':node,'run':'CPUonly'};live_reports[node]=report;observer=HostObserver();control=TimedControl(types.SimpleNamespace(collect=lambda key,value:hub.collect(key,node,value)),observer);reporter=CPUReporter(root/'results'/('node%d.json'%node),report,observer);run({'host_observer':observer,'progress_save':reporter.progress,'set_logging_diag':reporter.set_logging_diag,'lib':lib,'ctx':node+1,'r':root,'control':control,'report':report,'save':reporter.full,'node':node,'tail':node>=EMITTER,'emitter':EMITTER,'pp':PP,'tp_nodes':2,'vocab':128,'model':root,'memory_guard':lambda label,reset=False:{'mock_CPU_memory_guard':True,'label':label},'experiment_started_r9046':time.monotonic()});reports[node]=report
  except BaseException as e:errors.append((node,repr(e),traceback.format_exc()))
 threads=[threading.Thread(target=worker,args=(node,),daemon=True)for node in range(NODES)]
 for th in threads:th.start()
 for th in threads:th.join(160)
 t.join(2)
 assert not errors,errors
 assert draft_info['resident_append_after_idle_then_stop']
 assert all(len(z.get('command_rejections',[]))==1 for z in reports.values())
 assert len(reports)==NODES and draft_info['closed']and all(x['serving_protocol_complete']for x in reports.values())
 assert all(m.partial_nextchecks>0 for m in models.values())
 assert any(c['relaxed_gap_epsilon']==.10 for c in reports[EMITTER]['batch_cases'])
 assert any(z.get('emitted_relaxed',0)>0 for c in reports[EMITTER]['batch_cases']for z in c['cycles'])
 for c in reports[EMITTER]['batch_cases']:
  expected=[(u['generated_ids'],[(z['accepted_proposals'],z['actual_after_row'])for z in u['cycles']])for u in c['request_metrics']]
  for rr in reports.values():
   other=next(x for x in rr['batch_cases']if x['label']==c['label'])
   assert [(u['generated_ids'],[(z['accepted_proposals'],z['actual_after_row'])for z in u['cycles']])for u in other['request_metrics']]==expected
 # Independently interleave32 actual ContextStore keys, retire one, wrong epoch/past reject.
 isolated=ContextStore();isolated.begin(1)
 for key in range(1,33):isolated.append(1,key,0,1,bytes([key]),lambda a,b:a+b)
 isolated.retire(1,16);assert len(isolated.values)==31 and isolated.values[(1,17)][1]==bytes([17])
 rejected=0
 for args in [(2,1,1,1,b'a'),(1,1,0,1,b'a')]:
  try:isolated.append(*args,lambda a,b:a+b)
  except AssertionError:rejected+=1
 assert rejected==2
 q={'status':'FOUR_RANK_REAL_SOCKET_CPU_NATIVE_MOCK_PASS','GPU':False,'math_mocked':True,'models':NODES,'cohorts':len(reports[EMITTER]['batch_cases']),'draft':draft_info,'partial_then_next_engine_ledger_checks':[m.partial_nextchecks for m in models.values()],'four_rank_decisions_and_commits_identical':True,'relaxed_emitted':sum(z.get('emitted_relaxed',0)for c in reports[EMITTER]['batch_cases']for z in c['cycles']),'elapsed_s':time.perf_counter()-begin};pathlib.Path(__file__).with_name('CPU-full-protocol-r9258.json').write_bytes(json.dumps(q,indent=2).encode());print(json.dumps(q))
