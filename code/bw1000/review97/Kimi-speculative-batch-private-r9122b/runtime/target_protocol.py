import os,time,json,pathlib,re
from batch_wire_r9128 import listen,connect,send,recv
from batch_native_r9128 import Native
from batch_cohort_r9128 import Request,cohort

def startup(lib):
 from target_binding import startup as check
 return check(lib)
def run(g):
 node=g['node'];control=g['control'];report=g['report'];save=g['save'];r=g['r'];native=Native(g);report.update(scope='TRUE_NATIVE_PACKED_BATCH_closedcohort_flags3_Math0; independent perrequest drafter, no reference-token equality gate',batch_cases=[],batch_phase_calls=native.calls);sock=peer=server=None;seq=0;nonce=os.environ['KIMI_DRAFT_NONCE_R9046'];epoch=0;deadline=g['experiment_started_r9046']+5000
 def rpc(cmd,body=b'',**fields):
  nonlocal seq
  t=time.perf_counter();send(sock,{'cmd':cmd,'seq':seq,**fields},body);h,b=recv(sock);assert h['seq']==seq and not b;seq+=1;return h,time.perf_counter()-t
 try:
  if node==0:
   server=listen(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));peer,_=server.accept();h,b=recv(peer);assert h=={'cmd':'batch-peer','nonce':nonce}and not b;send(peer,{'ready':True});sock=connect(os.environ['KIMI_DRAFT_HOST_R9046'],int(os.environ['KIMI_DRAFT_PORT_R9046']));send(sock,{'cmd':'hello','nonce':nonce});h,b=recv(sock);assert h['ready']and h['request_keys']and h['block_size']==8 and not b;report['draft_handshake']=h
  elif node==2:
   peer=connect(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));send(peer,{'cmd':'batch-peer','nonce':nonce});h,b=recv(peer);assert h['ready']and not b
  native.peer=peer;control.collect('batch-transports-ready',{});report['actual_batch_capabilities']=native.cap;save()
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(str(g['model']),trust_remote_code=True);samples=json.loads(pathlib.Path(__file__).with_name('quality_samples.json').read_text())['samples'];fixed=json.loads((r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_text());eos=set(fixed['eos_token_ids']);prompts=[tok.apply_chat_template([{'role':'user','content':s['question']+'\nSolve this problem. End with #### followed by the numeric answer.'}],tokenize=True,add_generation_prompt=True)for s in samples]
  # C16/32 remain unsupported pending realpool33/65 and T64 qualification.
  report['concurrency_scope']={'requested':[1,2,4,8,16,32],'implemented':{'maxC':native.maxC,'maxT':native.maxT},'not_qualified':[16,32],'continuous_arrivals':False,'allcards':33};save()
  configs=[(1,1,'ar',False,16),(2,1,'ar',False,256),(2,8,'oldspec',False,256),(2,8,'state',False,256),(2,8,'state',True,256),(2,8,'state',True,256),(2,8,'state',False,256),(4,1,'ar',False,256),(4,4,'oldspec',False,256),(4,4,'state',False,256),(4,4,'state',True,256)]
  seen=set()
  def execute(command):
   nonlocal epoch
   label=command.get('id','invalid');assert re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',label)and label not in seen;seen.add(label)
   try:
    assert command['schema']==1 and command['action']=='case';C=int(command['C']);W=int(command['W']);mode=command['strategy'];fast=command['I192'];cap=int(command.get('cap',256));assert type(fast)is bool and mode in ['ar','oldspec','state'];assert 1<=C<=native.maxC and C*W<=native.maxT and ((mode=='ar'and W==1)or(mode!='ar'and 2<=W<=8))and 2<=cap<=512
    data=command.get('requests');assert data is None or isinstance(data,list)and len(data)==C
    prepared=[]
    for i in range(C):
     if data is None:prompt=prompts[i%len(prompts)];quality={'sample_index':i%len(samples),'answer':samples[i%len(samples)]['answer']}
     else:
      q=data[i];assert len(q['question'])<=8192;prompt=tok.apply_chat_template([{'role':'user','content':q['question']+'\nSolve this problem. End with #### followed by the numeric answer.'}],tokenize=True,add_generation_prompt=True);quality={'sample_index':int(q['sample_index']),'answer':str(q['answer'])}
     assert 1<=len(prompt)<=512 and len(prompt)+cap<=1024;prepared.append((prompt,quality))
   except (AssertionError,KeyError,ValueError,TypeError)as error:
    report.setdefault('command_rejections',[]).append({'id':label,'reason':'unsupported boundedcapacity/schema; no questiontruncation','error':repr(error)});save();control.collect('batch-command-reject:'+label,{});return
   allow=control.collect('batch-budget:'+label,{'allow':time.monotonic()+180<deadline if node==0 else None})[0]['allow']
   if not allow:report.setdefault('pending_budget',[]).append(label);save();return
   epoch+=1;reqs=[Request(i+1,prompt,cap,epoch,quality)for i,(prompt,quality)in enumerate(prepared)]
   def progress(value):report['phase']='batch:'+label;report['active_batch_progress']=value;save()
   report['phase']='batch:'+label;save();out=cohort(native,control,rpc,node,g['emitter'],reqs,label,W,mode,fast,epoch,eos,deadline,progress);out['command']=command
   for item,req in zip(out['request_metrics'],reqs):
    text=tok.decode(item['generated_ids'],skip_special_tokens=True);answers=re.findall(r'####\s*(-?[0-9][0-9,.]*)',text);answer=answers[-1].replace(',','').rstrip('.')if answers else None;item['quality']={'task':'GSM8K-smoke','sample_index':req.quality['sample_index'],'expected':req.quality['answer'],'extracted':answer,'correct':answer==req.quality['answer'],'response':text,'pilot_only':True}
   report['batch_cases'].append(out);save();control.collect('batch-case-done:'+label,{})
  for index,(C,W,mode,fast,cap)in enumerate(configs):execute({'schema':1,'action':'case','id':'C%dW%d-%s-I192%d-r%d'%(C,W,mode,int(fast),index),'C':C,'W':W,'strategy':mode,'I192':bool(fast),'cap':cap})
  name=os.environ.get('KIMI_BATCH_CASE_QUEUE_R9128');files_seen=set();poll=0
  if name:
   assert re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',name);queue=r/'results'/name;report['resident_queue']=str(queue);report['phase']='batch_resident_ready';save()
   while True:
    command=None
    if node==0:
     if time.monotonic()+180>=deadline:command={'action':'stop','reason':'reserved cleanup'}
     else:
      for path in sorted(queue.glob('q*.json')):
       if path.name in files_seen:continue
       try:raw=path.read_bytes();assert len(raw)<=131072;command=json.loads(raw)
       except (OSError,ValueError):continue
       files_seen.add(path.name);break
      if command is None:time.sleep(.5)
    command=control.collect('batch-queue-poll:'+str(poll),{'command':command})[0]['command'];poll+=1
    if command is None:continue
    if command.get('action')=='stop':report['resident_stop']=command;save();break
    execute(command)
  report['serving_protocol_complete']=True;report['phase']='batch_protocol_complete';save()
 finally:
  native.arm([],[],[],False)
  if node==0 and sock is not None:
   try:rpc('close')
   finally:sock.close()
  if peer is not None:peer.close()
  if server is not None:server.close()
