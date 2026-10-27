import os,time,json,pathlib,re
from batch_wire_r9128 import listen,connect,send,recv
from batch_native_r9128 import Native
from batch_cohort_r9128 import Request,cohort
from batch_frames_r9142 import send_query
from W2_policy_r9282 import W2Counter
from service_metrics_r9282 import service_metrics,screen_gate

def startup(lib):
 from target_binding import startup as check
 return check(lib)
def run(g):
 assert os.environ.get("KIMI_TRUE_DRAFT_BATCH_R9142")=="1", "r9142 defaultOFF"
 node=g['node'];control=g['control'];report=g['report'];save=g['save'];r=g['r'];native=Native(g);W2=W2Counter(g['lib']);service_starts={};service_parts={};report.update(scope='TRUE_NATIVE_PACKED_BATCH_closedcohort_flags3_Math0; independent perrequest drafter, no reference-token equality gate',batch_cases=[],batch_phase_calls=native.calls);sock=peer=server=None;seq=0;nonce=os.environ['KIMI_DRAFT_NONCE_R9046'];epoch=0;deadline=g['experiment_started_r9046']+int(os.environ.get('KIMI_RUNTIME_INNER_S_R9225','7000'))-150
 def rpc(cmd,body=b'',**fields):
  nonlocal seq
  t=time.perf_counter()
  if cmd=='query_batch':send_query(sock,{'cmd':cmd,'seq':seq,**fields},body)
  else:send(sock,{'cmd':cmd,'seq':seq,**fields},body)
  h,b=recv(sock);assert h['seq']==seq and not b;seq+=1;return h,time.perf_counter()-t
 try:
  if node==0:
   server=listen(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));server.listen(g.get('pp',2))
   for _ in range(int(g.get('pp',2))-1):
    accepted,_addr=server.accept();h,b=recv(accepted);assert h['cmd']=='batch-peer'and h['nonce']==nonce and h['node']%native.tp_nodes==0 and h['node']>0 and h['node']not in native.peers and not b;native.peers[h['node']]=accepted;send(accepted,{'ready':True})
   sock=connect(os.environ['KIMI_DRAFT_HOST_R9046'],int(os.environ['KIMI_DRAFT_PORT_R9046']));send(sock,{'cmd':'hello','nonce':nonce});h,b=recv(sock);assert h['ready']and h['request_keys']and h['block_size']==8 and h['true_torch_batch']and h['chunked_features']and h['supports_projected_cache']and not b;report['draft_handshake']=h
  elif node%native.tp_nodes==0:
   peer=connect(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));send(peer,{'cmd':'batch-peer','node':node,'nonce':nonce});h,b=recv(peer);assert h['ready']and not b
  native.peer=peer;control.collect('batch-transports-ready',{});report['actual_batch_capabilities']=native.cap;save()
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(str(g['model']),trust_remote_code=True);samples=json.loads(pathlib.Path(__file__).with_name('quality_samples.json').read_text())['samples'];fixed=json.loads((r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_text());eos=set(fixed['eos_token_ids']);prompts=[tok.apply_chat_template([{'role':'user','content':s['question']+'\nSolve this problem. End with #### followed by the numeric answer.'}],tokenize=True,add_generation_prompt=True)for s in samples]
  # CPU callable C16/32 candidate; runtime Model and capacity remain unqualified.
  report['concurrency_scope']={'requested':[1,2,4,8],'implemented':{'maxC':native.maxC,'maxT':native.maxT},'not_qualified':[8], 'unsupported_native':[16,32],'continuous_arrivals':False,'allcards':int(g.get('pp',2))*16+int(os.environ.get('KIMI_DRAFT_CARD_COUNT_R9142','1'))};save()
  configs=[]
  seen=set()
  def execute(command):
   nonlocal epoch
   label=command.get('id','invalid');service_id=command.get('service_id');service_part=command.get('service_part');
   if service_id is not None:
    assert re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',service_id)and service_part in [0,1]
    if service_part==0:assert service_id not in service_starts;service_starts[service_id]=time.perf_counter();service_parts[service_id]=[]
    else:assert service_id in service_starts and len(service_parts[service_id])in [0,1]
   assert re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',label)and label not in seen;seen.add(label)
   try:
    pair_enabled=command.get('W2pair',False);assert type(pair_enabled)is bool;assert command.get('epsilon',0)==0;assert not command.get('draft_projected_cache',False)
    assert command['schema']==1 and command['action']=='case';C=int(command['C']);W=int(command['W']);mode=command['strategy'];fast=command['I192'];adaptive_width=command.get('adaptive_width',False);assert type(adaptive_width)is bool;draft_projected_cache=command.get('draft_projected_cache',False);assert type(draft_projected_cache)is bool;logging_diag=command.get('logging_diag',False);assert type(logging_diag)is bool;case_budget_s=int(command.get('case_budget_s',600));assert 30<=case_budget_s<=1800;cap=int(command.get('cap',256));assert type(fast)is bool and mode in ['ar','oldspec','state'];assert not draft_projected_cache or mode=='state';assert not adaptive_width or mode!='ar';assert 1<=C<=native.maxC and C*W<=native.maxT and ((mode=='ar'and W==1)or(mode!='ar'and 2<=W<=8))and 2<=cap<=512
    data=command.get('requests');assert data is None or isinstance(data,list)and len(data)==C
    prepared=[]
    for i in range(C):
     if data is None:prompt=prompts[i%len(prompts)];quality={'sample_index':i%len(samples),'answer':samples[i%len(samples)]['answer']}
     else:
      q=data[i];assert len(q['question'])<=8192;from task_quality_r9258 import prepare
      text,quality=prepare(q);prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
     assert 1<=len(prompt)<=512 and len(prompt)+cap<=1024;prepared.append((prompt,quality))
   except (AssertionError,KeyError,ValueError,TypeError)as error:
    report.setdefault('command_rejections',[]).append({'id':label,'reason':'unsupported boundedcapacity/schema; no questiontruncation','error':repr(error)});save();control.collect('batch-command-reject:'+label,{});return
   allow=control.collect('batch-budget:'+label,{'allow':time.monotonic()+case_budget_s<deadline if node==0 else None})[0]['allow']
   if not allow:report.setdefault('pending_budget',[]).append(label);save();return
   W2.configure(pair_enabled);flags=control.collect('W2-config:'+label,{'enabled':pair_enabled});assert all(x['enabled']==pair_enabled for x in flags);W2before=W2.read()
   if g.get('set_logging_diag'):g['set_logging_diag'](logging_diag)
   epoch+=1;reqs=[Request(i+1,prompt,cap,epoch,quality)for i,(prompt,quality)in enumerate(prepared)]
   def progress(value):
    report['phase']='batch:'+label
    if g.get('progress_save'):g['progress_save'](value)
    else:report['active_batch_progress']=value;save()
   report['phase']='batch:'+label;save();memory_before=native.memory_guard('case-start:'+label,reset=True)if native.memory_guard else None;hist_before=native.Tstats();out=cohort(native,control,rpc,node,g['emitter'],reqs,label,W,mode,fast,epoch,eos,deadline,progress,draft_projected_cache,adaptive_width);out['command']=command;out['logging_mode']='legacy_full_progress'if logging_diag else'tiny_progress';hist_after=native.Tstats();out['I192_T65_before8x65']=hist_before;out['I192_T65_after8x65']=hist_after;out['I192_T65_delta8x65']=[[b-a for a,b in zip(x,y)]for x,y in zip(hist_before,hist_after)];out['memory_before_case']=memory_before;out['memory_minimum_case']=native.memory_guard('case-end:'+label)if native.memory_guard else None
   W2after=W2.read();out['W2pair_enabled']=pair_enabled;out['W2_counts_before8x2']=W2before;out['W2_counts_after8x2']=W2after;out['W2_counts_delta8x2']=W2.delta(W2before,W2after,pair_enabled)
   assert all(sum(pair)==counts[0]for pair,counts in zip(out['W2_counts_delta8x2'],out['I192_counts_delta8x2'])),'actual W2provider count must equal qualifiedI192 calls'
   for item,req in zip(out['request_metrics'],reqs):
    text=tok.decode(item['generated_ids'],skip_special_tokens=True);from task_quality_r9258 import score
    quality=dict(req.quality);quality.setdefault('task','gsm8k');quality.setdefault('domain','math');quality.setdefault('id',str(quality['sample_index']));quality.update(score(quality,text,'EOS'if item['EOS']else'max_new_tokens'if item['truncated']else'unfinished'));quality.update(expected=req.quality['answer'],response=text);item['quality']=quality
   if service_id is not None:
    out['service_group_id']=service_id;out['service_part']=service_part;out['service_shared_arrival_monotonic']=service_starts[service_id];service_parts[service_id].append(out)
    if service_part==1:report.setdefault('frontend_C32_services',[]).append(dict(service_metrics(service_parts[service_id],service_starts[service_id]),service_id=service_id,W2pair=pair_enabled))
   report['batch_cases'].append(out);gate=screen_gate(report['batch_cases']);
   if gate is not None:report['screen_gate']=gate
   save();control.collect('batch-case-done:'+label,{})
  for index,(C,W,mode,fast,cap)in enumerate(configs):execute({'schema':1,'action':'case','id':'C%dW%d-%s-I192%d-r%d'%(C,W,mode,int(fast),index),'C':C,'W':W,'strategy':mode,'I192':bool(fast),'cap':cap})
  name=os.environ.get('KIMI_BATCH_CASE_QUEUE_R9128');files_seen=set();poll=0;last_idle_heartbeat=0.
  if name:
   assert re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',name);queue=r/'results'/name;report['resident_queue']=str(queue);report['phase']='batch_resident_ready';save()
   while True:
    command=None
    if node==0:
     if report.get('screen_gate',{}).get('stop'):command={'action':'stop','reason':'frozen matched first8 screen excess errors/nonEOS'}
     elif time.monotonic()+180>=deadline:command={'action':'stop','reason':'reserved cleanup'}
     else:
      for path in sorted(queue.glob('q*.json')):
       if path.name in files_seen:continue
       try:raw=path.read_bytes();assert len(raw)<=131072;command=json.loads(raw)
       except (OSError,ValueError):continue
       files_seen.add(path.name);break
      if command is None:
       if time.perf_counter()-last_idle_heartbeat>=5:
        last_idle_heartbeat=time.perf_counter();report['resident_idle_heartbeat']={'monotonic':last_idle_heartbeat,'completed_cases':len(report['batch_cases']),'deadline_remaining_s':deadline-time.monotonic(),'files_consumed':len(files_seen)}
        if g.get('progress_save'):g['progress_save']({'label':'resident-idle','steps':poll,'active_C':0})
       time.sleep(.5)
    command=control.collect('batch-queue-poll:'+str(poll),{'command':command})[0]['command'];poll+=1
    if command is None:continue
    if command.get('action')=='stop':report['resident_stop']=command;save();break
    execute(command)
  report['frontend_C32_incomplete']=[{'service_id':key,'arrived_requests':32,'completed_native_cohorts':len(parts),'completed_requests':sum(len(x['request_metrics'])for x in parts),'unfinished_queued_requests':32-sum(len(x['request_metrics'])for x in parts),'reason':'boundedstop or remainingbudget beforesecondcohort'}for key,parts in service_parts.items()if len(parts)<2];report['serving_protocol_complete']=True;report['phase']='batch_protocol_complete';save()
 finally:
  native.arm([],[],[],False)
  for ownedpeer in native.peers.values():ownedpeer.close()
  if node==0 and sock is not None:
   try:rpc('close')
   finally:sock.close()
  if peer is not None:peer.close()
  if server is not None:server.close()
