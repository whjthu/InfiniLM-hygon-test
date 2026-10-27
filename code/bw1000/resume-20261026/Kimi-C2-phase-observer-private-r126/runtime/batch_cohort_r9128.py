import time,os,json,math,re
import numpy as np
from batch_policy_r9128 import greedy
from adaptive_width_r9223 import width_cap
class Request:
 def __init__(self,logical,prompt,cap,epoch,quality=None):
  self.logical=logical;self.prompt=list(prompt);self.cap=cap;self.epoch=epoch;self.quality=quality;self.past=len(prompt);self.draft_past=0;self.append_rows=len(prompt);self.features=None;self.tokens=[];self.events=[];self.done=False;self.finished=None;self.prefill_EOS=False;self.cycles=[]
 def emit(self,tokens,stamp,eos):
  for token in tokens:self.tokens.append(int(token));self.events.append({'time':stamp,'token':int(token),'EOS':token in eos})
  self.done=self.tokens[-1]in eos or len(self.tokens)>=self.cap
 def metrics(self,arrival,eos):
  visible=[e for e in self.events if not e['EOS']];N=len(visible);first=visible[0]['time']if N else None;last=visible[-1]['time']if N else None;gaps=[b['time']-a['time']for a,b in zip(visible,visible[1:])];span=last-first if N else None;clock=self.finished-first if N else None
  return {'logical':self.logical,'request_key':[self.epoch,self.logical],'input_tokens':len(self.prompt),'output_cap':self.cap,'generated_ids':self.tokens,'native_ready_events':list(self.events),'generated_tokens':len(self.tokens),'useful_tokens':N,'useful_decode_tokens':max(0,N-1),'first_ready_monotonic':first,'last_ready_monotonic':last,'completion_monotonic':self.finished,'native_TTFT_s':first-arrival if N else None,'user_firstlast_tok_s':(N-1)/span if N>1 and span>0 else None,'decode_user_tok_s':(N-1)/clock if N>1 and clock>0 else None,'native_decode_user_wall_s':clock,'ITL_P50_s':float(np.percentile(gaps,50))if gaps else None,'ITL_P95_s':float(np.percentile(gaps,95))if gaps else None,'ITL_P99_s':float(np.percentile(gaps,99))if gaps else None,'ITL_max_s':max(gaps)if gaps else None,'EOS':bool(self.tokens and self.tokens[-1]in eos),'truncated':not(self.tokens and self.tokens[-1]in eos)and len(self.tokens)>=self.cap,'prefill_EOS_native_row_phase1_retired_frontend':self.prefill_EOS,'cycles':self.cycles,'clock_scope':'one native emitter common monotonic; includes thinking/channel/control tokens, not HTTP visibleanswer'}
def cohort(native,control,rpc,node,emitter,requests,label,width,strategy,I192,epoch,eos,deadline,progress=None,draft_projected_cache=False,adaptive_width=False):
 assert type(adaptive_width)is bool and (not adaptive_width or strategy!='ar');assert strategy in ['ar','oldspec','state'];assert 1<=len(requests)<=native.maxC;assert(strategy=='ar'and width==1)or(strategy!='ar'and 2<=width<=8);assert len(requests)*width<=native.maxT
 for req in requests:assert req.epoch==epoch and 1<=len(req.prompt)<=1024 and 2<=req.cap<=1024 and len(req.prompt)+req.cap<=2048
 os.environ['INFINILM_KIMI_I192_GROUPED_R8669Y']=str(int(I192));os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=str(int(I192));control.collect('case-config:'+label,{'I192':I192,'strategy':strategy});native.reset(strategy=='state');initial_counts=native.I192();initial_stats=native.stats()
 assert type(draft_projected_cache)is bool
 if node==0:reply,_=rpc('begin_cohort',epoch=epoch,projected_cache=draft_projected_cache);assert reply['epoch']==epoch and reply['projected_cache']==draft_projected_cache
 control.collect('draft-cohort:'+label,{});arrival=time.perf_counter();arrival_observer=native.observer.snapshot()if native.observer else None;decode_observer=None;rounds=[];steps=0;prefill_calls=[]
 for begin in range(0,len(requests),1):
  micro_start=time.perf_counter()if native.phase.enabled else None;p_micro=native.phase.begin('prefill.micro',begin=begin);group=requests[begin:begin+1];ids=[q.logical for q in group]
  def prefill_emit(selected):
   nonlocal decode_observer
   stamp=time.perf_counter();assert len(selected)==len(group)
   for q,token in zip(group,selected):q.emit([token],stamp,eos);q.prefill_EOS=q.done
   if begin+len(group)==len(requests)and native.observer:decode_observer=native.observer.snapshot()
  _,features,record=native.call('prefill',ids,[0]*len(group),[q.prompt for q in group],strategy!='ar',prefill_emit);prefill_calls.append(record)
  for q in group:q.features=features[q.logical]if node==0 and strategy!='ar'else None
  for q in group:
   if q.done:
    if node==0:rpc('retire',epoch=epoch,request=q.logical)
    q.finished=time.perf_counter()
  if native.memory_guard:native.phase.call('memory_guard',native.memory_guard,'prefill:'+label+':'+str(begin))
  native.phase.end(p_micro);record['micro_prefill_host_start']=micro_start;record['micro_prefill_host_end']=time.perf_counter()if native.phase.enabled else None
 # Allprefill calls use true packed native subsets; no pretending serial B1 decode is batch.
 first_times=[q.events[0]['time']for q in requests];decode_start=max(first_times);prefill_complete=time.perf_counter()
 while any(not q.done for q in requests):
  if time.monotonic()>deadline:raise TimeoutError('reserved batch cleanup deadline')
  tick=time.perf_counter();active=[q for q in requests if not q.done];ids=[q.logical for q in active];wave_width=width_cap(len(active),native.maxT,8)if adaptive_width else width;effective={q.logical:(1 if strategy=='ar'else min(wave_width,q.cap-len(q.tokens),2048-q.past))for q in active};assert sum(effective.values())<=native.maxT;p_cycle=native.phase.start_cycle(steps,ids,[effective[x]for x in ids]);before=native.phase.call('native.rows_before',native.rows,ids);packet=None
  if node==0:
   from batch_frames_r9142 import request_group
   packet=native.phase.call('draft.request_group_fullRPC',request_group,rpc,active,effective,epoch)
  packet=control.collect('draft-batch:'+label+':'+str(steps),{'packet':packet})[0]['packet'];packets={v['logical']:v for v in packet['requests']};inputs=[]
  for q in active:
   props=packets[q.logical]['proposals'];assert len(props)==effective[q.logical]-1
   if props:q.draft_past+=q.append_rows
   inputs.append([q.tokens[-1]]+props)
  decisions=[]
  def verified(selected):
   offset=0;stamp=time.perf_counter()
   for q,values in zip(active,inputs):
    d=greedy(values[0],values[1:],selected[offset:offset+len(values)],eos,q.cap-len(q.tokens));decisions.append(d);q.emit(d.emitted,stamp,eos);offset+=len(values)
   assert offset==len(selected)
  _,features,verify=native.call('verify',ids,[q.past for q in active],inputs,strategy!='ar',verified);commitwall=native.phase.call('commit.whole',native.commit,ids,decisions);partials=[(q.logical,q.past,len(d.commit_inputs))for q,d in zip(active,decisions)if d.needs_replay];replaywall=0.;replayfeatures=None
  if partials:replayfeatures,rep=native.phase.call('replay.whole',native.replay,partials,strategy=='state');replaywall=rep['CAPI_maxhost_s']
  after=native.phase.call('native.rows_after',native.rows,ids)
  for q,d,old,new in zip(active,decisions,before,after):
   if d.terminal:assert new[3]==4
   else:
    assert new==[old[1],old[0],q.past+len(d.commit_inputs),1],(q.logical,old,new,d);q.past+=len(d.commit_inputs);q.append_rows=len(d.commit_inputs)
    if node==0 and strategy!='ar':q.features=replayfeatures[q.logical]if d.needs_replay and strategy=='oldspec'else features[q.logical][:q.append_rows*35840*2]
   q.cycles.append({'step':steps,'accepted_proposals':d.accepted,'proposed':len(inputs[ids.index(q.logical)])-1,'useful_emitted':sum(x not in eos for x in d.emitted),'retained_inputs':len(d.commit_inputs),'terminal':d.terminal,'needs_replay':d.needs_replay,'actual_before_row':old,'actual_after_row':new,'draft':packets[q.logical]})
   if q.done:
    if node==0:reply,_=native.phase.call('draft.retire_RPC',rpc,'retire',epoch=epoch,request=q.logical);assert reply['retired']
    q.finished=time.perf_counter()
  if native.memory_guard:native.phase.call('memory_guard',native.memory_guard,'decode:'+label+':'+str(steps))
  row={'step':steps,'planned_width_cap':wave_width,'actual_active_C':len(active),'logical':ids,'widths':list(map(len,inputs)),'flattened_T':sum(map(len,inputs)),'draft_batch_RPC':packet.get('batch_draft',{}),'draft_RPC_group_s':packet['group_wall_s'],'draft_RPC_sum_s':sum(x['RPC_s']for x in packet['requests']),'draft_query_sum_s':sum(x['draft_query_s']for x in packet['requests']),'per_request_draft':packet['requests'],'verify_CAPI_maxhost_s':verify['CAPI_maxhost_s'],'commit_local_s':commitwall,'replay_CAPI_maxhost_s':replaywall,'cycle_local_wall_s':time.perf_counter()-tick};rounds.append(row);steps+=1
  if progress and steps%8==0:native.phase.call('progress_save',progress,{'label':label,'steps':steps,'active_C':sum(not q.done for q in requests)})
  native.phase.end_cycle(p_cycle)
 end=time.perf_counter();metrics=[q.metrics(arrival,eos)for q in requests];final_counts=native.I192();deltas=[[b-a for a,b in zip(x,y)]for x,y in zip(initial_counts,final_counts)];assert all(v>=0 for row in deltas for v in row)
 if not I192:assert deltas==[[0,0]]*8
 return {'host_observer_whole_cohort':native.observer.delta(arrival_observer,end)if native.observer else None,'host_observer_decode_window':native.observer.delta(decode_observer,end)if native.observer and decode_observer else None,'prefill_micro_request_cap':1,'label':label,'strategy':strategy,'draft_projected_cache':draft_projected_cache,'cohort_epoch':epoch,'configured_C':len(requests),'configured_width':width,'adaptive_width':adaptive_width,'width_policy':'power2floor64byactive_capped8'if adaptive_width else'fixed','actual_native_batch':True,'continuous_admission':False,'I192_enabled':bool(I192),'I192_counts_before8x2':initial_counts,'I192_counts_after8x2':final_counts,'I192_counts_delta8x2':deltas,'state_stats_before64':initial_stats,'state_stats_after64':native.stats(),'admission_native_monotonic':arrival,'decode_window_start_native_monotonic':decode_start,'end_native_monotonic':end,'prefill_all_complete_s':prefill_complete-arrival,'decode_common_window_s':end-decode_start,'total_useful_decode_tokens':sum(q['useful_decode_tokens']for q in metrics),'aggregate_decode_tok_s':sum(q['useful_decode_tokens']for q in metrics)/(end-decode_start),'all_serving_GPU_cards':native.card_count,'aggregate_decode_tok_s_per_card':sum(q['useful_decode_tokens']for q in metrics)/(end-decode_start)/native.card_count,'request_metrics':metrics,'cycles':rounds,'prefill_calls':prefill_calls,'phase_observer':native.phase.export(),'phase_observer_summary':__import__('phase_observer_r126').summarize(native.phase.export()),'prefill_interference':__import__('phase_observer_r126').prefill_interference(requests,prefill_calls),'draft_execution':'true padded Torch batch; separate request feature cache; logicalRPC+binarycontinuation timing charged','client_HTTP_metrics':'UNKNOWN','EOS_rejected_padding_not_counted_as_useful':True}
