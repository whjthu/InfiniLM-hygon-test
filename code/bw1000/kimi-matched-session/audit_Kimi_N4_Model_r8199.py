"""Independent terminal audit, finite B16M4 same-shape only; never auto-promotes."""
import pathlib,json,hashlib,subprocess,datetime,statistics,math
def summary(v):
 v=sorted(v)
 if not v:return None
 def q(p):
  a=(len(v)-1)*p;i=int(a);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(a-i)
 return {'mean':statistics.mean(v),'min':min(v),'max':max(v),'p50':q(.5),'p95':q(.95),'count':len(v)}
def metrics(c,eos):
 ids=c['generated_ids'];ts=c['timestamps_after_common_anchor'];first=c['TTFT_seconds'];anchor=c['service_wall_seconds']-c['decode_wall_seconds'];wall=c['decode_wall_seconds'];assert len(ids)==len(ts)==len(first)==16 and wall>0;user=[];post=[];num=[];gaps=[]
 for seq,times,t0 in zip(ids,ts,first):
  assert len(seq)==len(times)and times[0]==0 and not any(v in eos for v in seq[:-1])and(len(seq)==128 or seq[-1]in eos)
  events=[(anchor+t,v)for t,v in zip(times[1:],seq[1:])];end=events[-1][0]if events else t0;assert all(math.isfinite(t)and t>=t0 for t,v in events)and all(b>=a for(a,_),(b,_)in zip(events,events[1:]));assert end<=anchor+wall+1e-7
  a=sum(v not in eos and t0<t<=end for t,v in events);b=sum(v not in eos and anchor<t<=end for t,v in events);num.append(b);user.append(a/(end-t0)if end>t0 else None);post.append(b/(end-anchor)if end>anchor else None);stamps=[t0]+[t for t,v in events];gaps.extend(b-a for a,b in zip(stamps,stamps[1:]))
 return {'first_ready_request_rates':user,'first_ready_request_rate_summary':summary([v for v in user if v is not None]),'postanchor_request_rates':post,'aggregate_postanchor_count':sum(num),'aggregate_decode_tok_s':sum(num)/wall,'aggregate_decode_tok_s_per_card':sum(num)/wall/32,'TTFT_seconds':summary(first),'TBT_seconds':summary(gaps),'prefill_wall_seconds':c['prefill_wall_seconds'],'prefill_input_tok_s':sum(c['input_lengths'])/c['prefill_wall_seconds'],'scope':'Internal emitter firstready includes laterprefill/anchor waiting perrequest; postanchor numerator independently timefiltered; firstprefillID/EOS excluded. No HTTP. usermean not aggregate/B.'}
def audit():
 r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();launch=json.loads((r/'Kimi-N4-Model-r8199-launch.json').read_text());jobs=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-23','-j','910371','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed'],text=True).splitlines()if s.split('|')[1]=='Kimi-N4-Model-r8199'];out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'run':launch['run'],'jobs':jobs,'performance_adopted':False,'candidate_qualified':False,'issues':[],'nodes':[],'cases':[],'qualification_scope':'B16M4/cap32/cache4 only; noB8/B32/B64 orSpecScope. finalrootreviewrequired'}
 if len(jobs)!=1 or jobs[0][2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:out['state']='NOT_TERMINAL';return out
 issues=out['issues'];reports=[]
 if jobs[0][2:4]!=['COMPLETED','0:0']:issues.append('SlurmnotCOMPLETED0')
 for rank in range(4):
  p=r/'results'/f'kimi-N4-r8199-{launch["run"]}-node{rank}.json'
  if not p.exists():issues.append('missing_node_'+str(rank));continue
  q=json.loads(p.read_text());reports.append(q);out['nodes'].append({'rank':rank,'SHA256':sha(p),**{k:q.get(k)for k in['completed','error','shutdown_error','smallB_quality_passed','smallB_matrix_completed','smallB_status','smallB_NOT_RUN','all8_owned_workers_shutdown_completed','final_N4_flag','final_serving_policy']}})
  if not(q.get('completed')and q.get('protocol_before_destroy_passed')and q.get('all8_owned_workers_shutdown_completed'))or q.get('error')or q.get('shutdown_error'):issues.append(f'rank{rank}:lifecycle')
  if q.get('library_sha256')!=launch['DSO_SHA256']or q.get('libkimi_grouped_prefill.so_sha256')!=launch['helper_SHA256']:issues.append(f'rank{rank}:binarybinding')
  if q.get('final_N4_flag')!='0'or q.get('final_serving_policy')!='0':issues.append(f'rank{rank}:notrestoredflags')
 if len(reports)!=4:out['state']='TERMINAL_INCOMPLETE_REPORTS';return out
 fp=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json';assert sha(fp)=='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625';f=json.loads(fp.read_text());gold=next(z['fixed_complete_IDs']for z in f['geometries']if(z['B'],z['M'])==(16,4));eos=set(f['eos_token_ids']);tail=reports[3];checks=tail.get('smallB_checks',[]);lookups=[{c['label']:c for c in q.get('serving_cases',[])}for q in reports];expected_labels=['N4-parent-B16M4-history','N4-candidate-B16M4-quality','N4-warm-0','N4-warm-1','N4-ABBA-0','N4-ABBA-1','N4-ABBA-2','N4-ABBA-3','N4-restore-B16M4'];expected_mode=[0,1,0,1,0,1,1,0,0]
 for check in checks:
  label=check['label'];assert label in expected_labels;c=lookups[3][label];limit=check['limit'];ids=c['generated_ids'];exact=ids==[z[:limit]for z in gold];cross=all(label in z and z[label]['generated_ids']==ids for z in lookups)
  if not cross:issues.append(label+':crossnodeIDs')
  if not exact:issues.append(label+':numeric_completeIDs')
  if (c['B'],c['frontend_cohort_M'],c['cache_pages_per_request'])!=(16,4,4)or c['N4_candidate_requested']!=bool(expected_mode[expected_labels.index(label)])or not c['timed_decode_token_only']:issues.append(label+':geometry/mode')
  out['cases'].append({'label':label,'candidate':check['candidate'],'limit':limit,'full_historical_IDs_EOS_exact':exact,'crossnode_IDs_exact':cross,'warmup':limit==8,'metrics':metrics(c,eos)if limit==128 else{'excluded_warmup':True}})
 if any(q.get('smallB_checks')!=checks for q in reports):issues.append('crossnodechecks')
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());qa_expected=2*len(chat['cases'])
 for rank,q in enumerate(reports):
  samples=q.get('local_forward_samples',[]);labels=[v['label']for v in samples];assert len(labels)==len(set(labels));cases=lookups[rank]
  for key,width in [('packed_W2_dispatch',8),('bounded_BF16_dispatch',4)]:
   rows=q.get(key,[])
   if [x['label']for x in rows]!=labels or any(x['enabled']is not False or x['delta']!=[[0]*width]*8 for x in rows):issues.append(f'rank{rank}:{key}')
  if [x['label']for x in q.get('counter_dispatch',[])]!=labels or any(x['delta']!=[x['expected']]*8 for x in q.get('counter_dispatch',[])):issues.append(f'rank{rank}:servingdispatch')
  rows=q.get('metadata_dispatch',[])
  if [x['label']for x in rows]!=labels or any(x['enabled']is not False or len(x['verified'])!=8 for x in rows):issues.append(f'rank{rank}:metadata')
  n4=q.get('N4_dispatch_r8167',[]);lo,hi=q['PP_ranges'][rank];moe=sum(i>=1 for i in range(lo,hi))
  if [x['label']for x in n4]!=labels:issues.append(f'rank{rank}:N4labels')
  for row in n4:
   matching=[k for k in cases if row['label'].startswith(k+'-')];assert len(matching)==1,(row['label'],matching);case=cases[matching[0]];expected=[moe,4*moe]if case['N4_candidate_requested']and row['N']==4 else[0,0]
   if row['delta']!=[expected]*8 or row['expected']!=expected:issues.append(f'rank{rank}:N4actualdispatch:{row["label"]}')
  qa=q.get('QA',[])
  for epoch,candidate in[('OFF-before',False),('ON-before',True),('OFF-after',False)]:
   subset=[v for v in qa if v['epoch']==epoch]
   if len(subset)!=qa_expected:issues.append(f'rank{rank}:QAcount:{epoch}')
   for z in subset:
    label=f'QA-{epoch}-p{z["policy"]}-{z["task"]}';c=cases[label];key='sort'if z['task'].startswith('sort_probe')else z['task'];valid=[x['generated_ids'][0]for x in known['cases']if x['task']==key and x['grouped']]
    if c['generated_ids'][0]not in valid or c['N4_candidate_requested']!=candidate:issues.append(f'rank{rank}:QAIDs:{label}')
 out['resource_NOT_RUN']=tail.get('smallB_NOT_RUN',[]);out['restore_status']=tail.get('smallB_restore_status');out['actual_ABBA_count']=sum(c['label'].startswith('N4-ABBA-')for c in out['cases']);out['matrix_completed_independent']=[x['label']for x in checks]==expected_labels and not out['resource_NOT_RUN'];out['numeric_failure']=any(not c['full_historical_IDs_EOS_exact']for c in out['cases']);out['post_KFD_passed']=launch.get('post_KFD_passed',False)
 out['candidate_qualified']=bool(not issues and out['matrix_completed_independent']and out['restore_status']=='PASS'and out['post_KFD_passed']);out['state']='TERMINAL_INDEPENDENT_AUDIT'
 abba=[c for c in out['cases']if c['label'].startswith('N4-ABBA-')]
 if len(abba)==4:
  keys=['aggregate_decode_tok_s','aggregate_decode_tok_s_per_card']
  out['paired_ABBA_metrics']={key:{'OFF_mean':statistics.mean(abba[i]['metrics'][key]for i in[0,3]),'ON_mean':statistics.mean(abba[i]['metrics'][key]for i in[1,2])}for key in keys};out['paired_ABBA_user_mean']={'OFF':statistics.mean(abba[i]['metrics']['first_ready_request_rate_summary']['mean']for i in[0,3]),'ON':statistics.mean(abba[i]['metrics']['first_ready_request_rate_summary']['mean']for i in[1,2])}
 return out
if __name__=='__main__':print(json.dumps(audit()))
