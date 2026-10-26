import pathlib,json,subprocess,hashlib,datetime,collections,statistics
"""CPU postprocessing only: natural-EOS+taskcorrect gates, separate from speed-only metrics."""
import math

def task_goodput(requests,window_seconds,cards,decode_common_seconds=None,minimum_user_decode=15.):
 assert window_seconds>0 and cards>0
 counts={'arrivals':len(requests),'natural_EOS_correct':0,'natural_EOS_correct_SLO15':0,'wrong_or_parsefailed_natural_EOS':0,'capped_even_if_correct':0,'unfinished_or_failed_or_rejected':0};tokens=good_tokens=decode_tokens=good_decode_tokens=0
 for req in requests:
  ev=[e for e in req.get('events',[])if e.get('visible')is True and e.get('kind')=='committed'and not any(e.get(x,False)for x in ['EOS','pad','draft','replay'])];stamps=[e['time']for e in ev];assert stamps==sorted(stamps)
  if req.get('status')!='completed':counts['unfinished_or_failed_or_rejected']+=1;continue
  if req.get('stop_reason')not in ['EOS','stop']:counts['capped_even_if_correct']+=1;continue
  if req.get('score',{}).get('flexible_correct')is not True:counts['wrong_or_parsefailed_natural_EOS']+=1;continue
  counts['natural_EOS_correct']+=1;tokens+=len(ev);nd=sum(e.get('phase')=='decode'for e in ev);decode_tokens+=nd;rate=(len(stamps)-1)/(stamps[-1]-stamps[0])if len(stamps)>1 and stamps[-1]>stamps[0]else None
  if rate is not None and rate>minimum_user_decode:counts['natural_EOS_correct_SLO15']+=1;good_tokens+=len(ev);good_decode_tokens+=nd
 assert sum(counts[k]for k in ['natural_EOS_correct','wrong_or_parsefailed_natural_EOS','capped_even_if_correct','unfinished_or_failed_or_rejected'])==len(requests)
 return {'CPU_only':True,'gate':'completed AND naturalEOS/stop AND flexible_task_correct; optionalstrictuserdecode>15intersection','counts':counts,'correct_natural_request_goodput_s':counts['natural_EOS_correct']/window_seconds,'correct_natural_output_tok_s_per_card':tokens/window_seconds/cards,'correct_natural_SLO15_request_goodput_s':counts['natural_EOS_correct_SLO15']/window_seconds,'correct_natural_SLO15_output_tok_s_per_card':good_tokens/window_seconds/cards,'correct_natural_decode_tok_s_per_card_common_window':decode_tokens/decode_common_seconds/cards if decode_common_seconds and decode_common_seconds>0 else None,'correct_natural_SLO15_decode_tok_s_per_card_common_window':good_decode_tokens/decode_common_seconds/cards if decode_common_seconds and decode_common_seconds>0 else None,'quality_criterion':'GSM8K flexible numericfinalanswer scorer, notformal fullsuite/noninferiorityproof','native_vs_service':'Common-window denominator includesserialmicro/control/hostgaps; notsumCAPI denominator and notHTTPclientclock'}


r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();launch=json.loads((r/'GLM-G100A-r9136-launch.json').read_text());run=launch['run'];q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'G100A readonly same694 scalar compact; source/live notmodified','launch':launch,'owners':[],'service':[],'phase_timing_by_owner':[],'original_failure677_preserved':True}
for rel in ['results/GLM-G100A-r9136-watch.json','results/GLM-G100A-r9136-audit.json','incoming/GLM-ragged-Model-r9136/GPU-manifest.json']:
 p=r/rel
 if p.exists():
  d=json.loads(p.read_text());q[rel]={'path':str(p),'SHA256':sha(p),'data':d if 'watch' in rel else {k:d.get(k)for k in ['qualified','Slurm','run','scope','model_SHA256','program_SHA256']}}
for stage in [0,1]:
 for rank in range(8):
  p=r/'results'/f'glm-actual-concurrency-r9136-{run}-stage{stage}-rank{rank}.json';phase=r/'results'/f'GLM-Model-phase-r9136-{run}-stage{stage}-rank{rank}.jsonl';o={'stage':stage,'rank':rank}
  if phase.exists():o['last_phase']=json.loads(phase.read_text().splitlines()[-1])
  if not p.exists():q['owners'].append(o);continue
  d=json.loads(p.read_text());o.update(path=str(p),SHA256=sha(p),report={k:d.get(k)for k in ['completed','actual_producer','benchmark_complete','budget_partial','normal_reset_close','MMLU_progress','service_progress','minimum_driver_free_bytes','actual_owner_weight_load_wall_seconds','worker_shutdown_completed','post_close_Torch_liveness','library_sha256','cache_after_destroy','host_bank_after_destroy_r4504','actual_final_cohort_counts','actual_final_ragged_counts','fixed_resident_mask']});q['owners'].append(o)
  groups={}
  for event in d.get('native_events',[]):
   key=event['label'];a=groups.setdefault(key,{'calls':0,'native_seconds':0.,'rows_histogram':collections.Counter(),'past_equal_calls':0,'past_varlen_calls':0,'cohort_counter_delta':[0]*4,'ragged_counter_delta':[0]*4});a['calls']+=1;a['native_seconds']+=event['native_seconds'];a['rows_histogram'][len(event['rows'])]+=1;a['past_equal_calls'if event['past_min']==event['past_max']else'past_varlen_calls']+=1;a['cohort_counter_delta']=[x+y for x,y in zip(a['cohort_counter_delta'],event['cohort_counter_delta'])];a['ragged_counter_delta']=[x+y for x,y in zip(a['ragged_counter_delta'],event['ragged_counter_delta'])]
  q['phase_timing_by_owner'].append({'stage':stage,'rank':rank,'groups':groups})
  if stage!=1 or rank!=0:continue
  q['MMLU']=[{**{k:a.get(k)for k in ['ON','elapsed_seconds','M','forced_empty_thinking','classification_not_generation']},'n':len(a['cases']),'correct':sum(x['flexible_correct']for x in a['cases']),'cases':a['cases']}for a in d.get('MMLU',[])];q['paired_MMLU']=d.get('paired_MMLU');q['paired_GSM']=d.get('paired_GSM');q['coverage']=d.get('benchmark_coverage')
  for arm in d.get('service',[]):
   a={k:v for k,v in arm.items()if k not in ['requests','task_scores']};reqs=arm['requests'];a['quality_goodput']=task_goodput(reqs,arm['elapsed_seconds'],16,arm.get('decode_common_window_seconds'));a['request_summary']=[]
   for req in reqs:
    ev=req.get('events',[]);ts=[x['time']for x in ev];itl=[y-x for x,y in zip(ts,ts[1:])];a['request_summary'].append({**{k:req.get(k)for k in ['id','status','stop_reason','input_tokens','score','end_time']},'visible_committed_tokens':len(ev),'first_token_ready':ts[0]if ts else None,'last_token_ready':ts[-1]if ts else None,'user_decode_tok_s':(len(ts)-1)/(ts[-1]-ts[0])if len(ts)>1 and ts[-1]>ts[0]else None,'ITL_max':max(itl)if itl else None,'ITL_mean':statistics.mean(itl)if itl else None,'generated_ids_SHA256':hashlib.sha256(json.dumps(req.get('generated_ids')).encode()).hexdigest(),'text_SHA256':hashlib.sha256(req.get('text','').encode()).hexdigest()})
   q['service'].append(a)
q['Slurm']=[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed,NodeList%100'],text=True).splitlines()if 'G100A'in x];print(json.dumps(q))
