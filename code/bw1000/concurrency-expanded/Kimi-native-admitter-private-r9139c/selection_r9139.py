import math,statistics,copy
from quality_bench_r9100 import score_case

def summarize(case,eos,cards=33):
 assert case['actual_native_batch']and not case['continuous_admission'];duration=case['end_native_monotonic']-case['decode_window_start_native_monotonic'];full=case['end_native_monotonic']-case['admission_native_monotonic'];assert duration>0 and full>0
 users=[];alltokens=goodtokens=slo_tokens=0
 for user in case['request_metrics']:
  ids=user['generated_ids'];nonEOS=sum(t not in eos for t in ids);decode=max(0,nonEOS-1);assert decode==user['useful_decode_tokens'];first,last=user['first_ready_monotonic'],user['last_ready_monotonic'];rate=(nonEOS-1)/(last-first)if nonEOS>1 and last>first else None;given=user['user_firstlast_tok_s'];assert rate is None and given is None or rate is not None and math.isclose(rate,given,rel_tol=1e-7)
  natural=bool(user['EOS']and ids and ids[-1]in eos and not user['truncated']);quality=user.get('quality')or{};scored=score_case({'id':str(quality.get('sample_index')),'gold_numeric':quality.get('expected')},quality.get('response',''),'EOS'if natural else'length');correct=scored['flexible_correct'];good=natural and correct;meets=good and rate is not None and rate>15
  alltokens+=decode;goodtokens+=decode if good else 0;slo_tokens+=decode if meets else 0
  partial=[];cycles=user.get('cycles',[])
  for j,c in enumerate(cycles):
   if c.get('needs_replay')and not c['terminal']:
    old,new=c['actual_before_row'],c['actual_after_row'];assert new==[old[1],old[0],old[2]+c['retained_inputs'],1]
    if j+1<len(cycles):assert cycles[j+1]['actual_before_row']==new;partial.append({'step':c['step'],'before':old,'after':new,'nextstep':cycles[j+1]['step']})
  users.append({'logical':user['logical'],'sample_index':quality.get('sample_index'),'input_tokens':user['input_tokens'],'cap':user.get('output_cap',case['command'].get('cap')),'actual_nonEOS_tokens':nonEOS,'decode_tokens':decode,'native_user_firstlast_tok_s':rate,'native_conservative_decode_tok_s':user.get('decode_user_tok_s'),'naturalEOS':natural,'correct':correct,'strict_correct':scored['strict_correct'],'parsefail':scored['parse_failed'],'captruncated':user['truncated'],'native_SLO_above15':meets,'partial_then_next':partial[:1]})
 C=case['configured_C'];W=case['configured_width'];assert len(users)==C;eligible=sum(c['flattened_T']==16 for c in case['cycles']);signature=(C,W,tuple((u['sample_index'],u['input_tokens'],u['cap'])for u in users))
 return {'label':case['label'],'strategy':case['strategy'],'C':C,'W':W,'I192':case['I192_enabled'],'signature':signature,'window_s':duration,'fullwindow_s':full,'all_decode_tokens':alltokens,'diagnostic_tok_s':alltokens/duration,'completed_correct_decode_tokens':goodtokens,'completed_correct_goodput_tok_s':goodtokens/duration,'completed_correct_goodput_per_card':goodtokens/duration/cards,'native_SLO_goodput_per_card':slo_tokens/duration/cards,'completed_correct_requests':sum(u['naturalEOS']and u['correct']for u in users),'all_natural_correct':all(u['naturalEOS']and u['correct']for u in users),'all_users_above15':all(u['native_SLO_above15']for u in users),'min_user_rate':min((u['native_user_firstlast_tok_s']or 0)for u in users),'eligible_T16_calls':eligible,'users':users,'prefill_s':case['prefill_all_complete_s'],'max_cycle_s':max(c['cycle_local_wall_s']for c in case['cycles'])if case['cycles']else 0,'goodput_source':'actual committed generatedIDs excludingEOS/firstprefill, actual native commonwindow+first/last boundaries; full pereventtimestamps notpersisted; not HTTP'}

def select(reports,eos):
 assert len(reports)==4 and all(not r.get('error')and not r.get('shutdown_error')for r in reports)
 assert all(len(r.get('batch_cases',[]))>=11 for r in reports);em=reports[2];summaries=[summarize(c,eos)for c in em['batch_cases'][:11]];candidates=[];rejected=[]
 for s in summaries:
  if s['strategy']!='state'or(s['C'],s['W'])not in [(2,8),(4,4)]:continue
  matching=[]
  for r in reports:
   c=next(c for c in r['batch_cases']if c['label']==s['label']);stats=c['state_stats_after64'];assert len(stats)==64 and all(stats[i]>0 and stats[i+1]>0 for i in range(0,64,8));counts=c['I192_counts_delta8x2'];matching.append({'node':r['node'],'state_stats64':stats,'I192_delta8x2':counts})
  if not s['all_natural_correct']:rejected.append({'label':s['label'],'reason':'statecohort not all naturallyEOS andtaskcorrect'});continue
  s['owner_evidence']=matching
  if s['I192']:
   activation=s['eligible_T16_calls']>0 and all(all(row[0]>0 and row[1]>0 for row in m['I192_delta8x2'])for m in matching)
   off=[x for x in summaries if x['strategy']=='state'and not x['I192']and x['signature']==s['signature']];offrate=statistics.median(x['completed_correct_goodput_tok_s']for x in off)if off else None
   if not activation or offrate is None or s['completed_correct_goodput_tok_s']<=offrate:rejected.append({'label':s['label'],'reason':'ON not actualallowneractivated+wholecyclepositivevsOFF','activation':activation,'OFF_median_goodput_tok_s':offrate});continue
   s['OFF_median_goodput_tok_s']=offrate;s['measured_wholecycle_gain']=s['completed_correct_goodput_tok_s']/offrate-1 if offrate else None
  candidates.append(s)
 assert candidates,'no basic naturallycomplete/correct statecohort to qualify admission'
 # Collapse repeated trialsof same config by median, preserveeach trialand itsusers.
 groups={}
 for c in candidates:groups.setdefault((c['C'],c['W'],c['I192']),[]).append(c)
 options=[]
 for key,trials in groups.items():
  rate=statistics.median(c['completed_correct_goodput_per_card']for c in trials);options.append({'C':key[0],'W':key[1],'I192':key[2],'trials':trials,'goodput_per_card_median':rate,'all_users_above15':all(c['all_users_above15']for c in trials),'min_user_rate':min(c['min_user_rate']for c in trials)})
 viable=[o for o in options if o['all_users_above15']];pool=viable or options;frontier=[o for o in pool if not any(p is not o and p['goodput_per_card_median']>=o['goodput_per_card_median']and p['min_user_rate']>=o['min_user_rate']and(p['goodput_per_card_median']>o['goodput_per_card_median']or p['min_user_rate']>o['min_user_rate'])for p in pool)];chosen=max(frontier,key=lambda o:(o['goodput_per_card_median'],o['min_user_rate']))
 return {'chosen':chosen,'options':options,'frontier':frontier,'user_above15_goal_met_in_selected_smokes':bool(viable),'not_general15_or_qualityparity':True,'rejected':rejected,'initial11_summaries':summaries,'selection_policy':'naturalEOS+taskcorrect state only; actual32owneractivation; ON needs measured netwholecyclegain vsmatchedOFF; prioritizeeveryuser>15 thencompletedcorrectgoodput/all33cards; ifnoneSLO select measuredPareto bygoodput thenminuser andmarkunmet'}
