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

def selftest():
 events=[{'time':1.,'visible':True,'kind':'committed','phase':'prefill'},{'time':1.01,'visible':True,'kind':'committed','phase':'decode'}]
 base={'status':'completed','stop_reason':'EOS','score':{'flexible_correct':True},'events':events}
 data=[base,{**base,'score':{'flexible_correct':False}},{**base,'stop_reason':'max_new_tokens'},{**base,'status':'in_progress'}];q=task_goodput(data,2.,16,1.);assert q['counts']['natural_EOS_correct']==q['counts']['natural_EOS_correct_SLO15']==1 and q['counts']['capped_even_if_correct']==q['counts']['unfinished_or_failed_or_rejected']==q['counts']['wrong_or_parsefailed_natural_EOS']==1;return {'passed':True,'CPU_only':True,'scope':'postprocessqualitygatefixture, noGPUperf'}
if __name__=='__main__':
 import json,sys,pathlib
 if len(sys.argv)==1:print(json.dumps(selftest()))
 else:
  src=pathlib.Path(sys.argv[1]);q=json.loads(src.read_text());out=[{'label':a['label'],'C':a['C'],'M':a['M'],'ON':a['ON'],'quality_goodput':task_goodput(a['requests'],a['elapsed_seconds'],16,a.get('decode_common_window_seconds'))}for a in q['service']];print(json.dumps({'source':str(src),'results':out},indent=2))
