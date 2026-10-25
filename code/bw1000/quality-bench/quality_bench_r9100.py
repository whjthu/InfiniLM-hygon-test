"""Quality and serving metrics; no reference-token equality acceptance gate."""
import decimal,math,random,re,statistics

def canonical_number(value):
    try:
        x=decimal.Decimal(str(value).strip().replace(',','').replace('$',''))
        return str(x.normalize()) if x.is_finite() else None
    except decimal.InvalidOperation:return None

def extract_numeric(text):
    strict=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',text)
    boxed=re.findall(r'\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}',text)
    flexible=re.findall(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?',text)
    return {'strict':canonical_number(strict[-1]) if strict else None,'boxed':canonical_number(boxed[-1]) if boxed else None,'flexible':canonical_number((strict or boxed or flexible)[-1]) if strict or boxed or flexible else None}

def score_case(case,text,stop_reason):
    parsed=extract_numeric(text);gold=canonical_number(case['gold_numeric'])
    assert gold is not None
    return {'id':case['id'],'gold_numeric':gold,'parsed':parsed,'strict_correct':parsed['strict']==gold,'flexible_correct':parsed['flexible']==gold,'parse_failed':parsed['flexible'] is None,'stop_reason':stop_reason,'hit_output_cap':stop_reason in ('max_new_tokens','length','context_capacity'),'reference_token_equality_required':False}

def paired_quality(baseline,candidate,seed=20261026,repeats=5000):
    a={v['id']:v for v in baseline};b={v['id']:v for v in candidate};assert len(a)==len(baseline) and len(b)==len(candidate),'duplicate question IDs are invalid';assert a and set(a)==set(b),'paired same question IDs required'
    ids=sorted(a);d=[int(b[i]['flexible_correct'])-int(a[i]['flexible_correct']) for i in ids]
    rng=random.Random(seed);boot=sorted(sum(rng.choice(d) for _ in d)/len(d) for _ in range(repeats))
    return {'n':len(ids),'baseline_accuracy':sum(a[i]['flexible_correct'] for i in ids)/len(ids),'candidate_accuracy':sum(b[i]['flexible_correct'] for i in ids)/len(ids),'accuracy_difference':statistics.mean(d),'paired_bootstrap95_percentile_interval':[boot[int(.025*(repeats-1))],boot[int(.975*(repeats-1))]],'baseline_correct_candidate_wrong':sum(a[i]['flexible_correct'] and not b[i]['flexible_correct'] for i in ids),'baseline_wrong_candidate_correct':sum(not a[i]['flexible_correct'] and b[i]['flexible_correct'] for i in ids),'both_wrong':sum(not a[i]['flexible_correct'] and not b[i]['flexible_correct'] for i in ids),'baseline_cap_count':sum(v['hit_output_cap'] for v in baseline),'candidate_cap_count':sum(v['hit_output_cap'] for v in candidate),'both_zero_accuracy':not any(a[i]['flexible_correct'] or b[i]['flexible_correct'] for i in ids),'claim':'Pilot evidence only; n32 or narrow/low-scoring/truncated sample cannot establish no quality degradation. Expand heldout benchmark and prespecify noninferiority margin.','same_output_tokens_required':False,'zero_observed_regressions_one_sided95_upper_probability':1-.05**(1/len(ids)) if not any(v<0 for v in d) else None,'bootstrap_degeneracy_warning':len(set(d))==1,'no_quality_parity_claim_from_degenerate_interval':True}

def quantile(values,p):
    if not values:return None
    x=sorted(values);at=(len(x)-1)*p;lo=int(at);hi=math.ceil(at)
    return x[lo]*(hi-at)+x[hi]*(at-lo) if lo!=hi else x[lo]

def describe(values):
    return {'n':len(values),'min':min(values) if values else None,'mean':statistics.mean(values) if values else None,'p50':quantile(values,.5),'p95':quantile(values,.95),'p99':quantile(values,.99),'max':max(values) if values else None}

def serving_metrics(requests,window_start,window_end,cards,minimum_user_decode=15.,ttft_slo=None):
    assert window_end>window_start and cards>0
    duration=window_end-window_start;finished=[];rates=[];ttfts=[];gaps=[];visible=good_tokens=0;good=speed_good=0;speed_good_tokens=0
    for q in requests:
        assert window_start<=q['arrival']<=window_end
        stamps=[float(e['time']) for e in q.get('events',[]) if e.get('visible') is True and e.get('kind')=='committed' and not any(e.get(k,False) for k in ('EOS','pad','draft','replay'))]
        assert stamps==sorted(stamps) and all(q['arrival']<=v<=window_end for v in stamps)
        visible+=len(stamps)
        if stamps:
            ttft=stamps[0]-q['arrival'];ttfts.append(ttft);gaps.extend(b-a for a,b in zip(stamps,stamps[1:]))
            rate=(len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 and stamps[-1]>stamps[0] else None
            if rate is not None:rates.append(rate)
        else:ttft=rate=None
        success=q.get('status')=='completed'
        if success:finished.append(q)
        meets_speed=success and rate is not None and rate>minimum_user_decode and (ttft_slo is None or ttft<=ttft_slo)
        natural=q.get('stop_reason') in ('EOS','stop') or any(e.get('kind')=='EOS' for e in q.get('events',[]))
        if meets_speed:speed_good+=1;speed_good_tokens+=len(stamps)
        if meets_speed and natural:good+=1;good_tokens+=len(stamps)
    return {'clock_scope':'Caller must declare client/emitter; one common monotonic clock required','window_seconds':duration,'cards':cards,'arrivals':len(requests),'completed':len(finished),'failed':sum(q.get('status')=='failed' for q in requests),'rejected':sum(q.get('status')=='rejected' for q in requests),'unfinished':sum(q.get('status') not in ('completed','failed','rejected') for q in requests),'useful_visible_output_tokens':visible,'overall_output_tok_s':visible/duration,'overall_output_tok_s_per_card':visible/duration/cards,'user_decode_tok_s':describe(rates),'TTFT_seconds':describe(ttfts),'ITL_seconds':describe(gaps),'SLO':{'strict_user_decode_above_tok_s':minimum_user_decode,'TTFT_seconds':ttft_slo},'good_requests':good,'goodput_requires_natural_EOS_or_stop':True,'speed_only_good_requests_including_caps':speed_good,'speed_only_output_tok_s_per_card':speed_good_tokens/duration/cards,'request_goodput_s':good/duration,'SLO_attainment_all_arrivals':good/len(requests) if requests else None,'good_output_tok_s_per_card':good_tokens/duration/cards,'output_cap_requests':sum(q.get('stop_reason') in ('max_new_tokens','length') for q in requests),'quality_of_capped_responses_not_established':True,'no_shifted_first_token_clock':True,'pad_EOS_rejected_unfinished_not_hidden':True}

def selftest():
    assert extract_numeric('reason 4+3=7.\n#### 1,234.00')['strict']=='1234'
    assert extract_numeric('therefore \\boxed{-2.5}')['flexible']=='-2.5'
    assert extract_numeric('No final numeric answer')['flexible'] is None
    c={'id':'a','gold_numeric':'1,234'}
    assert score_case(c,'Different reasoning. #### 1234.0','EOS')['flexible_correct']
    a=[score_case(c,'#### 1234','EOS')];b=[score_case(c,'Other reasoning #### 1234','EOS')]
    assert paired_quality(a,b,repeats=50)['accuracy_difference']==0
    q=[{'arrival':0.,'status':'completed','stop_reason':'EOS','events':[{'time':1.,'visible':True,'kind':'committed'},{'time':1.05,'visible':True,'kind':'committed'},{'time':1.06,'visible':False}]},{'arrival':0.,'status':'rejected','events':[]},{'arrival':.5,'status':'queued','events':[]}]
    m=serving_metrics(q,0.,2.,8);assert m['good_requests']==1 and m['rejected']==1 and m['unfinished']==1 and m['useful_visible_output_tokens']==2
    return {'passed':True,'CPU_only':True,'scope':'Answer scoring / paired data / EOS and rejected/unfinished accounting; no native model or online service qualification'}
if __name__=='__main__':
    import json
    print(json.dumps(selftest()))
