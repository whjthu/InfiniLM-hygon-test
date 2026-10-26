"""Prespecified pilot/heldout gate. CPU summary only, no threshold tuning."""
from collections import defaultdict
def paired_gate(baseline,candidate):
    a={x['id']:x for x in baseline};b={x['id']:x for x in candidate}
    if not a or set(a)!=set(b)or len(a)!=len(baseline)or len(b)!=len(candidate):raise ValueError('complete unique matched pairs required')
    groups=defaultdict(list)
    for key in a:
        if a[key]['domain']!=b[key]['domain']:raise ValueError('domain mismatch')
        groups[a[key]['domain']].append(key)
    def metrics(ids):
        # Incomplete/capped responses remain denominator; correctness alone does not give natural-EOS goodput.
        return {'n':len(ids),'strict_accuracy':sum(bool(a[i].get('correct',False))and a[i].get('status')=='completed'for i in ids)/len(ids),
                'candidate_accuracy':sum(bool(b[i].get('correct',False))and b[i].get('status')=='completed'for i in ids)/len(ids)}
    total=metrics(list(a));domains={k:metrics(v)for k,v in groups.items()}
    def loss(v):return v['strict_accuracy']-v['candidate_accuracy']
    reasons=[]
    if loss(total)>.02+1e-12:reasons.append('overall_accuracy_drop_gt2pp')
    for key,v in domains.items():
        if loss(v)>.05+1e-12:reasons.append('domain_accuracy_drop_gt5pp:'+key)
    for v in [total,*domains.values()]:v['accuracy_drop_pp']=100*loss(v)
    return {'stop':bool(reasons),'reasons':reasons,'overall':total,'domains':domains,'noninferiority_established':False,'no_threshold_search':True}
