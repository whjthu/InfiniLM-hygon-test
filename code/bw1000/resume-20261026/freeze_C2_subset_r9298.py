from pathlib import Path
import json,hashlib,re,statistics,sys
r=Path.cwd();p=r/sys.argv[1];q=r/sys.argv[2];assert not q.exists()
d=json.loads(p.read_bytes()); assert d['status']['manifest_SHA256']=='6276a2e373629c0433be79e5f80aefa3163dc7a681aa84d998320c35506390b6'
groups={}
for c in d['cases']:
    if c['category']!='main-C2':continue
    m=re.fullmatch(r'main-C2-g(\d+)-([01])',c['label']);assert m
    g=int(m[1]); arm=bool(int(m[2]));assert arm==c['W2pair_enabled']
    assert arm not in groups.setdefault(g,{})
    groups[g][arm]=c
paired=sorted(g for g,v in groups.items() if len(v)==2)
assert paired==list(range(len(paired))), 'Require complete chronological prefix, not selected best groups'
assert all([u['quality']['id']for u in groups[g][False]['users']]==[u['quality']['id']for u in groups[g][True]['users']] for g in paired)
arms=[]
for arm in [False,True]:
    cs=[groups[g][arm]for g in paired];us=[u for c in cs for u in c['users']]
    assert len(set(u['quality']['id']for u in us))==len(us)
    n=sum(c['total_useful_decode_tokens']for c in cs);wall=sum(c['decode_common_window_s']for c in cs)
    rates=[u['user_firstlast_tok_s']for u in us]
    arms.append({'W2pair':arm,'users':len(us),'decode_tokens':n,'decode_common_wall_s':wall,
        'pooled_decode_tok_s':n/wall,'pooled_decode_tok_s_per_all33cards':n/wall/33,
        'user_firstlast_min_median_max':[min(rates),statistics.median(rates),max(rates)],
        'user_GT15':sum(x>15 for x in rates),'correct':sum(u['quality']['correct']for u in us),
        'EOS':sum(u['EOS']for u in us),'cap':sum(u['truncated']for u in us),
        'task_output_tokens':sum(u['useful_tokens']for u in us),
        'task_common_wall_s':sum(c['end_native_monotonic']-c['admission_native_monotonic']for c in cs),
        'case_labels':[c['label']for c in cs]})
out={'schema':1,'status':'C2_SUBSET_COMPLETE_WHOLE_MODEL_PROTOCOL_STILL_PENDING'if len(paired)==16 else'LIVE_PROVISIONAL_C2_CHRONOLOGICAL_PAIRED_PREFIX',
    'source_snapshot':p.name,'source_snapshot_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'observed_UTC':d['UTC'],
    'actual_handles':['910379.474','910371.586','910406.277'],'manifest_SHA256':d['status']['manifest_SHA256'],
    'paired_groups':paired,'unique_questions':2*len(paired),'unpaired_completed_groups':sorted(g for g,v in groups.items()if len(v)!=2),
    'arms':arms,'pooled_decode_rate_gain_percent':100*(arms[1]['pooled_decode_tok_s']/arms[0]['pooled_decode_tok_s']-1),
    'memory':d['memory'],'lifecycle_last_observed':d['sacct'],'screen_gate':d['status'].get('screen_gate'),
    'limits':['Not HTTP; native emitter clock includes thinking/channel/control tokens; pad/EOS excluded.',
        'Per-user first-last includes later requests prefill after ownfirst; not shifted to common decode start.',
        'C2 subset completion does not prove whole50case Model qualification or normal cleanup.',
        'Same predefined32GSM questions reused from prior run; not new independent holdout.',
        'All serving33cards counted. No aggregate addition across node clocks.']}
q.write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({k:out[k]for k in ['status','unique_questions','arms','pooled_decode_rate_gain_percent']},ensure_ascii=False))
