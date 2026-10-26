from pathlib import Path
import json,hashlib,statistics,datetime
r=Path.cwd();p=r/'Kimi474-terminal-safe-root-r9305.json';d=json.loads(p.read_bytes());a=d['audit']
assert a['passed']and a['initial50_complete']and a['normal_target_owners']==32 and d['coordinator']['passed']and d['coordinator']['post_KFD_passed']
assert len(d['cases'])==50 and all(x[1:3]==['COMPLETED','0:0']for x in d['Slurm'])
groups={}
for name,prefix in [('GSM32_C2','main-C2-'),('GSM32_C8','secondary-C8-'),('MMLU31_C8','MMLU31-C8-')]:
    arms=[]
    for arm in [False,True]:
        cs=[c for c in d['cases']if c['label'].startswith(prefix)and c['W2pair_enabled']==arm]
        us=[u for c in cs for u in c['users']];ids=[u['quality']['id']for u in us];assert len(set(ids))==len(ids)
        n=sum(c['total_useful_decode_tokens']for c in cs);wall=sum(c['decode_common_window_s']for c in cs)
        taskwall=sum(c['end_native_monotonic']-c['admission_native_monotonic']for c in cs)
        pf=sum(c['prefill_all_complete_s']for c in cs);ints=sum(u['input_tokens']for u in us)
        rates=[u['user_firstlast_tok_s']for u in us];ce=[u for u in us if u['quality']['correct']and u['EOS']]
        arms.append({'W2pair':arm,'requests':len(us),'question_ids':ids,'cases':len(cs),'correct':sum(u['quality']['correct']for u in us),
            'natural_EOS':sum(u['EOS']for u in us),'caps':sum(u['truncated']for u in us),'correct_AND_natural_EOS':len(ce),
            'decode_tokens':n,'decode_common_wall_s':wall,'decode_tok_s':n/wall,'decode_tok_s_per_all33cards':n/wall/33,
            'user_firstlast_min_median_max':[min(rates),statistics.median(rates),max(rates)],'user_GT15':sum(x>15 for x in rates),
            'correct_naturalEOS_user_GT15':sum(u['user_firstlast_tok_s']>15 for u in ce),
            'task_common_wall_s':taskwall,'task_useful_output_tokens':sum(u['useful_tokens']for u in us),
            'task_output_tok_s':sum(u['useful_tokens']for u in us)/taskwall,'correct_naturalEOS_tasks_per_task_wall_s':len(ce)/taskwall,
            'prefill_all_complete_s':pf,'input_tokens':ints,'prefill_including_feature_guard_input_tok_s':ints/pf,
            'native_TTFT_median_s':statistics.median(u['native_TTFT_s']for u in us),
            'median_of_peruser_ITL_P99_s':statistics.median(u['ITL_P99_s']for u in us)})
    assert arms[0]['question_ids']==arms[1]['question_ids'];groups[name]={'arms':arms,'decode_rate_gain_percent':100*(arms[1]['decode_tok_s']/arms[0]['decode_tok_s']-1)}
o={'schema':1,'status':'ALL50_COMPLETE_LIFECYCLE_QUALIFIED_QA_CAP_LIMITS_RETAINED','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'hardware':'BW1000','all_serving_cards':33,'TP':16,'PP':2,'logical_capacity':8,'width':8,'prefill_micro':1,'ctx':1024,'output_cap':512,
    'native_SHA256':'3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd','manifest_SHA256':d['coordinator']['manifest_SHA256'],
    'source_snapshot':p.name,'source_snapshot_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'Slurm':d['Slurm'],
    'lifecycle':{'normal_target_owners':32,'draft_normal_terminal':d['coordinator']['draft_normal_terminal'],'audit_passed':True,'post_KFD_empty':True,'network':'IBext_v8/noSocketData onall4targetlogs verifiedbyoriginalaudit','GDR':'UNKNOWN'},
    'runtime_scalar':d['runtime_scalar'],'global_min_sampled_free_bytes':min(x['minimum_free_bytes']for x in d['case_minimum_free_bytes']if x['minimum_free_bytes']is not None),
    'groups':groups,'limits':['GSM C2/C8 reuse same32questions; MMLU31 is31distinct withinfamily; allrepeats do not expand independentN.',
        'Two predeclared short warm cases excluded from quality/performance, not cherrypicked.',
        'Decode useful nonEOS includes thinking/channel/control tokens atnativeemitter, notHTTPvisibleanswer. Pad notcounted.',
        'Peruser first-last includes otherrequests prefills. Native taskclock starts afterreset/draftbegin, excludesexternaladmission/networkqueue.',
        'Prefill rate includes per-micro feature harvest/guards, notpureGPUkernel. Median ofperuserP99 isnotglobalpooledITLP99.',
        'MMLU caps+wronganswers retained. All50 lifecyclequalification does notprove usable unrestricted benchmarkquality orperuser15SLA.',
        'Whole50family counts onceasnewModelwave; CPUaudit itself is0extra waves. No physicalpeak inferred.']}
q=r/'Kimi474-terminal-canonical-root-r9307.json';assert not q.exists();q.write_bytes((json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({'status':o['status'],'min_free':o['global_min_sampled_free_bytes'],'groups':{n:{'gain':g['decode_rate_gain_percent'],'arms':[{k:a[k]for k in ['W2pair','requests','correct','natural_EOS','caps','correct_AND_natural_EOS','decode_tok_s','user_firstlast_min_median_max','user_GT15','task_output_tok_s']}for a in g['arms']]}for n,g in groups.items()}},ensure_ascii=False))
