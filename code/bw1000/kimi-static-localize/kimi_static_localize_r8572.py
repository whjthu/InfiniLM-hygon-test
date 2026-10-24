"""Zero-newhook already-loaded static diagnosis; no performance promotion."""
import os,json,pathlib,time,hashlib,re
from kimi_prefill512_scope_r8563 import ScopedProvider,MODE
from kimi_static_cut_capture_r8572 import replay_cut
from kimi_matched_trace_r8467 import digest

COARSE=['stage_input','decoder','output_attn_res','final_norm']
DETAIL=['entry','attn_res','attn_norm','attn_output','attn_added','ffn_res','ffn_norm','ffn_output','exit']+['residual_slot_'+str(i)for i in range(94)]+['mla_input','mla_q_a','mla_q_norm','mla_q_b','mla_kv_a','mla_kv_norm','mla_kv_b','mla_key','mla_value_padded','mla_attention','mla_gate_linear','mla_gate_sigmoid','mla_gated','mla_output']

def jobs(plan,policy,end):
    assert policy in ('O','N')and end in (1,33)
    prompts=plan['prompts'];teachers=plan['teacher_tokens'];pasts=[0]*4;result=[]
    groups=[[i]for i in range(4)]if policy=='O'else[list(range(4))]
    for ii in groups:
        j=dict(seq=len(result),kind='prefill',keys=ii,rows=[i+1 for i in ii],pasts=[0]*len(ii),inputs=[prompts[i]for i in ii])
        result.append(dict(job=j,job_SHA256=digest(j),frozen_actual_ids=[teachers[i][0]for i in ii]))
        for i in ii:pasts[i]=512
    for ordinal in range(1,end+1):
        j=dict(seq=len(result),kind='decode',keys=list(range(4)),rows=[1,2,3,4],pasts=list(pasts),inputs=[[teachers[i][ordinal-1]]for i in range(4)])
        result.append(dict(job=j,job_SHA256=digest(j),frozen_actual_ids=[teachers[i][ordinal]for i in range(4)]))
        pasts=[v+1 for v in pasts]
    return dict(target_request=0,end_ordinal0=end,jobs=result)

def coarse_index(case):
    """Read small completed journals only; never rawarrays while owners live."""
    out={}
    for v in case['records']:
        if v['packed_target']is None:continue
        j=v['job'];ordinal=0 if j['kind']=='prefill'else j['pasts'][j['keys'].index(0)]-511
        for stage,directory in enumerate(v['perstage_trace_dirs']):
            d=pathlib.Path(directory);journals=[p for p in d.iterdir()if re.fullmatch(r'stage[0-9]+-rank[0-9]+\.jsonl',p.name)]
            assert len(journals)==8,(directory,'expected8realboundaryjournals')
            for p in journals:
                for line in p.read_text().splitlines():
                    row=json.loads(line);assert row['stage']==stage and len(row['requests'])==1
                    req=row['requests'][0];assert row['packed_target_only']and req['request']==v['packed_target']
                    assert row['tokens']==(512 if ordinal==0 else 1)and row['hidden']==7168
                    key=(ordinal,row['layer'],row['point'],stage,row['rank']);assert key not in out
                    out[key]=dict(fnv1a64=req['fnv1a64'],nonfinite=req['nonfinite'],original_shape=row['original_shape'],original_strides=row['original_strides'],original_offsets=row['original_offsets'])
    return out

def choose_detail(A,B):
    a,b=coarse_index(A),coarse_index(B);assert a.keys()==b.keys()and a
    assert all(v['nonfinite']==0 for v in [*a.values(),*b.values()])
    changed=[dict(ordinal0=k[0],layer=k[1],point=k[2],stage=k[3],rank=k[4],O=a[k],N=b[k])for k in a if a[k]['fnv1a64']!=b[k]['fnv1a64']]
    changed.sort(key=lambda v:(v['ordinal0'],v['layer'],0 if v['point']=='stage_input'else 1,v['rank']))
    if not changed:return dict(status='NO_COARSE_DIFFERENCE',detail_layer=None,differences=[])
    first=changed[0]
    if first['point']!='decoder' or not 0<=first['layer']<93:
        return dict(status='UNKNOWN_CARRY_OR_FINAL_RESIDUAL_BOUNDARY',detail_layer=None,first=first,differences=changed)
    return dict(status='ONE_DETAIL_INTERVAL_SELECTED_NOT_CAUSE',detail_layer=first['layer'],first=first,differences=changed,
        caution='Hash difference locates first observed floating boundary, not first harmful operation or generation correctness.')

def run(g):
    r,node,tag,lib,ctx,control=g['r'],g['node'],g['tag'],g['lib'],g['ctx'],g['control']
    report=g['report'];assert (g['tp_nodes'],g['pp'])==(1,4)
    plan_path=r/'incoming/Kimi-static-localize-plan-r8572.json';plan=json.loads(plan_path.read_text())
    assert len(plan['prompts'])==4 and all(len(v)==512 for v in plan['prompts'])
    assert plan['teacher_source_handle']=='910375.618'and plan['target_request']==0
    root=r/'results'/('Kimi-static-localize-r8572-'+tag)
    if node==0:root.mkdir(exist_ok=False)
    control.collect('static-localize-root-ready',{})
    report.update(scope='Static teacherforced O512/N2048 history with sameM4 decode/physicalrows; zero newhooks; no TPS/Modelpromotion',generation_acceptance='FAILED_OBSERVED_618_NOT_RECLASSIFIED',diagnostic_cases=[],GPU_qualification=False)
    os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='1';control.collect('static-localize-serving1',{})
    def capture(label,policy,end,*,detail=None,ordinals=(0,1),deadline=None):
        started=time.perf_counter();g['check'](lib.modern_kimi_engine_reset_r2475(ctx));control.collect(label+':reset',{})
        assert all(free>5*1024**3 for free,total in g['memory']())
        os.environ[MODE]='0'if policy=='O'else'1';control.collect(label+':mode',dict(mode=os.environ[MODE]))
        report['phase']=label;g['save']()
        def gate():assert all(free>1024**3 for free,total in g['memory']())
        a=ScopedProvider(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],memory_gate=gate,destroy=lambda p:None,full_logit_reference=True)
        path=root/(label+'-node'+str(node)+'-actuals.json');records=[]
        def persist(v):
            records.append(v);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(records,indent=2));tmp.replace(path)
        try:
            replay_cut(a,jobs(plan,policy,end),stage=node,case=label,root=root,collect=control.collect,persist=persist,
                layers=list(range(94))if detail is None else[detail],points=COARSE if detail is None else DETAIL,
                hash_only=detail is None,detail_layer=detail,observe_kv=detail is not None,state_rows='all',trace_ordinals=ordinals,deadline=deadline)
        finally:a.close()
        case=dict(label=label,policy=policy,end_ordinal0=end,detail_layer=detail,records=records,wall_s=time.perf_counter()-started,
            all_jobs_completed=len(records)==len(jobs(plan,policy,end)['jobs']),actuals_path=str(path),actuals_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['diagnostic_cases'].append(case);g['save']();control.collect(label+':all-capture-complete',dict(wall_s=case['wall_s']))
        return case
    A=capture('coarse-O','O',1);B=capture('coarse-N','N',1)
    def decision(a,b,label):
        path=root/(label+'-selection.json');value=None
        if node==0:
            try:value=choose_detail(a,b)
            except BaseException as error:value=dict(status='UNKNOWN_COARSE_SCHEMA',detail_layer=None,error=repr(error))
            path.write_text(json.dumps(value,indent=2))
        h=control.collect(label+':decision',dict(SHA256=hashlib.sha256(path.read_bytes()).hexdigest()if node==0 else None))[0]['SHA256']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h
        value=json.loads(path.read_text());report.setdefault('localization_decisions',[]).append(value);g['save']();return value
    selected=decision(A,B,'initial')
    if selected['status']=='NO_COARSE_DIFFERENCE':
        A=capture('late-O','O',33,ordinals=(31,33));B=capture('late-N','N',33,ordinals=(31,33));selected=decision(A,B,'late')
    if selected.get('detail_layer')is not None:
        begin=time.perf_counter();end=1 if selected['first']['ordinal0']<=1 else 33
        ordinals=tuple(sorted({0,1,selected['first']['ordinal0']}))
        first_detail=capture('detail-O','O',end,detail=selected['detail_layer'],ordinals=ordinals,deadline=begin+480)
        if first_detail['all_jobs_completed']:capture('detail-N','N',end,detail=selected['detail_layer'],ordinals=ordinals,deadline=begin+480)
        elapsed=time.perf_counter()-begin;report['selected_detail_capture_wall_s']=elapsed
        report['detail_budget_exceeded']=elapsed>480
        detail_cases=[v for v in report['diagnostic_cases']if v['detail_layer']is not None]
        report['selected_detail_complete']=len(detail_cases)==2 and all(v['all_jobs_completed']for v in detail_cases)
        # Captures finish before any parsing; no further detail layer is allowed.
    if node==0:report['actual_snapshot_bytes']=sum(p.stat().st_size for p in root.rglob('*')if p.is_file())
    report.update(serving_protocol_complete=True,diagnosis_capture_complete=True,
        final_scope='Existing raw/hidden capture only; offline numeric causal analysis afterterminal; no finaltoken repair or performance qualification',
        fullhidden_limit='Full hidden rank0/6 only, no residual_slot full arrays. Lastvector+hash does not establish wholecarry equality.',snapshot_byte_upper=10*1024**3)
    g['save']()
