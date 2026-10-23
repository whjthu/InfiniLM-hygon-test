"""Independent CPU saved-Model audit; no Engine, weights, GPU or restart."""
import pathlib,json,hashlib,datetime,statistics,sys,subprocess,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sys.path.insert(0,str(r/'incoming'))
from glm_transient_control_r5692 import RESERVE,ALLOWANCE,FLOOR
from glm_continuous_feedback_r5466 import tail_ready_metrics
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
 launch=json.loads((r/'glm-transient-Model-r5694-launch.json').read_text());run=launch['run']
 mp=r/'incoming/GLM-transient-Model-source-manifest-r5694.json';manifest=json.loads(mp.read_text());mh=sha(mp)
 assert mh==launch['source_manifest_SHA256']
 for n,h in manifest['source_SHA256'].items():assert sha(r/'incoming'/n)==h,('current_source',n)
 for n,h in manifest['DSO_SHA256'].items():assert sha(r/n)==h,('current_DSO',n)
 gold=json.loads((r/'incoming/GLM-lowbatch-frozen-gold-r8219.json').read_text())
 def hostpath(p):return pathlib.Path(p.replace('/refactor/',str(r)+'/'))
 assert sha(hostpath(gold['source_report']))==gold['source_report_SHA256'] and sha(hostpath(gold['fixture']))==gold['fixture_SHA256']
 actual=[c for c in json.loads(hostpath(gold['source_report']).read_text())['multi_batch_cases']if c['phase']==gold['phase']];assert len(actual)==1 and actual[0]['generated_ids']==gold['generated_ids'] and actual[0]['stopped_at_EOS']==gold['stopped_at_EOS']
 assert [c['input_ids']for c in json.loads(hostpath(gold['fixture']).read_text())['cases']]==gold['input_ids']
 payload=json.loads((r/'control'/f'glm-transient-Model-r5694-control-{run}-ids.json').read_text());eos=set(payload['chat_eos_token_ids'])
 d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'owners':[],'source_manifest_SHA256':mh,'qualified':False,'GPU_operations':False,'production_peak_proven':False,'allowner_report_SHA256':{}}
 samephase={};producers=set();tail=None;rejected=False;chosen=None
 cfg=json.loads((pathlib.Path.home()/'hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json').read_text());bounds=[0,38,78]
 for stage in (0,1):
  for rank in range(8):
   p=r/'results'/f'glm-transient-Model-r5694-{run}-stage{stage}-rank{rank}.json';q=json.loads(p.read_text());d['allowner_report_SHA256'][p.name]=sha(p)
   assert q['completed']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']
   assert q['library_sha256']==manifest['model_SHA256']and q['native_library_sha256']==manifest['native_SHA256']and q['lowbatch_manifest_SHA256_r8219']==mh
   assert q['owner_affinity']['verified']and q['owner_affinity']['width']==4 and q['BLAS_policy']['failures']==0
   assert q['cache_after_destroy']['created']==q['cache_after_destroy']['destroyed']and q['cache_after_destroy']['active']==q['cache_after_destroy']['active_bytes']==0
   assert q['host_bank_after_destroy_r4504'][:3]==[bounds[stage+1]-bounds[stage]]*2+[0]
   assert q['transient_default_OFF_no_Scratch_alloc']and q['resident_final_OFF_empty']and q['planned_phase_accounting_completed']
   assert q['continuous_transport_closed']['pending_drained_on_emitter']and q['continuous_transport_closed']['all_cases_ended']
   producers.add(q['actual_producer']);rejected |= bool(q.get('resident_candidate_rejected')or q.get('normal_numeric_rejection'))
   for ep in ('before_select','after_select'):
    old=json.loads((r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{3 if stage else 0}-rank{rank}.json').read_text())
    assert [c['generated_ids']for c in q[ep+'_chat_cases']]==[c['generated_ids']for c in old[ep+'_chat_cases']]
    assert all(c['stopped_at_eos']for c in q[ep+'_chat_cases'])and q[ep+'_frozen_IDs_exact_global']
   if q.get('resident_candidate_quality_passed'):
    for arm in ('T','V','B'):
     records=q['candidate_'+arm+'_chat_cases'];assert [c['generated_ids']for c in records]==[c['generated_ids']for c in q['before_select_chat_cases']]
     assert all(c['stopped_at_eos']for c in records)and q['candidate_'+arm+'_frozen_IDs_exact_global']
   cases=q['multi_batch_cases'];skipped=q.get('skipped_rejected_candidate_phases',[]);planned=[c['label']for c in q['resident_plan']]
   assert len(cases)+len(skipped)==len(planned)in[26,27] and len(set([c['phase']for c in cases]+skipped))==len(planned)
   policy={a['phase']:a for a in q['transient_policy_phases']};after={a['phase']:a for a in q['transient_phase_after']}
   references={};qa_refs={};phase_summary=[];S=sum(t=='sparse'for t in cfg['mlp_layer_types'][bounds[stage]:bounds[stage+1]])
   full=sum(t=='full'for t in cfg['indexer_types'][bounds[stage]:bounds[stage+1]])
   for row in cases:
    name=row['phase'];events=row['events'];gen=row['generated_ids'];on=row['resident_on'];selected=row['selected_global_layer_count']
    assert row['batch']==16 and row['micro']==(16 if name.endswith('OFF-m16')else 8)and row['continuous_cohort_policy_r5467']==1
    assert row['official_performance_phase']==name.startswith('perf-')and row['decode_dedup_policy_r5386']==0
    pr=policy[name];assert pr['on']==on and selected==((bounds[stage+1]-bounds[stage])if on else 0)
    assert pr['free_after_admission']>=RESERVE+ALLOWANCE
    if on:assert pr['fresh_free']>=RESERVE+2*ALLOWANCE and pr['native']['GPU_logical_bytes']==41943040
    assert pr['vector_on']==row['vector_on']and pr['native']['vector_mode']==int(row['vector_on']);assert pr['native']['enabled']==on and not pr['native']['busy']and not pr['native']['poison']
    assert row['memory_free_after_phase']>=FLOOR and after[name]['free_after']>=FLOOR
    assert not after[name]['observed_is_peak']
    visited=[[]for _ in gen];prev=None;native_total=[0]*5
    for e in events:
     assert all(e['round']<len(gen[i])for i in e['rows'])
     for i in e['rows']:visited[i].append(e['round'])
     flat=sum(row['input_lengths'][i]for i in e['rows'])if e['round']==0 else len(e['rows']);assert e['flat_tokens']==flat
     b,a=e['native_before'],e['native_after'];delta=[y-x for x,y in zip(b,a)]
     assert e['native_policy']==1 and e['decode_scope']==0 and e['decode_dedup_policy']==0
     native_on=flat>=512;expect=[0]*5 if row['vector_on']and flat in[4,8]else[2*S,S if native_on else 0,S if native_on else 0,0 if native_on else 2*S,0]
     assert delta==expect and(prev is None or prev==b),('native_counter',stage,rank,name,e['round'],delta,expect)
     assert [y-x for x,y in zip(e['decode_native_before'],e['decode_native_after'])]==[0]*4
     prev=a;native_total=[x+y for x,y in zip(native_total,delta)]
     assert math.isfinite(e['native_seconds'])and e['native_seconds']>=0 and e['Torch_matmul_modes']['float32_precision']=='highest'
     assert not any(v for k,v in e['Torch_matmul_modes'].items()if k!='float32_precision')
    assert all(v==list(range(len(g)))for v,g in zip(visited,gen))
    assert native_total==row['native_operator_count_delta']and row['decode_native_count_delta']==[0]*4
    assert all(g and(g[-1]in eos or len(g)==row['max_output_tokens'])and all(v not in eos for v in g[:-1])for g in gen)
    assert row['stopped_at_EOS']==[g[-1]in eos for g in gen]and row['no_forward_after_EOS']and row['cross_geometry_full_IDs_passed']and row['same_micro_reference_passed']
    if not row['qa']:
     assert row['input_lengths']==list(map(len,gold['input_ids']))and gen==[g[:row['max_output_tokens']]for g in gold['generated_ids']]
     if row['max_output_tokens']==128:assert row['stopped_at_EOS']==gold['stopped_at_EOS']
    else:
     assert all(row['stopped_at_EOS']);case=next(c for c in payload['chat_cases']if name.startswith('QA-'+c['label']+'-'));key=case['label']
     if key in qa_refs:assert gen==qa_refs[key],('QA logical row',stage,rank,name)
     else:qa_refs[key]=gen
    if name in samephase:assert gen==samephase[name]
    else:samephase[name]=gen
    dc=sum(e['round']>0 for e in events);dr=sum(len(e['rows'])for e in events if e['round']>0);sr=sum(len(e['rows'])for e in events if e['round']>0 and len(e['rows'])==8)
    assert row['fastpath_counter_delta']==[full*dc,full*dc,(bounds[stage+1]-bounds[stage])*dr-selected*sr]
    eligible=[e for e in events if e['round']>0 and len(e['rows'])>=8 and len({row['input_lengths'][i]for i in e['rows']})==1 and row['input_lengths'][e['rows'][0]]+e['round']<=640]
    assert row['batch_attention_counter_delta']==[(bounds[stage+1]-bounds[stage]-selected)*len(eligible),(bounds[stage+1]-bounds[stage]-selected)*sum(len(e['rows'])for e in eligible)]
    before=pr['native'];post=after[name]['native'];L=bounds[stage+1]-bounds[stage]
    assert post['calls']-before['calls']==L*sum(e['round']>0 and len(e['rows'])==8 for e in events)*(1 if on else 0)
    assert post['Runtime_identity']==before['Runtime_identity']and not post['busy']and not post['poison']
    ex=[2*L*sum(len(e['rows'])for e in events),8192*L*sum(row['input_lengths'][i]+e['round']for e in events for i in e['rows']),L*sum(2*len(e['rows'])-(14 if on and e['round']>0 and len(e['rows'])==8 else 0)for e in events),8192*L*sum(e['flat_tokens']for e in events)]
    delta=[y-x for x,y in zip(before['host_counters'],post['host_counters'])];assert delta[:3]==[0,0,0]and delta[3:]==ex
    assert after[name]['host_counter_delta']==delta and after[name]['expected_host_counter_delta_tail']==ex
    vh=[e for e in events if row['vector_on']and e['flat_tokens']in[4,8]];hits=S*len(vh);fallback=S*(len(events)-len(vh));tokens=S*sum(e['flat_tokens']for e in vh);ve=[hits,fallback,hits,fallback,0,tokens,tokens,0]
    assert [y-x for x,y in zip(before['vector_counts'],post['vector_counts'])]==ve and after[name]['vector_counter_delta']==ve
    primary=None
    if stage==1 and rank==0:
     primary=tail_ready_metrics(gen,row['tail_ready_times'],eos,row['elapsed_seconds'],16,row['max_output_tokens']);assert primary==row['tail_ready_metrics']
    phase_summary.append({'phase':name,'ON':on,'vector_on':row['vector_on'],'arm':row['arm'],'micro':row['micro'],'selected_layers':selected,'primary_tail_ready_metrics':primary,'TTFT_mean_seconds':statistics.mean(primary['TTFT_seconds_per_request'])if primary else None,'prefill_CAPI_seconds_sum_not_GPUbusy':sum(e['native_seconds']for e in events if e['round']==0),'memory_free_after':row['memory_free_after_phase']})
   d['owners'].append({'stage':stage,'rank':rank,'normal_closed':True,'rejected':bool(q.get('normal_numeric_rejection')),'phases':phase_summary,'skipped':skipped,'minimum_observed_free_not_peak':min((a['free_after']for a in q['transient_phase_after']),default=None)})
   if stage==1 and rank==0:tail=q
 assert len(producers)==1;producer=next(iter(producers));d['producer']=producer
 job=[v.split('|')for v in subprocess.check_output(['sacct','-j',producer,'-n','-P','-o','JobID,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=25).splitlines()]
 assert len(job)==1 and job[0][0]==producer;d['Slurm']=job
 # Exactly parent+8 spawned workers per node. Each resolves privateModel/new CAPI
 # and actual native5386 symbols before any Enginecreate or tensor weight IO.
 pre=[json.loads(p.read_text())for p in sorted((r/'results').glob('glm-transient-preengine-r5694-'+producer+'-*.json'))]
 assert len(pre)==18 and sorted(__import__('collections').Counter(p['hostname']for p in pre).values())==[9,9]
 assert all(p['completed']and p['library_sha256']==manifest['model_SHA256']and p['visible_GPU0_no_Enginecreate_no_weight_tensor_IO']and len(p['actual_symbol_owners'])==6 for p in pre)
 d['preengine_records']=18;d['raw_CPUargmax']={'snapshots':0,'rows':0,'bytes':0,'skipped_inprogress':[]}
 cases={c['phase']:c for c in tail['multi_batch_cases']}
 if 'last_completed_generation'in tail:cases[tail['last_completed_generation']['phase']]=tail['last_completed_generation']
 for e in tail['raw_checkpoints']:
  if e['phase']not in cases:d['raw_CPUargmax']['skipped_inprogress'].append(e['phase']);continue
  raw=(r/'results'/e['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==e['sha256']and len(raw)==e['shape'][0]*e['shape'][1]*2
  b=np.frombuffer(raw,dtype='<u2').reshape(e['shape']);f=(b.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all()
  assert f.argmax(axis=1).tolist()==e['GPU_argmax']==[cases[e['phase']]['generated_ids'][i][e['step']]for i in e['rows']]
  d['raw_CPUargmax']['snapshots']+=1;d['raw_CPUargmax']['rows']+=len(e['rows']);d['raw_CPUargmax']['bytes']+=len(raw);del raw,b,f
 perf=[c for c in d['owners'][8]['phases']if c['phase'].startswith('perf-')]
 d['candidate_rejected']=rejected;d['selected_counts']=chosen;d['ABBA']={}
 if not rejected:
  assert len(perf)==8 and [c['arm']for c in perf]==['O','T','V','B','B','V','T','O']
  for arm in ('O','T','V','B'):
   values=[c for c in perf if c['arm']==arm];assert len(values)==2
   d['ABBA'][arm]={k:statistics.mean(c['primary_tail_ready_metrics'][k]for c in values)for k in ['aggregate_useful_decode_tokens_s','decode_tokens_s_per_card','including_prefill_useful_tokens_s','including_prefill_tokens_s_per_card','mean_request_useful_decode_tokens_s']};d['ABBA'][arm]['raw_arms']=values
 d['qualified']=job[0][1:3]==['COMPLETED','0:0']and not rejected and len(perf)==8 and not d['raw_CPUargmax']['skipped_inprogress']
 d['scope']='CPU independent wholeModel IDs/EOS/same logicalrow QA/frozen448/no forward-after-EOS/actual native and transient hostbank/dispatch counters/Runtimeidentity/18preengine/normalclose/Slurm0/raw finiteargmax. PostKFD and network logs are separate terminal watcher gates. Sampled memory not productionpeak; CAPI includes sync/PPwait and is not GPUbusy.'
 return d
if __name__=='__main__':print(json.dumps(audit()))
