"""Independent saved-result CPU auditor. No GPU, no weight tensors or restart."""
import pathlib,json,hashlib,statistics,sys,subprocess,math,collections
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sys.path.insert(0,str(r/'incoming'))
from glm_crossrequest_environment_r8864 import network_log_gate
from glm_crossrequest_contract_r8864 import validate_reference,validate_fixture,I512_SHA256
from glm_continuous_feedback_r8864 import tail_ready_metrics
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit_replica(replica):
 launch=json.loads((r/'glm-transient-Model-r8864-launch.json').read_text());common_run=launch['run'];run=common_run+'-'+replica;nodes=['a02r3n03','a02r3n07']if replica=='A'else['a02r3n15','a02r3n16'];node_offset=0 if replica=='A'else 2
 mp=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8864.json';m=json.loads(mp.read_text());mh=sha(mp);assert mh==launch['source_manifest_SHA256']
 for n,h in m['source_SHA256'].items():assert sha(r/'incoming'/n)==h,n
 for n,h in m['DSO_SHA256'].items():assert sha(r/n)==h,n
 gold=json.loads((r/'incoming/GLM-lowbatch-frozen-gold-r8219.json').read_text())
 fp=r/'results/glm-I512x32-real-inputs-r5732.json';assert sha(fp)==I512_SHA256;i512=validate_fixture(json.loads(fp.read_text()))
 def hostpath(p):return pathlib.Path(p.replace('/refactor/',str(r)+'/'))
 assert sha(hostpath(gold['source_report']))==gold['source_report_SHA256']and sha(hostpath(gold['fixture']))==gold['fixture_SHA256']
 oldcase=[x for x in json.loads(hostpath(gold['source_report']).read_text())['multi_batch_cases']if x['phase']==gold['phase']];assert len(oldcase)==1 and oldcase[0]['generated_ids']==gold['generated_ids']and oldcase[0]['stopped_at_EOS']==gold['stopped_at_EOS']
 payload=json.loads((r/'control'/f'glm-transient-Model-r8864-control-{run}-ids.json').read_text());assert payload['I512_input_ids']==i512
 eos=set(payload['chat_eos_token_ids']);bounds=[0,38,78];cfg=json.loads((pathlib.Path.home()/'hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json').read_text())
 historical_i512=json.loads((r/'incoming/GLM477-frozen-gold-r8844.json').read_text());assert sha(r/'incoming/GLM477-frozen-gold-r8844.json')=='ff305cd7c4aa8b414ef1a150bb872422fafed2c02eafb56ba53b4798b4a76176'
 assert sha(hostpath(historical_i512['source_report']))==historical_i512['source_report_SHA256']
 global_rows=m['DP_global_rows'][replica];i512=[i512[i]for i in global_rows]
 historical_i512={**historical_i512,**{k:[historical_i512[k][i]for i in global_rows]for k in ['input_ids','generated_ids','stopped_at_EOS']}}
 d={'run':run,'qualified':False,'GPU_operations':False,'allowner_report_SHA256':{},'owners':[],'independent_original_mathematics_proof':False,'H2D_exclusive_wall_seconds':None,'H2D_wall_scope':'No existing wholeModel copy timer; exact bytes plus paired Model wall/TLS only, not copy fraction.'}
 producers=set();samephase={};rejected=False;tail=None;chosen=None
 for stage in (0,1):
  L=bounds[stage+1]-bounds[stage];S=sum(t=='sparse'for t in cfg['mlp_layer_types'][bounds[stage]:bounds[stage+1]]);F=sum(t=='full'for t in cfg['indexer_types'][bounds[stage]:bounds[stage+1]])
  for rank in range(8):
   path=r/'results'/f'glm-transient-Model-r8864-{run}-stage{stage}-rank{rank}.json';q=json.loads(path.read_text());d['allowner_report_SHA256'][path.name]=sha(path)
   hw=q['hardware_gate_before_create'];assert hw['gcnArchName'].split(':')[0]=='gfx936'and hw['compute_units']==80 and hw['total_memory']>=63*1024**3 and hw['tiny_F32_sum']==120.0 and hw['before_Model_create_and_weights']
   assert q['DP_replica']==replica and q['DP_global_rows']==global_rows
   assert q['completed']and q['worker_shutdown_completed']and q['post_close_Torch_liveness']and q['cache_rows']==32 and q['cache_capacity']==640
   assert q['library_sha256']==m['model_SHA256']and q['native_library_sha256']==m['native_SHA256']and q['lowbatch_manifest_SHA256_r8219']==mh
   assert q['shared_validation_final_OFF']
   if not(q.get('normal_numeric_rejection')or q.get('resident_candidate_rejected')):assert q['after_select_shared_flag']==1
   assert q['owner_affinity']['verified']and q['BLAS_policy']['failures']==0 and q['resident_final_OFF_empty']and q['planned_phase_accounting_completed']
   assert q['cache_after_destroy']['created']==q['cache_after_destroy']['destroyed']and q['cache_after_destroy']['active']==q['cache_after_destroy']['active_bytes']==0
   assert q['host_bank_after_create_r4504']==[L,0,2*32*640*8*256*2*L,0,0,0,0]and q['host_bank_after_destroy_r4504'][:3]==[L,L,0]
   assert q['transient_fixed_OFF_no_Scratch_alloc']and q['vector_fixed_ON_in_all_paired_arms']
   assert q['continuous_transport_closed']=={'all_cases_ended':True,'pending_drained_on_emitter':True}
   producers.add(q['actual_producer']);rejected|=bool(q.get('normal_numeric_rejection')or q.get('resident_candidate_rejected'))
   for ep in ['before_select','after_select']:
    if ep=='after_select'and q.get('resident_candidate_rejected'):
     assert q['after_select_QA_scope']=='SKIPPED_REJECTED_CANDIDATE'and q['after_select_shared_flag']==0
     continue
    old=json.loads((r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{3 if stage else 0}-rank{rank}.json').read_text())
    assert q[ep+'_frozen_IDs_exact_global']and [x['generated_ids']for x in q[ep+'_chat_cases']]==[x['generated_ids']for x in old[ep+'_chat_cases']]
   cases=q['multi_batch_cases'];skipped=q.get('skipped_rejected_candidate_phases',[]);plan=[x['label']for x in q['resident_plan']]
   assert len(cases)+len(skipped)==len(plan)and len(plan)==2 and len(set([x['phase']for x in cases]+skipped))==len(plan)
   phase_after=q.get('resident_phase_after')
   if phase_after is None:
    assert q.get('resident_candidate_rejected')and not cases and(q.get('resident_setter_rejection')or q.get('resident_admission_rejection')),'missingphaseafterwithoutknownpre-forwardrejection'
    phase_after=[]
   policy={x['phase']:x for x in q['resident_policy_phases']};after={x['phase']:x for x in phase_after};ledger={};refs={};phase_summary=[]
   for row in cases:
    name=row['phase'];events=row['events'];gen=row['generated_ids'];pm=row['prefill_micro'];assert pm in(4,8)and row['prefill_factor_arm']==('O8'if pm==8 else'N4');selected=row['selected_global_layer_count'];on=row['resident_on'];pr=policy[name];post=after[name];before=pr['native'];native=post['native']
    assert row['batch']==16 and row['micro']==8 and row['continuous_cohort_policy_r5467']==1 and row['vector_on']and row['official_performance_phase']==name.startswith('perf-')
    assert row['shared_validation_enabled']and on and row['arm']=='N'and selected==sum(before['Context_mask'])and selected<=L
    assert pr['physical_budget_bytes']==selected*m['resident_physical_bytes_per_layer'] and pr['logical_bank_bytes']==selected*160*1024**2
    assert pr['fresh_free_before']-pr['physical_budget_bytes']>=6*1024**3+64*1024**2 and pr['free_after_admission']>=6*1024**3+64*1024**2
    if on:
     if chosen is None:chosen=pr['selected_counts']
     assert pr['selected_counts']==chosen and selected==chosen[stage]
    assert before['resident_physical_bytes_per_layer']==native['resident_physical_bytes_per_layer']==m['resident_physical_bytes_per_layer']
    assert before['vector_mode']==native['vector_mode']==1 and before['transient']==native['transient']and not native['transient']['enabled']and native['transient']['GPU_logical_bytes']==0
    assert native['Runtime_identity']==before['Runtime_identity']and row['memory_free_after_phase']>=512*1024**2 and post['free_after']>=512*1024**2
    assert not post['observed_is_peak'];visited=[[]for _ in gen];native_sum=[0]*5
    for e in events:
     flat=sum(row['input_lengths'][i]for i in e['rows'])if e['round']==0 else len(e['rows']);assert flat==e['flat_tokens']
     for i in e['rows']:assert e['round']<len(gen[i]);visited[i].append(e['round'])
     assert e['native_policy']==1 and e['decode_scope']==0 and e['decode_dedup_policy']==0
     delta=[b-a for a,b in zip(e['native_before'],e['native_after'])]
     expected=[0]*5 if flat in(4,8)else[2*S,S if flat>=512 else 0,S if flat>=512 else 0,0 if flat>=512 else 2*S,0]
     assert delta==expected;native_sum=[a+b for a,b in zip(native_sum,delta)]
     assert [b-a for a,b in zip(e['decode_native_before'],e['decode_native_after'])]==[0]*4
     assert math.isfinite(e['native_seconds'])and e['native_seconds']>=0 and e['Torch_matmul_modes']['float32_precision']=='highest'
     assert not any(v for k,v in e['Torch_matmul_modes'].items()if k!='float32_precision')
    assert visited==[list(range(len(g)))for g in gen]and native_sum==row['native_operator_count_delta']
    pre=[e for e in events if e['round']==0];assert [e['rows']for e in pre]==[list(range(i,min(i+pm,row['batch'])))for i in range(0,row['batch'],pm)]
    assert all(e['flat_tokens']<=4096 for e in pre)
    assert row['no_forward_after_EOS']and row['cross_geometry_full_IDs_passed']and row['same_micro_reference_passed']
    assert row['stopped_at_EOS']==[g[-1]in eos for g in gen]and all(g and(g[-1]in eos or len(g)==row['max_output_tokens'])and not any(t in eos for t in g[:-1])for g in gen)
    kind=row['reference_kind'];prompts=i512[:row['batch']] if kind=='I512'else gold['input_ids']if kind=='short'else[next(c for c in payload['chat_cases']if name.startswith('QA-'+c['label']+'-'))['input_ids']]*16
    assert row['input_lengths']==list(map(len,prompts))
    ok,detail=validate_reference(kind,'R'if row['crossrequest_prefill_enabled']else'O',prompts,gen,row['stopped_at_EOS'],row['max_output_tokens'],gold,i512,ledger,historical_i512);assert ok,detail
    key=(kind,tuple(map(tuple,prompts)),row['max_output_tokens'])
    if key in refs:assert refs[key]==gen
    else:refs[key]=gen
    if kind=='qa':
     assert all(row['stopped_at_EOS'])
     if stage==1 and rank==0:assert len(row['answers'])==16 and all(x['known_answer']and x['thinking_closed']for x in row['answers'])
    if name in samephase:assert samephase[name]==gen
    else:samephase[name]=gen
    dc=sum(e['round']>0 for e in events);dr=sum(len(e['rows'])for e in events if e['round']>0);sr=sum(len(e['rows'])for e in events if e['round']>0 and len(e['rows'])==8)
    ce=(1 if stage==1 else 0)*sum(e['round']==0 and len(e['rows'])==8 and e['flat_tokens']==4096 and all(row['input_lengths'][i]==512 for i in e['rows'])for e in events)if row['crossrequest_prefill_enabled']else 0
    assert row['crossrequest_call_delta']==row['crossrequest_call_expected']==ce
    assert ce==0
    shared=L-F;on_shared=row['shared_validation_enabled'];skip=shared*dr if on_shared else 0
    assert row['shared_validation_counter_delta']==row['shared_validation_expected']==[skip,shared*sum(len(e['rows'])for e in events)-skip]
    exactbytes=shared*8*sum(row['input_lengths'][i]+e['round']for e in events if e['round']>0 for i in e['rows'])if on_shared else 0
    assert row['shared_validation_source_exact_avoided_D2H_bytes']==exactbytes
    assert row['fastpath_counter_delta']==[F*dc,F*dc,L*dr-selected*sr]
    eligible=[e for e in events if e['round']>0 and len(e['rows'])>=8 and len({row['input_lengths'][i]for i in e['rows']})==1 and row['input_lengths'][e['rows'][0]]+e['round']<=640]
    assert row['batch_attention_counter_delta']==[(L-selected)*len(eligible),(L-selected)*sum(len(e['rows'])for e in eligible)]
    hc=[b-a for a,b in zip(before['host_counters'],native['host_counters'])];host=L-selected
    he=[0,0,0,2*host*sum(len(e['rows'])for e in events),8192*host*sum(row['input_lengths'][i]+e['round']for e in events for i in e['rows']),2*host*sum(len(e['rows'])for e in events),8192*host*sum(e['flat_tokens']for e in events)]
    assert hc==he==post['host_counter_delta']
    for a,b in zip(before['layers'],native['layers']):
     scale=int(a['enabled']);assert b['enabled']==a['enabled']
     assert b['append_D2D_bytes']-a['append_D2D_bytes']==scale*8192*sum(e['flat_tokens']for e in events)
     assert b['exact_shared_calls']-a['exact_shared_calls']==scale*sum(e['round']>0 and len(e['rows'])==8 for e in events)
     assert b['resident_original601_row_calls']-a['resident_original601_row_calls']==scale*sum(len(e['rows'])for e in events if e['round']>0 and len(e['rows'])<8)
     assert b['packed_prefix_bytes']==0
    hit=[e for e in events if e['flat_tokens']in(4,8)];n=S*len(hit);fallback=S*(len(events)-len(hit));tr=S*sum(e['flat_tokens']for e in hit)
    assert [b-a for a,b in zip(before['vector_counts'],native['vector_counts'])]==[n,fallback,n,fallback,0,tr,tr,0]
    primary=None
    if stage==1 and rank==0:
     primary=tail_ready_metrics(gen,row['tail_ready_times'],eos,row['elapsed_seconds'],16,row['max_output_tokens']);assert primary==row['tail_ready_metrics']
     assert row['tail_prefill_CAPI_return_seconds']>0 and math.isclose(row['prefill_input_tokens_per_service_second'],sum(row['input_lengths'])/row['tail_prefill_CAPI_return_seconds'])
    phase_summary.append({'phase':name,'kind':kind,'shared_validation_enabled':row['shared_validation_enabled'],'shared_validation_counter_delta':row['shared_validation_counter_delta'],'avoided_selection_D2H_bytes':row['shared_validation_source_exact_avoided_D2H_bytes'],'arm':row['prefill_factor_arm'],'prefill_micro':pm,'selected':selected,'primary':primary,'prefill_input_tokens_per_service_second':row['prefill_input_tokens_per_service_second'],'host_H2D_bytes':hc[4],'host_D2H_bytes':hc[6],'free_after_not_peak':row['memory_free_after_phase']})
   d['owners'].append({'stage':stage,'rank':rank,'normal_closed':True,'pinned_host_bytes':q['host_bank_after_create_r4504'][2],'load_wall_seconds':q['actual_owner_weight_load_wall_seconds'],'CPU_memory_before_load':q['CPU_memory_before_weight_load'],'CPU_memory_after_load':q['CPU_memory_after_weight_load'],'phases':phase_summary})
   samples=q.get('native_pipeline_profile_samples_r8219',[])
   for sample in samples:
    assert sample['phase'].startswith('prof-')and sample['round']in[0,1,2,63,127]and sample['stage']==stage and sample['rank']==rank
    case=next(x for x in cases if x['phase']==sample['phase']);event=next(e for e in case['events']if e['round']==sample['round']and e['rows']==sample['rows'])
    assert sample['flat_tokens']==event['flat_tokens']and len(sample['values_seconds'])==6 and all(math.isfinite(v)and v>=0 for v in sample['values_seconds'])
   d['owners'][-1]['profile_samples']=samples
   if stage==1 and rank==0:tail=q
 assert len(producers)==1;producer=next(iter(producers));d['producer']=producer
 jobs=[s.split('|')for s in subprocess.check_output(['sacct','-j',producer,'-n','-P','-o','JobID,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=25).splitlines()];assert len(jobs)==1 and jobs[0][0]==producer;d['Slurm']=jobs
 pre=[json.loads(p.read_text())for p in(r/'results').glob('glm-transient-preengine-r8864-'+producer+'-*.json')]
 pre=[x for x in pre if x['hostname']in nodes]
 assert len(pre)==18 and sorted(collections.Counter(x['hostname']for x in pre).values())==[9,9]and all(x['completed']and x['library_sha256']==m['model_SHA256']for x in pre)
 d['network_logs']=[]
 for node in(0,1):
  lp=r/'logs'/f'glm-transient-Model-r8864-{producer}-{node+node_offset}.log';gate=network_log_gate(lp.read_text(errors='replace'));assert gate['passed'],(str(lp),gate);d['network_logs'].append(gate)
 d['preengine_records']=18;d['raw_CPUargmax']={'snapshots':0,'rows':0,'bytes':0}
 cases={x['phase']:x for x in tail['multi_batch_cases']+tail.get('layer77_diagnostic_cases',[])};cases.update(tail.get('rejected_generation_records',{}))
 if 'last_completed_generation'in tail:cases[tail['last_completed_generation']['phase']]=tail['last_completed_generation']
 for entry in tail['raw_checkpoints']:
  assert entry['phase']in cases;raw=(r/'results'/entry['file']).read_bytes();assert sha(r/'results'/entry['file'])==entry['sha256']and len(raw)==entry['shape'][0]*entry['shape'][1]*2
  a=np.frombuffer(raw,dtype='<u2').reshape(entry['shape']);f=(a.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all()and f.argmax(axis=1).tolist()==entry['GPU_argmax']==[cases[entry['phase']]['generated_ids'][i][entry['step']]for i in entry['rows']]
  d['raw_CPUargmax']['snapshots']+=1;d['raw_CPUargmax']['rows']+=len(entry['rows']);d['raw_CPUargmax']['bytes']+=len(raw)
 d['ABBA']={};d['candidate_rejected']=rejected;d['selected_counts']=chosen
 tailcases=[c for c in tail['multi_batch_cases']if c['phase'].startswith('perf-I512-')]
 d['tail_ordinary_cases']=tailcases
 for stage in (0,1):
  for rank in range(8):
   owner=json.loads((r/'results'/f'glm-transient-Model-r8864-{run}-stage{stage}-rank{rank}.json').read_text());assert not owner.get('native_pipeline_profile_samples_r8219')
   for c in owner['multi_batch_cases']:
    on=int(c['reference_kind']=='I512'and '-N-'in c['phase']);assert c['prefix_plan_enabled']==on
    delta=c['prefix_plan_counter_delta'];assert (delta[0]>0 and delta[1]>0)if on else delta==[0,0,0]
 d['rejection_kind']='admission'if any(json.loads((r/'results'/f'glm-transient-Model-r8864-{run}-stage{s}-rank0.json').read_text()).get('resident_admission_rejected')for s in(0,1))else'numeric_or_none'
 d['qualified']=jobs[0][1:3]==['COMPLETED','0:0']and not rejected and len(tailcases)in(2,4)
 d['performance_adopted']=False;d['scope']='One unchanged qualified561 model/replica, isolated thenjoint B16M8 mappedoriginal477globalrows/full128/EOS/QA/16normal,commoncollectorclockseparate;60min3500total4nodes,oneweightloadperreplica.'
 return d
def audit():
 from glm_DP_clock_metrics_r8864 import common_metrics
 launch=json.loads((r/'glm-transient-Model-r8864-launch.json').read_bytes());common_run=launch['run'];owners={a:audit_replica(a)for a in('A','B')}
 payloads={a:json.loads((r/'control'/('glm-transient-Model-r8864-control-'+common_run+'-'+a+'-ids.json')).read_bytes())for a in('A','B')}
 fingerprints={a:{hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()for v in [p['world'],*p['tp'],p['control']]}for a,p in payloads.items()};assert fingerprints['A'].isdisjoint(fingerprints['B'])
 cp=r/'results'/('GLM-DP-common-clock-r8864-'+common_run+'.json');clock=json.loads(cp.read_bytes());payload=json.loads((r/'control'/('glm-transient-Model-r8864-control-'+common_run+'-A-ids.json')).read_bytes());metrics=common_metrics(clock,{a:q['tail_ordinary_cases']for a,q in owners.items()},set(payload['chat_eos_token_ids']))
 return {'hardware':'BW1000','level':'wholeModel_DP','run':common_run,'qualified':all(q['qualified']for q in owners.values())and metrics['joint']['actual_joint_overlap_proved'],'both_replicas_Model_quality_passed':all(q['qualified']for q in owners.values()),'ABBA':{},'all_serving_cards':32,'communicator_ID_fingerprints':{a:sorted(v)for a,v in fingerprints.items()},'independent_replica_ID_sets':True,'replicas':owners,'common_clock':{'file':cp.name,'SHA256':sha(cp),'bytes':cp.stat().st_size,'metrics':metrics},'performance_adopted':False,'scope':'Actual overlappingjoint arrivals/commonCPUclock; isolatedA andB onlycontentioncontrols, never sum their sequentialrates asjointmeasurement. Bothmodels one load/twoarms;originalemitter metrics includedperreplica. Not matchedproviderPP4comparison.'}
if __name__=='__main__':print(json.dumps(audit()))
