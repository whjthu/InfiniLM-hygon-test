import pathlib,json,hashlib,statistics,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-W2-INT8-phase-K114B';hp=r/'results'/f'{label}-control.json';w=json.loads(hp.read_bytes());assert w['completed']and w['passed'],w;actual=w['actual_handle'];sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();med=statistics.median
raw=[];owners=[]
for node in range(2):
 p=r/'results'/f'{label}-{actual}-node{node}.json';q=json.loads(p.read_bytes());assert q['completed']and q['normal_CUDA_cleanup']and q['native']['all8normal_before_communicators'];raw.append({'path':str(p),'SHA256':sha(p),'node':node,'native_session_s':q['native_session_s'],'actual_bindings':q['actual_bindings'],'manifest_SHA256':q['manifest_SHA256']});owners+=q['native']['reports']
assert len(owners)==16 and sorted(x['global_tp_rank']for x in owners)==list(range(16));keys=['owner_stream_route_W13_ms','owner_stream_quant_jobs_ms','owner_stream_W2matrix_or_pair_ms','owner_stream_reduce_ms','owner_stream_provider_span_ms','signed_wall_minus_provider_ms'];wall='full_MoE_complete_wall_ms';scenes=[]
for T,constructed in [(8,False),(64,False),(64,True)]:
 scene={'T':T,'constructed_low_reuse':constructed,'blocks':{},'perowner_event_durations':[],'branches':[],'quality':{}}
 aa={o['global_tp_rank']:[a for a in o['arms']if a['T']==T and a['constructed_low_reuse']==constructed]for o in owners};assert all(len(v)==24 for v in aa.values())
 for block in ['pre_control','events','post_control']:
  arms=[]
  for arm in range(8):
   local={rank:next(a for a in v if a['diagnostic_block']==block and a['arm']==arm)for rank,v in aa.items()};samples=[]
   for rep in range(8):
    values={rank:a['samples'][rep]for rank,a in local.items()};rank=max(values,key=lambda rank:values[rank][wall]);z=values[rank];v={'rep':rep,'wholecriticalmax_ms':z[wall],'critical_global_rank':rank}
    if block=='events':v.update({k:z[k]for k in keys})
    samples.append(v)
   arms.append({'arm':arm,'kind':local[0]['kind'],'paired_comparison':arm//2,'median_criticalmax_ms':med(z['wholecriticalmax_ms']for z in samples),'samples':samples})
  off=[a['median_criticalmax_ms']for a in arms if a['kind']=='W2_pair_baseline_OFF'];on=[a['median_criticalmax_ms']for a in arms if a['kind']=='W2_INT8_ON'];pairs=[]
  for i in range(4):
   pair=arms[i*2:i*2+2];offp=next(a['median_criticalmax_ms']for a in pair if a['kind']=='W2_pair_baseline_OFF');onp=next(a['median_criticalmax_ms']for a in pair if a['kind']=='W2_INT8_ON');pairs.append({'pair':i,'OFF_ms':offp,'ON_ms':onp,'inversewall_rate_gain_percent':100*(offp/onp-1),'ON_minus_OFF_ms':onp-offp})
  scene['blocks'][block]={'OFF_4arm_median_criticalmax_ms':med(off),'ON_4arm_median_criticalmax_ms':med(on),'inversewall_rate_gain_percent':100*(med(off)/med(on)-1),'ON_minus_OFF_ms':med(on)-med(off),'pairs':pairs,'arms':arms}
 for rank,v in aa.items():
  row={'global_rank':rank,'OFF':{},'ON':{},'paired_ON_minus_OFF_median_ms':{},'paired_ON_minus_OFF_all4_ms':{}}
  ev=[a for a in v if a['diagnostic_block']=='events']
  for enabled,kind in [('OFF','W2_pair_baseline_OFF'),('ON','W2_INT8_ON')]:
   ss=[z for a in ev if a['kind']==kind for z in a['samples']];assert len(ss)==32
   row[enabled]={k:{'median':med(z[k]for z in ss),'min':min(z[k]for z in ss),'max':max(z[k]for z in ss)}for k in [wall]+keys}
  for k in [wall]+keys:
   delta=[]
   for pair in range(4):
    az=[a for a in ev if a['paired_comparison']==pair];off=next(a for a in az if a['kind']=='W2_pair_baseline_OFF');on=next(a for a in az if a['kind']=='W2_INT8_ON');delta.append(med(z[k]for z in on['samples'])-med(z[k]for z in off['samples']))
   row['paired_ON_minus_OFF_median_ms'][k]=med(delta);row['paired_ON_minus_OFF_all4_ms'][k]=delta
  scene['perowner_event_durations'].append(row)
  owner=next(o for o in owners if o['global_tp_rank']==rank);assert owner['K114B_events_closed_before_Runtime']and owner['normal_Runtime_close']and owner['module_closed']and owner['INT8_atom_CPU_integer_oracle_all256_pass'];branch=next(b for b in owner['current_branch']if b['T']==T and b['constructed_low_reuse']==constructed);assert branch['I192_calls']==196 and branch['I192_tokens']==196*T and branch['I8_ON_calls']==98 and branch['I8_OFF_calls']==98 and branch['W2_original_OFF_calls']==0 and branch['W2_pair_ON_calls']==98 and branch['legacy_variant_calls']==0;scene['branches'].append({'rank':rank,**branch})
 ss=[a for v in aa.values()for a in v if a['kind']=='W2_INT8_ON'];scene['quality']={'candidate_RMS_minmax':[min(a['output_relative_RMS_vs_firstOFF']for a in ss),max(a['output_relative_RMS_vs_firstOFF']for a in ss)],'candidate_maxabs_minmax':[min(a['output_maxabs_vs_firstOFF']for a in ss),max(a['output_maxabs_vs_firstOFF']for a in ss)],'finite_canary_all_pass':all(a['input_readonly_and_tail_canary']for v in aa.values()for a in v)}
 control={};pre=scene['blocks']['pre_control'];post=scene['blocks']['post_control'];event=scene['blocks']['events']
 for enabled in ['OFF','ON']:
  key=enabled+'_4arm_median_criticalmax_ms';reference=(pre[key]+post[key])/2;control[enabled]={'pre_ms':pre[key],'events_ms':event[key],'post_ms':post[key],'bracket_midpoint_ms':reference,'post_minus_pre_ms':post[key]-pre[key],'events_minus_midpoint_ms':event[key]-reference,'events_vs_midpoint_percent':100*(event[key]/reference-1)}
 control['ON_minus_OFF_gap_ms']={'pre':pre['ON_minus_OFF_ms'],'events':event['ON_minus_OFF_ms'],'post':post['ON_minus_OFF_ms'],'events_minus_bracket_midpoint':event['ON_minus_OFF_ms']-(pre['ON_minus_OFF_ms']+post['ON_minus_OFF_ms'])/2}
 scene['control_perturbation']=control;scene['phase_delta_perowner_median_distribution_ms']={k:{'median_over16owners':med(z['paired_ON_minus_OFF_median_ms'][k]for z in scene['perowner_event_durations']),'min':min(z['paired_ON_minus_OFF_median_ms'][k]for z in scene['perowner_event_durations']),'max':max(z['paired_ON_minus_OFF_median_ms'][k]for z in scene['perowner_event_durations'])}for k in [wall]+keys};scenes.append(scene)
handles={actual,w['CPU_actual_step']}
for k in ['fresh_before','post_KFD']:
 if k in w:handles.update(x['step']for x in w[k]['rows'])
out={'schema':1,'status':'K114B_DIAGNOSTIC_TERMINAL','GPU_actual':actual,'CPUkeeper_actual':w['CPU_actual_step'],'source_publication':'479891d7e206271146550a131b7ce2db68ce6b08','source_manifest_SHA256':'98675a65d0626deaf6494798d83bcad069adec17aad6e1906297a3df3d9daf9f','lifecycle':w,'raw_evidence':raw,'control_file_SHA256':sha(hp),'scenarios':scenes,'Slurm_terminal':subprocess.check_output(['sacct','-j',','.join(sorted(handles)),'--format=JobID,State,ExitCode,Elapsed,NodeList','-P','-n'],text=True,timeout=25),'interpretation_rules':['Each wholecriticalmax sample choosesmaxwall over16owners; phase fields come fromthat same rank. Never sum crossrankphase maxima.','Perowner phases and paired deltas remain sameowner; aggregatingphase medians is descriptive, not criticalpath reconstruction.','Event elapsed on originalownerstream is not GPUbusy; phase can include streamidle duehostlaunch gaps.','signedwallminusproviderspan contains CPU/otherGPU/router/shared/TP/drain, not pureTP.','Pre/post controls quantifyinstrumentation+time drift; when masking smallT8difference report UNKNOWN/PARK; diagnostic neverpromotesModel.'],'GPU_F64':False,'default_dispatch':'OFF/PARK','no_new_math_grid_Model':True}
print(json.dumps(out))
