import pathlib,json,sys,math,statistics,hashlib
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text());arms=[0,1,2,0,1,2,2,1,0,0,2,1,1,2,0,0];assert d['completed'] and d['no_Model_TPS'] and d['arms']==arms and d['formal_trials']==list(range(3,15));assert d['tokens']==2048 and d['routes']==12288 and d['unique_experts']==251 and d['weight_keys']==1536
assert sorted(o['dev'] for o in d['owners'])==list(range(8));U=251
for o in d['owners']:
 assert len(o['quant1_boundary_actual_quality'])==12 and all(z['packed_and_scale_bytes_exact'] for z in o['quant1_boundary_actual_quality'])
 assert all(any(z['kind']==kind and z['fused']==f and z['CPU_duplicate_rowcopy_oracle'] for z in o['quant1_boundary_actual_quality']) for kind in ['actual','finite_edges','nan_inf'] for f in [0,1])
 assert o['normal_owner_closed'] and o['destroy_code']==0 and o['Runtime_expired']==1 and len(o['records'])==16
 assert o['free_after_load']>=6*1024**3 and o['free_after_quality_prepare']>=6*1024**3 and o['observed_memory_floor_bytes']>=512*1024**2
 for i,z in enumerate(o['records']):
  mode=arms[i];assert z['trial']==i and z['mode']==z['quant1_mode']==mode and z['warm']==(i<3) and z['formal']==(3<=i<15)
  assert z['finite'] and z['numeric_guard'] and z['tokens']==2048 and z['routes']==12288 and z['unique_experts']==U
  assert z['metadata_packed'] and not z['scatter_retire'] and not z['bank_retire'] and z['bank_success_fence_calls_source_contract']==z['scatter_post_fences']==U and z['scatter_post_fences_skipped']==0
  assert z['phase_counter_delta']==[0,1,0,U,0,2,0,1,0,196608,0,0]
  assert z['quant1_calls']==int(mode>0) and z['quant1_rows']==(2048 if mode else 0) and z['gather_scale_calls']==(U if mode else 0) and z['gather_route_rows']==(12288 if mode else 0)
  assert z['complete_CAPI_wall_seconds']>=z['native_routed_copy_drain_seconds']>0 and math.isfinite(z['complete_CAPI_wall_seconds'])
assert len(set.union(*[set(o['affinity']) for o in d['owners']]))==32
trials=[]
for i in range(16):
 zs=[o['records'][i] for o in d['owners']];trials.append({'trial':i,'mode':arms[i],'maxowner_CAPI_s':max(z['complete_CAPI_wall_seconds'] for z in zs),'maxowner_native_s':max(z['native_routed_copy_drain_seconds'] for z in zs),'samehost_envelope_s':max(z['CPU_CAPI_end'] for z in zs)-min(z['CPU_CAPI_begin'] for z in zs)})
blocks=[]
for start in [3,9]:
 block={'first_trial':start,'comparisons':{}}
 for k in ['maxowner_CAPI_s','maxowner_native_s','samehost_envelope_s']:
  means={mode:statistics.mean(trials[i][k] for i in range(start,start+6) if arms[i]==mode) for mode in [0,1,2]};block['comparisons'][k]={'A_mean_s':means[0],'Q1_mean_s':means[1],'F_mean_s':means[2],'A_over_Q1':means[0]/means[1],'A_over_F':means[0]/means[2],'A_endpoint_drift_fraction':trials[start+5][k]/trials[start][k]-1}
 blocks.append(block)
ownrepeat={str(o['dev']):{str(mode):len(set(z['output_SHA256'] for z in o['records'] if z['quant1_mode']==mode))==1 for mode in [0,1,2]} for o in d['owners']}
print(json.dumps({'passed':True,'actual_handle':d['step'],'normal_owners':8,'numeric_outputs':128,'quant1_boundary_cases':96,'maxabs':max(z['maxabs'] for o in d['owners'] for z in o['records']),'NRMSE_max':max(z['NRMSE'] for o in d['owners'] for z in o['records']),'ownrepeat_hash_equal':ownrepeat,'all_exact_diagnostic':all(z['byteexact_diagnostic'] for o in d['owners'] for z in o['records']),'trials':trials,'paired_blocks':blocks,'source_report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Eight independent TP0 actual2048 routedFFN; ordinary fullGemmEx+quant2+scatter; noModel or decodegain/adoption'}))
