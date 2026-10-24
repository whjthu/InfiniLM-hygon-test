import json,pathlib,sys,math,statistics
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text());assert d['completed']and d['no_Model_TPS'];arms=[0,1,1,0,0,1,1,0,0,1,1,0,0];assert d['arms']==arms and d['formal_trials']==list(range(4,12));assert sorted(o['dev']for o in d['owners'])==list(range(8));assert d['weight_keys']==1536
for o in d['owners']:
 assert o['normal_owner_closed']and o['destroy_code']==0 and o['Runtime_expired']==1 and len(o['records'])==13
 for t,z in enumerate(o['records']):
  assert z['trial']==t and z['mode']==arms[t]and z['warm']==(t<4)and z['formal']==(4<=t<12);assert z['finite']and z['numeric_guard'];assert z['native_gate_calls']==arms[t]and z['native_down_calls']==arms[t]
  for k in ['complete_CAPI_wall_seconds','whole_routed_return_seconds','routed_plus_outputcopy_drain_seconds']:assert math.isfinite(z[k])and z[k]>0
  assert z['routed_plus_outputcopy_drain_seconds']>=z['whole_routed_return_seconds'];assert z['complete_CAPI_wall_seconds']>=z['routed_plus_outputcopy_drain_seconds']
assert len(set.union(*[set(o['affinity'])for o in d['owners']]))==32
rows=[]
for t in range(13):
 z=[o['records'][t]for o in d['owners']];rows.append({'trial':t,'mode':arms[t],'slowest_owner_CAPI_s':max(v['complete_CAPI_wall_seconds']for v in z),'slowest_owner_native_s':max(v['routed_plus_outputcopy_drain_seconds']for v in z),'commonhost_envelope_s':max(v['CPU_CAPI_end']for v in z)-min(v['CPU_CAPI_begin']for v in z),'owner_native_s':[v['routed_plus_outputcopy_drain_seconds']for v in z],'owner_CAPI_s':[v['complete_CAPI_wall_seconds']for v in z]})
blocks=[]
for a in [4,8]:
 b={'first_trial':a}
 for k in ['slowest_owner_CAPI_s','slowest_owner_native_s','commonhost_envelope_s']:
  o=statistics.mean(rows[t][k]for t in[a,a+3]);n=statistics.mean(rows[t][k]for t in[a+1,a+2]);b[k]={'O_mean_s':o,'N_mean_s':n,'speedup_fraction':o/n-1,'O_endpoint_drift_fraction':rows[a+3][k]/rows[a][k]-1}
 blocks.append(b)
print(json.dumps({'passed':True,'actual_handle':d['step'],'normal_owners':8,'numeric_outputs':104,'all_exact_diagnostic':all(v['byteexact_diagnostic']for o in d['owners']for v in o['records']),'maxabs':max(v['maxabs']for o in d['owners']for v in o['records']),'NRMSEmax':max(v['NRMSE']for o in d['owners']for v in o['records']),'trials':rows,'two_ABBA':blocks,'scope':'8 independent TP0 replicas, rank not timing repetitions, original complete routed expert contribution; no ModelTPS/finaltoken qualification'}))
