"""Independent CPU finite component verifier; ranks are not timing repetitions."""
import json,pathlib,sys,math,statistics
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text());assert d['completed']and d['no_Model_TPS']and d['arms']==[0,1,0,1,1,0,0,1,1,0];assert sorted(o['dev']for o in d['owners'])==list(range(8));out={'passed':True,'actual_handle':d['step'],'all8normal':True,'scope':'Actual checkpoint layer3 C128; embedding-derived hidden/teacher input component, NOT699 Model replay/TPS','prefixes':[]}
for o in d['owners']:
 assert o['normal_owner_closed']and len(o['records'])==20
 for z in o['records']:
  assert z['ON']==bool(d['arms'][z['trial']])and z['warm']==(z['trial']<2)and z['past']in[575,639];assert len(z['seed'])==8 and len(z['all17_outputs'])==17 and all(v['numeric_guard']for v in z['all17_outputs']);f=z['decode'];assert f['tokens']==32 and math.isfinite(f['CAPI_wall_seconds'])and f['CAPI_wall_seconds']>0;assert f['counter_delta']==([1,256,0]if z['ON']else[0,0,1]);assert f['CPU_CAPI_end']>=f['CPU_CAPI_begin'];assert all(x['tokens']==4*z['past']and x['counter_delta']==([1,4*z['past']*8,0]if z['ON']else[0,0,1])for x in z['seed'])
for past in [575,639]:
 rows=[]
 for trial in range(10):
  fs=[next(z for z in o['records']if z['past']==past and z['trial']==trial)for o in d['owners']];rows.append({'trial':trial,'ON':bool(d['arms'][trial]),'warm':trial<2,'slowest_owner_CAPI_wall_s':max(z['decode']['CAPI_wall_seconds']for z in fs),'commonhost_CAPI_envelope_s':max(z['decode']['CPU_CAPI_end']for z in fs)-min(z['decode']['CPU_CAPI_begin']for z in fs),'owner_CAPI_wall_s':[z['decode']['CAPI_wall_seconds']for z in fs],'owner_CPU_s':[z['decode']['owner_thread_CPU_seconds']for z in fs],'all_outputs_byte_exact_diagnostic':all(v['OFF0_exact_diagnostic']for z in fs for v in z['all17_outputs'])})
 blocks=[]
 for a in [2,6]:
  off=statistics.mean(rows[t]['slowest_owner_CAPI_wall_s']for t in[a,a+3]);on=statistics.mean(rows[t]['slowest_owner_CAPI_wall_s']for t in[a+1,a+2]);blocks.append({'OFF_s':off,'ON_s':on,'latency_speedup_fraction':off/on-1,'OFF_endpoint_drift_fraction':rows[a+3]['slowest_owner_CAPI_wall_s']/rows[a]['slowest_owner_CAPI_wall_s']-1})
 out['prefixes'].append({'past':past,'trials':rows,'two_ABBA_blocks':blocks,'timing_scope':'Python CAPI wall includes original producer sync/Native whole layer/completion, CPU scheduling and CCL. No new sync. owner max per trial; 8 ranks are not8independent repeats.'})
print(json.dumps(out))
