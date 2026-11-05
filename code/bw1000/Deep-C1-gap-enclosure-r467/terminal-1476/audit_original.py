"""Offline original HCpre1476 audit. No source imports, device, or remote actions."""
import argparse,base64,gzip,hashlib,json,math,statistics
from pathlib import Path,PurePosixPath
SEAL='4233bec10c6906d0190fc0e519c0fbd38940282fde635e93422d89d68a2bc118'
ROOT='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
ENTRY='458748fc82a48a0472677f8b8d93e9fec0713a9dab12339548ea574a6135d447'
NATIVE='5fd76082aa15f039515728c16bcb320b3b1c245952843e8f93f2038d17c5e0a6'
PROMPT='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
MANIFESTS={'hc-pre-entry-v1':'413dff01e160851b494c4f1ce2b4314e982b1909a00bf0d11faf5aecc0d8b5e4','hc-pre-deployment-v2':'67b9a849b4c7481ff71d9946b56bc37de328804324e5839383ab6a7701464f36','hc-pre-split-cpu-v2':'b368d0c33db11b65a7bc72d60c0ed38cf397d748dcd602758d34e5b62f7efff9'}
LABELS=['cast_square_mean_normalize','mixes_GEMM','affine_sigmoid_softmax','sinkhorn','weighted_collapse_cast']
STAGES=['decoder_envelope','metadata_validation_readback','attention_HC_pre','attention_norm','attention','attention_HC_post','FFN_envelope','FFN_HC_pre','FFN_norm_sync','MoE_envelope','FFN_HC_post','router','routed_experts','shared_experts','TP_reduction','output_cast']
METRICS=['stream_ms','host_wall_ms','thread_cpu_ms']
def sha(b):return hashlib.sha256(b).hexdigest()
def finite(x):assert type(x)in(int,float)and math.isfinite(x)and x>=0;return x
def summary(a):return dict(min=min(a),median=statistics.median(a),max=max(a))
def audit(path):
 raw=Path(path).read_bytes();assert sha(raw)==SEAL;s=json.loads(gzip.decompress(raw));assert len(s['files'])==69;fs={}
 for f in s['files']:
  b=gzip.decompress(base64.b64decode(f['gzip_base64'],validate=True));assert len(b)==f['bytes']and sha(b)==f['SHA256'];assert f['path']not in fs;fs[f['path']]=b
 def actual(p):return ROOT+p[len('/refactor'):]if p.startswith('/refactor/')else p
 def get(p):return fs[actual(p)]
 def one(suffix):
  matches=[(p,b)for p,b in fs.items()if p.endswith(suffix)];assert len(matches)==1,(suffix,len(matches));return matches[0]
 def readone(suffix):return json.loads(one(suffix)[1])
 def packet(f):
  b=get(f['path']);assert len(b)==f['bytes']and sha(b)==f['SHA256'];return json.loads(b)
 sourcepins={}
 for name,h in MANIFESTS.items():
  p,b=one('/'+name+'/SOURCE_MANIFEST.json');assert sha(b)==h;m=json.loads(b)
  for n,v in m['files_SHA256'].items():assert sha(fs[str(PurePosixPath(p).parent/n)])==v,(name,n)
  sourcepins[p]=h
 ep,eb=one('/entry-report.json');assert sha(eb)==ENTRY;e=json.loads(eb);w=readone('-watch.json');closed=readone('-GPU_PROCESS_CLOSED.json');started=readone('-GPU_PROCESS.json');cfg=readone('-actual-config.json')
 assert e['completed']and e['passed']and e['phase']=='normal_closed'and e['normal_Native_shutdown']and e['engine_worker_shutdown_completed']and e['observed_all8_threads_exited']
 assert e['HTTP']==0 and e['Model_READY']is False and e['loaded_key_count']==67569 and e['cache_rows']==1 and e['cache_capacity']==32768
 ident=e['identity'];assert ident['actual_step']=='910374.1476'and ident['Native_SHA256']==NATIVE and ident['gpu_start_epoch']==1793862245.117231
 assert cfg['Native_SHA256']==NATIVE and cfg['HC_mode']==0 and cfg['precision_policy']==0 and not cfg['fullpre']and cfg['r366']==0
 assert e['precision_policy']['policy']==0 and e['precision_policy']['status']==0 and not e['precision_policy']['F64_allowed']and e['compact_default_OFF']
 assert closed['normal']and closed['actual_step']==ident['actual_step']and closed['gpu_start_epoch']==ident['gpu_start_epoch']and closed['entry_report_SHA256']==ENTRY
 assert closed['PID']==started['PID']==3666236 and closed['UID']==started['UID']==154459 and closed['startticks']==started['startticks']=='388271757'
 assert len(e['owner_thread_observation']['owners'])==8 and {r['device']for r in e['owner_thread_observation']['owners']}==set(range(8))
 owners=e['owner_affinity_after_weight_load'];assert owners['verified']and owners['width']==4;masks=[]
 for r in owners['actual_owner_report']['owners']:
  assert r['changes']==1;mask=set(map(int,r['verified_mask'].split(',')));assert len(mask)==4 and all(not(mask&m)for m in masks);masks.append(mask)
 assert len(masks)==8
 for key in ['GPU_memory_after_create','GPU_memory_after_load']:assert len(e[key])==8 and all(f>=16*1024**3 and t==68702699520 for f,t in e[key])
 assert e['blas_policy_before_destroy']['failures']==0 and len(e['blas_policy_before_destroy']['owners'])==8 and all(r['created']==r['verified']==1 for r in e['blas_policy_before_destroy']['owners'])
 cpu=readone('/CPU-build.json');assert sha(one('/CPU-build.json')[1])=='97702f1120ebf6fdd7d56c01d58dfc1be4ab03b7b3177ea3c496b33fff3b5a00';assert cpu['passed']and cpu['retained_count']==49 and cpu['new_HOST']==1 and cpu['new_GPU']==0 and cpu['library_SHA256']==NATIVE and cpu['no_new_sync']and cpu['strip_equals_original']
 assert w['completed']and not w['passed']and w['forced_cleanup']and w['error']=='AssertionError()'and "closed['entry_report_SHA256']==result['entry_report_SHA256']"in w['traceback']
 cleanup=w['owned_cleanup'];assert cleanup['closed']and not cleanup['terminal_unknown_or_unclosed']and not cleanup['PGID_still_live'];assert all(x['row'][2:4]==['COMPLETED','0:0']and x['srun_rc']==0 and not x['group_live']for x in cleanup['observations'])
 assert len(cleanup['signals'])==2 and cleanup['signals'][0]['signal']=='TERM'and cleanup['signals'][0]['rc']==0 and cleanup['signals'][1]['already_gone']and not any(x['signal']=='KILL'for x in cleanup['signals'])
 post=w['post_after_failure'];assert post['actual_step']=='910374.1483'and post['KFD_available']and not post['kfd_pids']and not post['matching_component_processes']and not post['expected_PIDs_still_exist']and all(post['ports_bindable'].values())and len(set(post['physical_PCI']))==8
 ac=s['accounting'];assert ac['rc']==0
 for step,state,rc in [('910374.1476','COMPLETED','0:0'),('910407.1035','FAILED','1:0'),('910374.1483','COMPLETED','0:0')]:assert any(x.split('|')[0]==step and x.split('|')[2:4]==[state,rc]for x in ac['rows'])
 rs=e['requests'];assert len(rs)==5 and[r['armed']for r in rs]==[False,False,False,True,False]and[r['warm']for r in rs]==[True,True,False,False,False]
 capi=e['actual_single_CAPI_calls'];assert len(capi)==530==e['HC_dispatch_checked_forwards']==e['layout_dispatch_checked_forwards'];calls=[];base=0;req=[];windows=[];seenpacket=[]
 for r in rs:
  ids=r['generated_token_ids'];assert r['prompt_tokens']==8489 and r['prompt_IDs_SHA256']==PROMPT and r['output_budget']==128 and r['normal_reset_before']and r['normal_reset_after'];assert len(ids)==98 and ids[-1]==1 and ids.count(1)==1 and all(type(t)is int and 0<=t<129280 for t in ids)and r['natural_EOS']and not r['cap'];assert r['forward_attempts']==r['forward_completed']==9+len(ids)-1
  p=capi[base:base+r['forward_completed']];base+=len(p);calls.append(p);assert [x['tokens']for x in p[:9]]==[1024]*8+[297]and all(x['tokens']==x['count']==1 for x in p[9:])and all(x['count']==1 and x['rc']==0 and abs(finite(x['seconds'])-(x['return_monotonic']-x['start_monotonic']))<1e-8 for x in p)
  ticks=r['token_monotonic'];assert len(ticks)==len(ids)and all(b>a for a,b in zip(ticks,ticks[1:]));visible=ticks[:-1];assert len(visible)==97
  assert len(r['HC_owner_counts'])==8 and all(not any(v['fullpre'].values())and v['legacy']['f32_t1']==0 for v in r['HC_owner_counts'])
  for key in ['GPU_free_total_before','GPU_free_total_after']:assert len(r[key])==8 and all(f>=16*1024**3 for f,t in r[key])
  req.append(dict(index=r['index'],armed=r['armed'],warm=r['warm'],generated_including_EOS=len(ids),visible_tokens=len(visible),natural_EOS=True,cap=False,whole_with_reset_s=r['whole_request_with_reset_s'],prefill_CAPI_sum_s=sum(x['seconds']for x in p[:9]),decode_CAPI_sum_s=sum(x['seconds']for x in p[9:]),CAPI_sum_s=sum(x['seconds']for x in p),outside_CAPI_enclosure_s=r['whole_request_with_reset_s']-sum(x['seconds']for x in p),visible_decode_tokens_s=(len(visible)-1)/(visible[-1]-visible[0])))
  for win in r['profile_windows']:
   assert r['index']==3 and win['native_window_closed']and win['owner_event_validation_passed']and win['inner_event_validation_passed'];g=win['geometry'];i=g['decode_call_1based'];assert i in [8,87]and g['past']==[8489+i-1]and g['rows']==[0]and g['tokens']==g['requests']==1
   op=packet(win['scalar_packets']['owner']);ip=packet(win['scalar_packets']['inner']);hp=packet(win['HC_pre_split']['packet']);seenpacket.extend([win['scalar_packets']['owner'],win['scalar_packets']['inner'],win['HC_pre_split']['packet']]);assert op['schema']==ip['schema']==hp['schema']==1 and len(op['rows'])==len(ip['rows'])==344 and len(hp['rows'])==688 and hp['labels']==LABELS
   byowner={(x['device'],x['layer']):x for x in op['rows']};assert set(byowner)=={(d,l)for d in range(8)for l in range(43)};owner={k:{m:[0.]*8 for m in METRICS}for k in STAGES}
   for row in op['rows']:
    assert row['markers']==32 and row['event_create_during_device']==row['event_query_during_device']==row['device']and row['owner']and len(row['spans'])==16
    for j,x in enumerate(row['spans']):
     assert x['stage']==j and abs(finite(x['stream_ms'])-(x['end_ms']-x['start_ms']))<.01
     for m in METRICS:owner[STAGES[j]][m][row['device']]+=finite(x[m])
   for name,arr in owner.items():
    for m,record in [('stream_ms','old_perstage_perdevice_ms'),('host_wall_ms','old_host_wall_perstage_perdevice_ms'),('thread_cpu_ms','old_thread_cpu_perstage_perdevice_ms')]:assert arr[m]==win[record][name]
   innerseen=set();inner=[[0.]*8 for _ in range(6)]
   for row in ip['rows']:
    d,l=row['device'],row['layer'];assert (d,l)not in innerseen;innerseen.add((d,l));parent=byowner[(d,l)];assert(row['owner'],row['stream'])==(parent['owner'],parent['stream'])and row['tokens']==row['requests']==1 and row['events_created']==row['events_queried']==row['events_destroyed']==7 and row['create_during_device']==row['query_during_device']==row['destroy_during_device']==d and len(row['segments'])==6
    for j,x in enumerate(row['segments']):
     assert x['segment']==j and abs(finite(x['stream_ms'])-(x['end_ms']-x['start_ms']))<.01
     for m in METRICS:finite(x[m])
     inner[j][d]+=x['stream_ms']
    assert abs(sum(x['stream_ms']for x in row['segments'])-row['whole_stream_ms'])<.02
   assert len(innerseen)==344 and list(win['inner_persegment_perdevice_ms'].values())==inner
   sums={k:{m:[0.]*8 for m in METRICS}for k in LABELS};seen=set();drain=[0.]*8
   for row in hp['rows']:
    d,l,hc=row['device'],row['layer'],row['HC'];key=d,l,hc;assert key not in seen and d in range(8)and l in range(43)and hc in[0,1];seen.add(key);parent=byowner[(d,l)];assert(row['owner'],row['stream'])==(parent['owner'],parent['stream'])and row['thread']and row['created']==row['queried']==row['destroyed']==6 and len(row['spans'])==5;drain[d]+=finite(row['original_drain_host_ms'])
    for j,x in enumerate(row['spans']):
     assert x['stage']==j
     for m in METRICS:sums[LABELS[j]][m][d]+=finite(x[m])
    assert abs(sum(x['stream_ms']for x in row['spans'])-row['whole_stream_ms'])<.02
   for label,a in sums.items():
    for m,field in [('stream_ms','stream'),('host_wall_ms','host'),('thread_cpu_ms','threadCPU')]:assert a[m]==win['HC_pre_split']['perstage_perowner_ms'][label][field]
   siblings=[sum(sums[label]['stream_ms'][d]for label in LABELS)for d in range(8)];parenthc=[owner['attention_HC_pre']['stream_ms'][d]+owner['FFN_HC_pre']['stream_ms'][d]for d in range(8)];decoder=owner['decoder_envelope']['stream_ms']
   windows.append(dict(decode_call=i,past=g['past'][0],HC_rows=688,inner_rows=344,owner_rows=344,new_events_created_queried_destroyed_each=4128,inner_events_created_queried_destroyed_each=2408,phase_per_owner_ms=sums,phase_owner_sum_summary_ms={k:{m:summary(v)for m,v in a.items()}for k,a in sums.items()},HC_siblings_stream_per_owner_ms=siblings,original_HC_parent_stream_per_owner_ms=parenthc,HC_parent_minus_siblings_stream_per_owner_ms=[p-c for p,c in zip(parenthc,siblings)],original_HC_drain_host_per_owner_ms=drain,decoder_stream_per_owner_ms=decoder,HC_siblings_over_decoder_per_owner=[a/b for a,b in zip(siblings,decoder)],owner_nested_perstage_ms=owner,inner_sibling_stream_perowner_ms=inner,observer_timers_s={k:v for k,v in win.items()if k.endswith('_s')},HC_read_and_IO_s={k:v for k,v in win['HC_pre_split'].items()if k.endswith('_s')},whole_observer_wrapper_s=r['calls'][9+i-1]['whole_observer_wrapper_s']))
 assert len(windows)==2 and len(seenpacket)==6
 overhead={}
 for i in [8,87]:
  rec=[]
  for j in [2,3,4]:rec.append(dict(request=j,past=8489+i-1,prefix_SHA256=sha(json.dumps(rs[j]['generated_token_ids'][:i],separators=(',',':')).encode()),CAPI_s=calls[j][9+i-1]['seconds']))
  same=len({x['prefix_SHA256']for x in rec})==1;off=(rec[0]['CAPI_s']+rec[2]['CAPI_s'])/2;overhead[str(i)]=dict(observed=rec,same_input_prefix=same,armed_extra_fraction=rec[1]['CAPI_s']/off-1 if same else None,OFF_endpoint_fraction=rec[2]['CAPI_s']/rec[0]['CAPI_s']-1)
 # Exact original collector's p-shadow predicts final HC packet hash, not entry hash.
 _,collector=one('/hc-pre-deployment-v2/collect.py');assert b"p=output/'entry-report.json'"in collector and b"p=actual(hc['packet']['path'])"in collector and b'entry_report_SHA256=sha(p)'in collector
 erroneous=seenpacket[-1]['SHA256'];assert erroneous==rs[3]['profile_windows'][-1]['HC_pre_split']['packet']['SHA256']and erroneous!=ENTRY
 return dict(schema='Deep467-HCpre1476-independent-original-audit-v1',passed=True,passed_scope='Offline file/source/schema/clock/event/lifecycle audit; original keeper remains FAILED',archive=dict(path=str(path),SHA256=sha(raw),files=len(fs),decoded_bytes=sum(map(len,fs.values()))),source_manifest_pins=sourcepins,entry_report_SHA256=ENTRY,Native_SHA256=NATIVE,original_outcome=dict(GPU='910374.1476 COMPLETED0',keeper='910407.1035 FAILED1',post='910374.1483 COMPLETED0',watch_passed=False,forced_cleanup=True,Native_normal8=True,signals=cleanup['signals'],post_clean=True),collector_failure=dict(source_SHA256=sha(collector),cause='p overwritten from entry report path to final HC packet path; terminal footer compares wrong digest',original_erroneous_return_SHA256=erroneous,GPU_closed_actual_entry_report_SHA256=ENTRY,immutable_reader_binding='entry_report_path is retained separately; no producer or original report modified'),requests=req,CAPI_calls=530,profile_windows=windows,same_prefix_CAPI_overhead=overhead,packet_refs=seenpacket,quality_qualification=False,exclusive_GPUbusy=False,exclusive_CPUbusy=False,limitations=['Five HC sibling intervals can sum only within each owner; nested decoder/HC/FFN stages and eight concurrent owners cannot be added.','Stream elapsed includes launch gaps/dependency/communication waits. Thread CPU includes driver/spin; neither proves exclusive compute.','New event creation/query/destruction lies outside HC sibling intervals but inside original HC/CAPI enclosure; parent-minus-siblings is not solely event overhead.','No raw activations/logits captured. Finite/counter/dispatch evidence is original producer assertions; this reader cannot replay numerical equality.','Two matched-state samples estimate observer overhead on this run, not optimization gain or task quality.'])
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);Path(a.output).write_text(json.dumps(q,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=q['passed'],files=q['archive']['files'],HC_rows=sum(w['HC_rows']for w in q['profile_windows']),CAPI=q['CAPI_calls'],original_keeper=q['original_outcome']['keeper'],output_SHA256=sha(Path(a.output).read_bytes()))))
if __name__=='__main__':main()
