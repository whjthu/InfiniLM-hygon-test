import pathlib,json,sys,statistics,hashlib
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_text());out={'raw_evidence_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'published_commit':q['launch']['published_commit'],'post_clean':q['post_clean'],'normal_ranks':q['normal_ranks'],'scope':q['scope'],'variants':[]}
for item in q['results'].values():
 z=item['raw'];a={'variant':z.get('fusion_variant'),'node':z.get('node'),'actual_step':z.get('actual_step'),'completed':z.get('completed',False),'normal':z.get('normal',False),'error':z.get('error'),'fullchain':[],'activation':[],'component_retained':False}
 for t in [1,8,16]:
  baseline=[v for v in z.get('trials',[]) if v['T']==t and v['mode']==0];candidate=[v for v in z.get('trials',[]) if v['T']==t and v['mode']==1]
  if len(baseline)!=4 or len(candidate)!=4:continue
  rates=[b['wall_mean_seconds']/c['wall_mean_seconds']-1 for b,c in zip(baseline,candidate)];drift=baseline[-1]['wall_mean_seconds']/baseline[0]['wall_mean_seconds']-1
  a['fullchain'].append({'T':t,'paired_speedup':rates,'median_speedup':statistics.median(rates),'baseline_first_last_drift':drift,'baseline_wall_seconds':[x['wall_mean_seconds'] for x in baseline],'candidate_wall_seconds':[x['wall_mean_seconds'] for x in candidate],'passes':sum(x>0 for x in rates)>=3 and statistics.median(rates)>=.05 and abs(drift)<=.05})
 for v in z.get('activation_trials',[]):
  rates=[p['0']['mean_wall_seconds']/p['1']['mean_wall_seconds']-1 for p in v['pairs']];a['activation'].append({'rows':v['rows'],'median_speedup':statistics.median(rates),'packed_mismatch_fraction':v['packed_mismatch_fraction'],'dequant_relative_RMS':v['dequant_relative_RMS']})
 a['component_retained']=a['completed'] and a['normal'] and len(a['fullchain'])==3 and all(v['passes'] for v in a['fullchain']) and q['post_clean'];out['variants'].append(a)
out['retained_variants']=[x['variant'] for x in out['variants'] if x['component_retained']]
target=pathlib.Path(sys.argv[2]);assert not target.exists();target.write_text(json.dumps(out,indent=2));print(json.dumps(out))
