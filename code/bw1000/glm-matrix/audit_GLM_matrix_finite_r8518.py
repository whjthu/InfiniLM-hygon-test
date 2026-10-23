import pathlib,json,statistics,hashlib,sys,math
r=pathlib.Path(sys.argv[1]);step=sys.argv[2];rows=[];summaries=[]
for rank in range(8):
 p=r/'results'/f'glm-matrix-finite-r8518-a14r2n06-rank{rank}.json';q=json.loads(p.read_text())
 assert q['completed']and q['producer']==step and q['rank']==rank and q['layer']==38 and q['offset']==rank*256 and q['weight_load_count']==1 and q['normal_Torch_close']and q['normal_drain_before_release']
 assert q['library_SHA256']=='058141002f2df00aeeb666d34ffcfc8692889ee7d9bc41d5c60fc583ebb1a9dd'
 assert len(q['affinity'])==4 and len(q['cases'])==3
 for c,(tokens,U)in zip(q['cases'],[(1,8),(4,27),(8,53)]):
  assert c['tokens']==tokens and c['routes']==tokens*8 and c['U']==U and c['max_rows_per_expert']<=3 and c['executed_to_useful_FLOPs_ratio']==16 and len(c['projections'])==2
  for proj,gate in zip(c['projections'],[True,False]):
   assert proj['gate_up']==gate and proj['scalar_restore_exact']
   arms=proj['arms'];assert [a['mode']for a in arms]==[0,1,1,0]
   for a in arms:
    assert a['finite']and a['canary']and len(a['samples'])==3 and len(a['own_arm_CPU_oracle'])==3
    assert all(math.isfinite(v[k])and v[k]>0 for v in a['samples']for k in ['event_ms','wall_ms'])
    for o in a['own_arm_CPU_oracle']:assert o['abs_error']<=.03125+.02*abs(o['FP64_CPU_reference'])
   off=statistics.mean(arms[i]['event_median_ms']for i in [0,3]);on=statistics.mean(arms[i]['event_median_ms']for i in [1,2]);summaries.append({'rank':rank,'tokens':tokens,'gateup':gate,'off_event_ms':off,'on_event_ms':on,'speedup':off/on})
 rows.append({'rank':rank,'affinity':q['affinity'],'file':p.name,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(set(v for row in rows for v in row['affinity']))==32
out=r/'results'/('glm-matrix-finite-audit-r8518-'+step+'.json');assert not out.exists();d={'passed':True,'reports':rows,'summaries':summaries,'scope':'Component finite and selected CPU oracle gate only. Numeric gate2percent plus.03125 is diagnostic envelope, not Model acceptance. No Model IDs/EOS/TPS measured. M16 duplicates15rows, usefulFLOPs must excludepadding.'};out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
