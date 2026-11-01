"""Read only635 scalar durations; no remote access, source mutation or GPU work."""
from pathlib import Path
from collections import Counter
import hashlib,json,statistics,math,sys

G=Path(__file__).resolve().parents[2];D=G/'results/bw1000/2026-11-02';med=statistics.median
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quantile(values,q):
 a=sorted(values);k=(len(a)-1)*q;i=int(k);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(k-i)
def describe(values):return dict(n=len(values),median=med(values),mean=statistics.mean(values),p90=quantile(values,.9),p95=quantile(values,.95),p99=quantile(values,.99),maximum=max(values))

paths=sorted(D.glob('Kimi345-C1-layout-Delta-910371.635-node*.json'));assert len(paths)==2
reports=[json.loads(p.read_bytes())for p in paths];owners=sorted([w for r in reports for w in r['workers']],key=lambda w:w['rank'])
assert[w['rank']for w in owners]==list(range(16))
for r in reports:
 assert r['actual_GPU']=='910371.635'and r['completed']and r['normal_close']and r['C']==r['T']==1 and r['pool']==3 and r['Math']==0 and r['capture']is False
assert len({r['source_manifest_SHA256']for r in reports})==len({r['CPU_receipt_SHA256']for r in reports})==len({r['Native_SHA256']for r in reports})==len({r['Main_SHA256']for r in reports})==1

bykey={};values={arm:{rank:[]for rank in range(16)}for arm in ['OFF','ON']};rankrows=[]
for w in owners:
 assert w['fused_counts']==[160,0]and w['first_ON_dispatch']['counts']==[1,0]and len(w['samples'])==320 and w['normal_close']
 for i,s in enumerate(w['samples']):
  assert(s['block'],s['trial'],s['round'])==(i//10,i%10,i//40)
  r=s['round'];on=([False,True,True,False]if r%2==0 else[True,False,False,True])[s['block']%4]
  assert s['arm']==('ON'if on else'OFF')and s['warmup']==(s['trial']<2)and s['profile']is None
  assert s['source_rows_unchanged']and math.isfinite(s['API_wall_ms'])and s['API_wall_ms']>0
  if s['arm']=='OFF':assert s['output_state_equal_to_OFF']
  bykey.setdefault((s['block'],s['trial']),[]).append((w['rank'],s))
  if not s['warmup']:values[s['arm']][w['rank']].append(s['API_wall_ms'])
 ratios=[]
 for r in range(8):
  a=[s['API_wall_ms']for s in w['samples']if s['round']==r and s['arm']=='OFF'and not s['warmup']];b=[s['API_wall_ms']for s in w['samples']if s['round']==r and s['arm']=='ON'and not s['warmup']];assert len(a)==len(b)==16;ratios.append(med(b)/med(a))
 rankrows.append(dict(rank=w['rank'],node=w['rank']//8,device=w['rank']%8,OFF=describe(values['OFF'][w['rank']]),ON=describe(values['ON'][w['rank']]),round_median_ratios=ratios,median_round_ratio=med(ratios),rounds_ratio_below1=sum(v<1 for v in ratios),measurement_elapsed_s=w['measurement_elapsed_s'],sum_of_saved_API_durations_including_warmups_s=sum(s['API_wall_ms']for s in w['samples'])/1000))

trials=[]
for(block,trial),group in sorted(bykey.items()):
 assert len(group)==16 and {r for r,s in group}==set(range(16))
 assert len({(s['arm'],s['round'],s['warmup'])for r,s in group})==1
 ordered=sorted(group,key=lambda x:x[1]['API_wall_ms']);s=ordered[-1][1]
 times=[v['API_wall_ms']for r,v in group]
 trials.append(dict(block=block,trial=trial,round=s['round'],arm=s['arm'],warmup=s['warmup'],maximum=max(times),maximum_rank=ordered[-1][0],second=ordered[-2][1]['API_wall_ms'],median=med(times),mean=statistics.mean(times),p90=quantile(times,.9),node0_max=max(v['API_wall_ms']for r,v in group if r<8),node1_max=max(v['API_wall_ms']for r,v in group if r>=8)))

arms={};operators={}
for arm in ['OFF','ON']:
 t=[v for v in trials if v['arm']==arm and not v['warmup']];assert len(t)==128
 counts=Counter(v['maximum_rank']for v in t);nodecounts=Counter(v['maximum_rank']//8 for v in t);switches=total=0
 for block in sorted({v['block']for v in t}):
  local=sorted([v for v in t if v['block']==block],key=lambda x:x['trial'])
  for a,b in zip(local,local[1:]):switches+=a['maximum_rank']!=b['maximum_rank'];total+=1
 winner_local_percentiles=[]
 for v in t:
  vv=values[arm][v['maximum_rank']];winner_local_percentiles.append(sum(x<=v['maximum']for x in vv)/len(vv))
 arms[arm]=dict(maximum_rank_counts=dict(sorted(counts.items())),maximum_node_counts=dict(sorted(nodecounts.items())),distinct_maximum_ranks=len(counts),dominant_rank=counts.most_common(1)[0],within_block_maximum_rank_switches=switches,within_block_transitions=total,within_block_switch_rate=switches/total,selected_value_local_percentile=describe(winner_local_percentiles),max_minus_second_ms=describe([v['maximum']-v['second']for v in t]),node0_maximum_ms=describe([v['node0_max']for v in t]),node1_maximum_ms=describe([v['node1_max']for v in t]))
for op in ['maximum','second','p90','median','mean']:
 off=[v[op]for v in trials if v['arm']=='OFF'and not v['warmup']];on=[v[op]for v in trials if v['arm']=='ON'and not v['warmup']]
 operators[op]=dict(OFF_median_ms=med(off),ON_median_ms=med(on),median_ratio=med(on)/med(off),OFF_mean_ms=statistics.mean(off),ON_mean_ms=statistics.mean(on),mean_ratio=statistics.mean(on)/statistics.mean(off))

rounds=[]
for r in range(8):
 row=dict(round=r,individual_rank_medians_improved=sum(v['round_median_ratios'][r]<1 for v in rankrows),individual_rank_medians_worse=sum(v['round_median_ratios'][r]>1 for v in rankrows))
 for arm in ['OFF','ON']:
  t=[v for v in trials if v['round']==r and v['arm']==arm and not v['warmup']]
  local_medians=[med(s['API_wall_ms']for s in w['samples']if s['round']==r and s['arm']==arm and not s['warmup'])for w in owners]
  mm=max(local_medians);mx=med(v['maximum']for v in t);row[arm]=dict(max_of_rank_medians_ms=mm,median_of_trial_maxima_ms=mx,selection_tail_premium_ms=mx-mm,rank_maximum_counts=dict(sorted(Counter(v['maximum_rank']for v in t).items())))
  if arm=='OFF':
   blocks=sorted({v['block']for v in t});bracket=[med(v['maximum']for v in t if v['block']==b)for b in blocks];row['OFF_bracket_medians_ms']=bracket;row['OFF_bracket_relative_spread']=abs(bracket[1]-bracket[0])/med(bracket)
 row['max_of_rank_medians_ratio']=row['ON']['max_of_rank_medians_ms']/row['OFF']['max_of_rank_medians_ms']
 row['median_of_trial_maxima_ratio']=row['ON']['median_of_trial_maxima_ms']/row['OFF']['median_of_trial_maxima_ms']
 row['delta_ms']=dict(max_of_rank_medians=row['ON']['max_of_rank_medians_ms']-row['OFF']['max_of_rank_medians_ms'],selection_tail_premium=row['ON']['selection_tail_premium_ms']-row['OFF']['selection_tail_premium_ms'],median_of_trial_maxima=row['ON']['median_of_trial_maxima_ms']-row['OFF']['median_of_trial_maxima_ms'])
 assert abs(row['delta_ms']['max_of_rank_medians']+row['delta_ms']['selection_tail_premium']-row['delta_ms']['median_of_trial_maxima'])<1e-12
 rounds.append(row)

blockrows=[]
for b in range(32):
 t=[v for v in trials if v['block']==b and not v['warmup']];blockrows.append(dict(block=b,round=b//4,arm=t[0]['arm'],max16_median_ms=med(v['maximum']for v in t),cross_rank_median_ms=med(v['median']for v in t),argmax_counts=dict(sorted(Counter(v['maximum_rank']for v in t).items()))))

sourcepaths=[G/'code/bw1000/Kimi-KDA-C1-layout-Delta-r345'/n for n in ['driver.py','layer0_native.cpp','fused_r343.hpp','patch_fused.py','patch_current_delta.py','original_delta.cpp','GPU_inside.sh','SOURCE_MANIFEST.json','GPU_MANIFEST.json']]+[G/'code/bw1000/Kimi-C1-Delta-profile-r309/delta_profile_r309.hpp',D/'Kimi345-GPU-launch.json']
position=[]
for t in range(10):
 row={'trial_within_block':t,'warmup_excluded':t<2}
 for arm in ['OFF','ON']:row[arm]=describe([v['maximum']for v in trials if v['trial']==t and v['arm']==arm])
 position.append(row)
result=dict(state='READ_ONLY_CAUSAL_AUDIT_NO_NEW_GPU',actual_GPU='910371.635',inputs_SHA256={str(p.relative_to(G)):sha(p)for p in paths+sourcepaths},samples=dict(owners=16,total=5120,per_owner=320,per_arm_warmed_trials=128,per_arm_per_owner_warmed=128,rounds=8,alignment='Every row matches block*10+trial; round=block//4; all16 same arm/warmup at each key; genuine160hit/0fallback perowner'),rankrows=rankrows,arms=arms,operators=operators,rounds=rounds,blocks=blockrows,
 conclusions=[
 'No indexing or ON-dispatch miss found. Different statistics aggregate the same valid rows in different orders.',
 'Per-owner median ratios suppress occasional large durations. Per-trial maximum selects one owner tail each trial. max and median do not commute; measured tail premiums are descriptive order-statistic terms, not identified physical phases.',
 'Only API scalar durations are saved. Local and cross-node admission barriers identify a trial but do not timestamp simultaneous CAPI starts. Max16 of local durations is the declared conservative fixture metric; it is not a directly measured global makespan.',
 'The timed Python expression includes ctypes call/return and check plus Native input/metadata upload, forward, final drain and output D2H. Phase fractions, pureGPU busy, collective wait, host scheduling and exact first/last global timestamps are UNKNOWN.',
 'State restore occurs before admission; state extraction, output guard checks, numeric comparison and reporting occur after each local API timer. The source permits those operations from one owner to overlap another owner\'s timed API, but existing durations do not establish actual overlap or its cost.',
 'Keep the fixedC1 KDA family PARKED. Local median improvements do not qualify max16 Delta gain, Model gain or a new target load. No distinct large physical critical interval is isolated, so no new experimental card is justified by this evidence.'
 ],rank_round_directions=dict(improved=sum(v['rounds_ratio_below1']for v in rankrows),total=128,meaning='All16 medians across8ratios below1 does not imply every rank/round improves'),trial_position_diagnostics=position,
 binding_evidence=dict(Native_SHA256=reports[0]['Native_SHA256'],Main_SHA256=reports[0]['Main_SHA256'],CPU_receipt_SHA256=reports[0]['CPU_receipt_SHA256'],source_manifest_SHA256=reports[0]['source_manifest_SHA256'],actual_DSO_bindings=reports[0]['actual_DSO_bindings'],node_device_mapping='node0 a02r3n03 ranks0..7/devices0..7; node1 a02r3n07 ranks8..15/devices0..7; CAPI create checks rank%8=current CUDAdevice',CPU_requested='-c64 per node, cpu-bind=none,8Pythonworker threads, INFINILM_OWNER_CPU_WIDTH_R2392=4',CPU_actual_affinity_masks='UNKNOWN: not saved in635 reports',GPU_per_trial_clocks_busy_or_utilization='UNKNOWN',first_ON_geometry=owners[0]['first_ON_dispatch']['metadata']),
 timer_boundary=dict(measured='Python perf_counter surrounds ctypes call plus check and return. Native includes buffer allocation/wrapping, input and metadata uploads, Delta forward including original o_proj/TP communication dependency, stream drain, output D2H and final drain; temporary destruction occurs before API returns.',input_H2D_bytes_per_owner=14336,metadata_H2D_bytes_per_owner=16,output_D2H_bytes_per_owner=14336,unmeasured_in_API_timer='input-array creation/input hash/output guard allocation; pool restore before admission; state D2H, output guards/input hash comparison, output.copy, numeric conversion/comparison, JSON saves and control barriers after/between local calls',state_restore_H2D_bytes_per_owner=631296,state_inspection_D2H_bytes_per_owner=631296,profile_mode='OFF for all320calls perowner; no event pools created or phase GPU/host intervals saved',raw_time_fields_saved=['API_wall_ms','measurement_elapsed_s aggregate'],raw_time_fields_missing=['CAPI entry/end timestamps','native t0/t2/t3/t4/t5/t6/t7 segments','aligned global start/end','per-thread CPU time','GPU events','CCL wait/GPU busy','OS schedule/affinity trace'],aggregate_elapsed_warning='Owner measurement_elapsed_s includes an unsaved512-token prefix plus320calls/restores/state inspection/barriers/numeric checks/reporting. Subtracting saved API sum does not isolate a removable physical phase or model overhead.'),
 causal_status=dict(proved=['exact sample alignment and genuine candidate path','noncommuting statistics explain why per-owner medians need not imply max-of-trial gain','mobile maximum owners and measured percentile selection','OFF bracket drift up to8.46pct'],not_identified=['whether GIL/host scheduling, copies, stream drain or collective waiting caused each tail','a fixed slow GPU/NIC/NUMA owner','global synchronized makespan','distinct new large Model critical interval']),
 CPU_analysis_only=True,analysis_started_UTC='2026-11-01T19:30:11Z',analyzer_SHA256=sha(Path(__file__)),GPU_new=0,remote_calls=0,running_source_changed=False,Modelwall_impact='UNKNOWN',next_card=None)
target=Path(sys.argv[1])if len(sys.argv)>1 else Path(__file__).with_suffix('.json');assert not target.exists();target.write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(operators=operators,arms=arms,rank_ratio_range=[min(v['median_round_ratio']for v in rankrows),max(v['median_round_ratio']for v in rankrows)],rounds=rounds),indent=2))
