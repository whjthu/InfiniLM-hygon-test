"""Publishable metadata only; consume existing immutable terminal evidence."""
from pathlib import Path
import json,hashlib,datetime,statistics
D=Path(__file__).resolve().parent;G=D.parents[2];E=G/'results/bw1000/2026-11-02'
path=E/'GLM361-r1-final-audit.json';a=json.loads(path.read_bytes())
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()=='11cf0416e344e0fe8795c809c2cf54a87588f9dc6c8093a8d203749068deed8e'
assert hashlib.sha256((D/'runtime-r1/owner/SOURCE_MANIFEST.json').read_bytes()).hexdigest()==a['owner_manifest_SHA256']
assert hashlib.sha256((D/'runtime-r1/controller/SOURCE_MANIFEST.json').read_bytes()).hexdigest()==a['controller_manifest_SHA256']
assert a['completed']and a['passed']and a['all12_independent_wire_gold']and a['all32_normal_and_signatures_equal']and a['all5_original_handles_COMPLETED0']and a['all4_srun_exit0']and a['all_independent_post_passed']and a['independent_cost_reader_equal']
mean=lambda values:statistics.mean(values)
metrics={}
for row,fields in a['primary_full_request_metrics'].items():
 metrics[row]={k:{**v,'ON_vs_OFF_fraction':v['ON_mean']/v['OFF_mean']-1}for k,v in fields.items()}
goodput={field:{'OFF':[a['emitter'][i][field]for i in [2,5]],'ON':[a['emitter'][i][field]for i in [3,4]]}for field in ['total_tok_s_per_all32cards','decode_gt15_goodput_tok_s_per_all32cards']}
for field,values in goodput.items():
 values['OFF_mean']=mean(values['OFF']);values['ON_mean']=mean(values['ON'])
 if values['OFF_mean']:values['ON_vs_OFF_fraction']=values['ON_mean']/values['OFF_mean']-1
http={'OFF_mean_complete_pair_s':mean(a['HTTP_trials'][i]['complete_pair_wall_s']for i in [2,5]),'ON_mean_complete_pair_s':mean(a['HTTP_trials'][i]['complete_pair_wall_s']for i in [3,4])}
q=dict(schema=1,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,passed=True,
 decision='SCOPED_MODEL_TAIL_LATENCY_TRADEOFF_NO_UNCONDITIONAL_PROMOTION',
 scope='One immutable 32card TP8PP4 C2M1 residency; short479 read_file plus anchor8900 calculate; two warm pairs then formal ABBA,12 complete HTTP requests; fixed deferreddecodeON, only completed-long-W4-boundary interleave changes',
 source_commit=a['source_commit'],owner_manifest_SHA256=a['owner_manifest_SHA256'],controller_manifest_SHA256=a['controller_manifest_SHA256'],Native_SHA256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',
 unchanged='Native/weights/quantization/SDK/context32768/ordinary4096/W4-internal-order/decode-deferON',
 actual_stages=a['actual_stages'],keeper=a['keeper'],accounting=a['accounting'],all4_srun_exit0=True,all32_normal_destroy=True,all12_independent_wire_gold_and_natural_EOS=True,
 formal_OFF_indices=[2,5],formal_ON_indices=[3,4],warmup_indices=[0,1],aligned_trials=a['aligned_trials'],
 complete_pair_OFF_mean_s=a['complete_pair_OFF_mean_s'],complete_pair_ON_mean_s=a['complete_pair_ON_mean_s'],complete_pair_rate_gain_fraction=a['complete_pair_rate_gain_fraction'],OFF_endpoint_drift_fraction=a['OFF_endpoint_drift_fraction'],ON_endpoint_drift_fraction=a['ON_endpoint_drift_fraction'],
 full_request_metrics=metrics,all32_emitter_goodput=goodput,complete_buffered_HTTP=http,
 all_six_warm_and_formal_emitter_timelines=a['emitter'],all_six_HTTP=a['HTTP_trials'],
 warmup_interpretation='Warm OFF/ON retained separately; cold warm0TTFT is not formal causal gain',
 tradeoff='Reducing one long short-row gap into several W4-sized gaps can improve maxITL while worsening P99; report complete first-to-last,TTFT,long and pair costs together',
 clock='All request metrics use original same-emitter admission/token ticks; same-trial max of32 local elapsed schedule durations, no cross-node timestamp subtraction or overlapping-stage sum',
 HTTP_incremental_qualified=False,online_arrival_or_SLO_qualified=False,decode_gt15_qualified=False,broad_benchmark_quality_qualified=False,old349_or355_gain_inherited=False,remaining_decode_window_used_as_whole_gain=False,
 original_runtime_edited=False,new_GPU_or_keeper_or_Model_reload_started=False,
 post=a['post'],independent_post=a['independent_post'],all_independent_post_passed=True,
 minimum_observed_driver_free_bytes=a['all32_minimum_observed_driver_free_bytes'],memory_is_peak=False,
 audit_path=str(path.relative_to(G)).replace('\\','/'),audit_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),remote_audit_path=a['remote_terminal_audit_path'],remote_audit_SHA256=a['remote_terminal_audit_SHA256'],remote_cost_path=a['cost_path'],remote_cost_SHA256=a['cost_SHA256'],owner_reports=a['owners'],reader_SHA256=a['reader_SHA256'])
assert metrics['0']['ITL_max_s']['ON_mean']<metrics['0']['ITL_max_s']['OFF_mean']
assert metrics['0']['ITL_P99_s']['ON_mean']>metrics['0']['ITL_P99_s']['OFF_mean']
out=E/'GLM361-r1-final-decision.json';assert not out.exists();out.write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
task=G/'coordination/tasks/GLM-prefill-window-interleave-r361.json';t=json.loads(task.read_bytes())
t.update(state='COMPLETE_SCOPED_MODEL_LATENCY_TRADEOFF_NO_UNCONDITIONAL_PROMOTION',last_observed_UTC=q['UTC'],performance_gain='Short maxITL benefit with P99/whole-request tradeoff; see full metrics',max_tok_s='Strict>15 not achieved',exit='Original twelve HTTP requests complete;32 normal destroy;five original handles COMPLETED0;four independent postchecks PID/KFD/port clean; no new load or grid',terminal_result=str(out.relative_to(G)).replace('\\','/'),terminal_result_SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),all12_gold=True,all32_normal_destroy=True,all5_original_handles_COMPLETED0=True,all4_srun_exit0=True,independent_post_passed=True,new_Model_load=False)
task.write_text(json.dumps(t,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:q[k]for k in ['decision','complete_pair_OFF_mean_s','complete_pair_ON_mean_s','complete_pair_rate_gain_fraction','OFF_endpoint_drift_fraction','ON_endpoint_drift_fraction','full_request_metrics','all32_emitter_goodput','complete_buffered_HTTP']},indent=2))
