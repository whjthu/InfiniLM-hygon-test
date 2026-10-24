"""Read once by unique terminal watcher; no Engine, resubmit or extra KFD."""
import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
run=json.loads((r/'Kimi-CCL512-r8607-launch.json').read_text())['run']
w=json.loads((r/'Kimi-CCL512-r8607-watch.json').read_text())
out=dict(run=run,scope='Uniform512 only; unsupported/mixedvariablelength fallback remains unqualified',
 generation_acceptance='NOT_MEASURED',owners=[],all32normal=False,model_qualified=False,
 terminal=w['terminal'],post_KFD_passed=w['post_KFD_passed'])
try:
 reports=[]
 for n in range(4):
  p=r/'results'/f'kimi-CCL512-r8607-{run}-node{n}.json';q=json.loads(p.read_text());reports.append(q)
  out['owners'].append(dict(node=n,completed=q['completed'],normal8=q.get('all8_owned_workers_shutdown_completed',False),
   report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),candidate_qualified=q.get('candidate_qualified',False),
   branch_counts=q.get('final_prefill512_counts'),CCL512_counts=q.get('final_CCL512_counts'),error=q.get('error'),shutdown_error=q.get('shutdown_error')))
 out['all32normal']=all(v['normal8']for v in out['owners'])
 out['generation_acceptance']='QUALIFIED_UNIFORM512_N8_DECODE_M4_ONLY'if all(v['candidate_qualified']for v in out['owners'])else'FAILED_OBSERVED'
 q=reports[0]
 out.update(QA=q.get('QA'),static_final_IDs_equal=q.get('static_final_IDs_equal'),baseline_IDs_SHA256=q.get('baseline_IDs_SHA256'),
  phase_timings=q.get('phase_timings'),loading_seconds_by_owner=[v.get('loading_seconds')for v in reports],
  historical618_exact_O8=q.get('historical618_exact_O8'),paired_timing=q.get('paired_timing'),timing_final_IDs_equal=q.get('timing_final_IDs_equal'),bounded_remaining_interval=q.get('bounded_remaining_interval'),AUX0=q.get('AUX0_independent_arm'),service_boundaries='Ordinary staticN8/decodeM4 same process clocks; node0 coordinator feedback only, client NOT_MEASURED; ONNO twice if complete IDs/EOS pass; no online or B64 claim')
 assert out['all32normal']and all(v['completed']for v in out['owners'])and out['post_KFD_passed']
 assert out['terminal']['State']=='COMPLETED'and out['terminal']['ExitCode']=='0:0'
 out['model_qualified']=out['generation_acceptance']=='QUALIFIED_UNIFORM512_N8_DECODE_M4_ONLY'
except BaseException as error:out['error']=repr(error);raise
finally:
 p=r/'results'/f'Kimi-CCL512-r8607-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2))
 print(json.dumps(dict(run=run,generation_acceptance=out['generation_acceptance'],all32normal=out['all32normal'],model_qualified=out['model_qualified'],error=out.get('error'))))
