from pathlib import Path
import ast,hashlib,json,shutil
r=Path(__file__).resolve().parent;b=r/'Kimi-crossover-Model-private-r9026';b.mkdir(exist_ok=False)
old=r/'Kimi-ordinary-pair-private-r9007';driver=r/'Kimi-crossover-private-r9008';taps=r/'Kimi-target-taps-private-r9022'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,s):
 p=b/n;p.write_bytes(s.replace('\r\n','\n').encode());
 if n.endswith('.py'):ast.parse(s)
 return p
main=(driver/'modern_Kimi_crossover_Model_r9008.py').read_text()
main=main.replace("_cgroup_name='Kimi-pair-Model-cgroup-r9007-'","_cgroup_name='Kimi-crossover-Model-cgroup-r9026-'")
write('modern_Kimi_crossover_Model_r9008.py',main)
run=(driver/'kimi_crossover_r9008.py').read_text().replace("'actual_gpu_calls_source':'native trace; never inferred from Python call count'","'actual_gpu_calls_source':'actual native diagnostic callback invocation count; not hardware kernel count'")
write('kimi_crossover_r9008.py',run)
shutil.copyfile(taps/'Kimi_crossover_binding_r9008.py',b/'Kimi_crossover_binding_r9008.py')
wrapper=(old/'Kimi_pair_Model_r9007_inside.sh').read_text()
last='timeout --kill-after=30s 2640 python -u /refactor/incoming/modern_Kimi_pair_Model_r9007.py'
assert wrapper.count(last)==1
wrapper=wrapper.replace(last,'export INFINILM_KIMI_TARGET_TAPS_DIR_R9022="/refactor/results/Kimi-crossover-private-r9026-${SLURM_JOB_ID}.${SLURM_STEP_ID}"\n'+last.replace('modern_Kimi_pair_Model_r9007.py','modern_Kimi_crossover_Model_r9008.py'))
write('Kimi_crossover_Model_r9026_inside.sh',wrapper)
controller=(old/'launch_Kimi_pair_Model_r9007.py').read_text().replace('r9007','r9026').replace('Kimi-TP16PP2-I192','Kimi-crossover-TP16PP2')
controller=controller.replace('Kimi-crossover-TP16PP2-manifest-r9026.json','Kimi-crossover-manifest-r9008.json')
controller=controller.replace('candidates/Kimi-ordinary-pair-r9026/CPU-build.json','candidates/Kimi-target-taps-r9022/CPU-build.json')
controller=controller.replace('Kimi_pair_Model_r9026_inside.sh','Kimi_crossover_Model_r9026_inside.sh').replace('audit_Kimi_pair_Model_r9026.py','audit_Kimi_crossover_Model_r9026.py')
assert 'Kimi-ordinary-pair-r9026' not in controller
write('launch_Kimi_crossover_Model_r9026.py',controller)
audit='''import pathlib,json,hashlib,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
name='Kimi-crossover-TP16PP2-r9026';launch=json.loads((r/(name+'-launch.json')).read_text());run=launch['run'];w=json.loads((r/(name+'-watch.json')).read_text());m=json.loads((r/'incoming/Kimi-crossover-manifest-r9008.json').read_text())
out={'diagnostic_complete':False,'Model_performance_claim':False,'speculative_serving_qualified':False,'run':run,'owners':[]}
try:
 reports=[]
 for node in range(4):
  p=r/'results'/f'kimi-crossover-r9008-{run}-node{node}.json';q=json.loads(p.read_text());reports.append(q)
  out['owners'].append({'node':node,'normal8':q.get('all8_owned_workers_shutdown_completed'),'error':q.get('error'),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
  assert q['completed'] and q['all8_owned_workers_shutdown_completed'] and not q.get('error') and not q.get('shutdown_error')
  assert q['library_sha256']==m['native_DSO_SHA256'] and q['crossover_complete'] and q['performance_claim'] is False and q['allposition_finalIDs_EOS_pass'] is False
  arms=q['crossover_arms'];assert [a['arm'] for a in arms]==list('ABCDE')
  assert all(len(a['actual_ids'])==32 and len(a['expected_A_ids'])==32 and sum(a['widths'])==32 for a in arms)
  assert all(a['real_acceptance_claim'] is False for a in arms)
  expected_calls=33+33+26+12+5
  assert len(q['crossover_calls'])==expected_calls
  assert all(z['arrays']['flags_argument']==3 and z['arrays']['count_argument']==1 for z in q['crossover_calls'])
  if node==2:
   for arm in 'ABCDE':
    calls=[z for z in q['crossover_calls'] if z['arm']==arm]
    head=[h for z in calls for owner in z['owners'] for h in owner['head'] if owner['node']==2]
    assert {h['input_position'] for h in head}=={536,537},(arm,head)
    traces=[v for z in calls for owner in z['owners'] for v in owner['trace_after']['records'] if owner['node'] in (0,2)]
    assert {536,537}<={p for v in traces for p in v['positions']}
    assert {19,37,66,78,90}<={v['layer'] for v in traces if v['point']=='decoder'}
 assert all(q['crossover_arms']==reports[0]['crossover_arms'] for q in reports)
 assert w['post_KFD_passed'] and w['terminal']['State']=='COMPLETED' and w['terminal']['ExitCode']=='0:0'
 handle=w['terminal']['JobID'];logs=[]
 for node in range(4):
  p=r/'logs'/f'Kimi-crossover-TP16PP2-{run}-{handle}-{node}.log';raw=p.read_bytes();s=raw.decode(errors='replace')
  assert 'Using network IBext_v8' in s and 'Using network Socket' not in s
  assert not any(re.search(r'rocBLAS.*(?:error|709)|Traceback \\(most recent|HIP error|Segmentation fault|terminate called',line,re.I) for line in s.splitlines())
  logs.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()})
 out.update(diagnostic_complete=True,all32normal=True,logs=logs,arms=reports[2]['crossover_arms'],scope='Actual Math0 fixed causal crossover; finite differences retained. No EOS/replay/page255/serving/speculative throughput qualification.')
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
write('audit_Kimi_crossover_Model_r9026.py',audit)
m=json.loads((old/'Kimi-TP16PP2-I192-manifest-r9007.json').read_bytes())
files=['modern_Kimi_crossover_Model_r9008.py','kimi_crossover_r9008.py','Kimi_crossover_binding_r9008.py','Kimi_crossover_Model_r9026_inside.sh','launch_Kimi_crossover_Model_r9026.py','audit_Kimi_crossover_Model_r9026.py']
m.update(revision='r9026',status='SOURCE_READY_WAIT_ACTUAL_RUNTIME_FIXTURE_AND_ROOT_GO',runtime_files=files,source_sha256={n:sha(b/n) for n in files},native_DSO_SHA256='2caae7170e840a9cf51c0e32758c6b093c6a6160a8897aea15b8b1fa0826d3e1',native='/refactor/candidates/Kimi-target-taps-r9022/libKimi_target_taps_r9022.so',scope='Single loaded currentTP16PP2 Math0 B1 fixedA-E causal crossover plus realfivefeatures. Finite differences collected; no speculative acceptance/serving/performance qualification.',root_final_GO='PENDING_ACTUAL_WEIGHTLESS_FIXTURE_AND_INTEGRATION',GPU='NOT_SUBMITTED')
# Retain only meaningful current parent provenance; old experimental plans are not this run.
for key in ['ordinary_clock','time_budget_minutes','root_adjustment','root_final_GO_old','frozen_original_reference','derived_capacity_evidence','shared_actual_caller_smoke']:
 m.pop(key,None)
m['time_budget_minutes']={'observed_675_weightload_seconds':[1506.4325,1470.7886,1597.9993,1622.3224],'diagnostics_max_seconds':600,'drain_seconds':180,'hard_Slurm':45,'inside_seconds':2640}
m['CPU_receipt_SHA256']='CPU241 linkednative and actual binding startup rechecks object/sourceSHA;runtime fixture pending'
write('Kimi-crossover-manifest-r9008.json',json.dumps(m,indent=2)+'\n')
print(json.dumps({'files':files,'manifest_SHA256':sha(b/'Kimi-crossover-manifest-r9008.json'),'Model_GO':False}))
