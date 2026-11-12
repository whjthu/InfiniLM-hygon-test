"""Verify the actual new collector blocks and unchanged qualified lifecycle."""
from pathlib import Path
import ast,copy,hashlib,json,sys,time
D=Path(__file__).resolve().parent;B=D.parent;P=D/'package';old=B/'training-minicpm-native-first-tail-entry-v1/package';F=B/'hygon-training-integration'
for p in P.glob('*.py'):ast.parse(p.read_bytes())
for name in ('controller.py','receive_once.py'):
 expected=(old/name).read_text(encoding='utf8').replace('training-minicpm-native-first-tail-entry-v1',D.name)
 assert (P/name).read_text(encoding='utf8')==expected
for name in ('snapshot.py','owned_processes.py','slurm_accounting.py'):assert (P/name).read_bytes()==(old/name).read_bytes()
s=(P/'scope.py').read_text(encoding='utf8');ns=dict(__file__=str(P/'scope.py'));exec(compile(s,'new_exact_scope','exec'),ns)
for offset in (0.,30.,60.,90.,179.9,180.):
 minutes=ns['gpu_minutes'](1000.,1000.+offset);assert 1000.+offset+minutes*60>1000.+ns['INSIDE_SECONDS']+ns['KILL_SECONDS'] and 1000.+offset+minutes*60<=2500.
# Exercise the actual new collector guard using bounded, explicitly synthetic
# metadata files. No real activations, checkpoints, model or device is touched.
sys.path.insert(0,str(F));from vlm.runtime.forward_boundary_trace import compare_boundary_traces
fixture=D/'local-boundary-fixtures';fixture.mkdir(exist_ok=False);results=[]
native=(P/'native.py').read_text(encoding='utf8');block=native[native.index("   boundary=root/'forward-boundaries.json'"):native.index("   if owner.is_file()",native.index("   boundary=root/'forward-boundaries.json'"))];block='\n'.join(line[3:] if line.startswith('   ') else line for line in block.splitlines())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def trace(ordinal=41):
 names=['fixture-boundary-%02d'%i for i in range(52)]
 return dict(schema=1,complete=True,failed=False,raw_values_exported=False,observer_RNG_unchanged=True,global_step=20,forward_ordinal=ordinal,planned_count=52,completed_boundaries=52,planned_boundaries=names,copied_logical_bytes=208,max_logical_bytes=2*1024**3,observer_seconds=.1,total_wall_seconds=.2,records=[dict(name=n,ordinal=i+1,value_SHA256=hashlib.sha256(n.encode()).hexdigest(),observer_RNG_unchanged=True) for i,n in enumerate(names)])
def collect(label,mutate,passed):
 root=fixture/label;root.mkdir();p=root/'forward-boundaries.json';v=trace();mutate(v);p.write_text(json.dumps(v));path=root/'result.json';path.write_text('{}');item={};value=dict(boundary_trace_SHA256=sha(p))
 success=True
 try:exec(compile(block,'actual_new_native_boundary_guard','exec'),dict(root=root,path=path,value=value,item=item,name='control',sha=sha,json=json))
 except (AssertionError,ValueError):success=False
 assert success==passed,(label,success);results.append(dict(case=label,passed=True,expected_acceptance=passed,actual_acceptance=success))
collect('complete52',lambda v:None,True)
collect('wrong-first-ordinal',lambda v:v.update(forward_ordinal=1),False)
collect('missing-boundary',lambda v:v['records'].pop(),False)
collect('incomplete',lambda v:v.update(complete=False),False)
collect('wrong-step',lambda v:v.update(global_step=21),False)
collect('RNG-changed',lambda v:v.update(observer_RNG_unchanged=False),False)
collect('raw-export',lambda v:v.update(raw_values_exported=True),False)
collect('copy-budget-exceeded',lambda v:v.update(copied_logical_bytes=3*1024**3),False)
a,b=trace(),trace(1);equal=compare_boundary_traces(a,b);assert equal['equal_boundaries']==52
b['records'][17]['value_SHA256']='f'*64;different=compare_boundary_traces(a,b);assert different['first_value_difference']==dict(ordinal=18,name='fixture-boundary-17') and not different['operator_cause_proven']
assert 2*8*52==832
v=dict(passed=True,native_collector_cases=results,comparator_equal_full52=True,first_difference_fixture=True,planned_traces=16,planned_boundary_records=832,qualified_controller_receiver_snapshot_ownership_accounting_unchanged=True,new_scope_epoch_budget_examples=6,actual_native_model_runs=0,actual_GPU_launches=0,qualification='Local stdlib syntheticmetadata/newcollector and unchanged lifecycle checks only;not real52native boundaries or actual device/CPU model qualification.')
with (D/'local-checks.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(dict(passed=True,native_collector_cases=len(results),planned_records=832,actual_GPU_launches=0)))
