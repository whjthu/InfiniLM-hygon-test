"""Correct only Python3.10/3.12 unittest descriptions; require exact methods."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
first=json.loads((D/'root-reader-result.json').read_bytes());assert first['transport_rc']==1
original=json.loads((D/'transport-outcome.json').read_bytes());cpu=original['value']['CPU-result.json']['value'];assert cpu['tests_run']==26
lines=[r for r in cpu['test_log'].splitlines() if r.startswith('test_') and r.endswith(' ... ok')];assert len(lines)==26
s=(D/'read_original_terminal.py').read_text(encoding='utf8')
needle="assert 'Ran 26 tests' in tests and tests.rstrip().endswith('OK') and len(ok)==26 and sum('BoundaryHookTest.' in r for r in ok)==8 and sum('ResumeWitnessTest.' in r for r in ok)==18"
assert needle in s
replacement='''assert 'Ran 26 tests' in tests and tests.rstrip().endswith('OK') and len(ok)==26
import ast,re
expected_methods=set()
for module,cls,count in [('vlm.tests.test_forward_boundary_trace','BoundaryHookTest',8),('vlm.tests.test_resume_witness','ResumeWitnessTest',18)]:
 tree=ast.parse((W/'framework'/Path(*module.split('.'))).with_suffix('.py').read_bytes())
 selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls];assert len(selected)==1
 methods={n.name for n in selected[0].body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')};assert len(methods)==count
 expected_methods.update((module+'.'+cls,name) for name in methods)
actual_methods=[]
for line in ok:
 match=re.fullmatch(r'(test_[A-Za-z0-9_]+) \\(([^)]+)\\) \\.\\.\\. ok',line);assert match
 method,description=match.groups()
 candidates=[(cls,name) for cls,name in expected_methods if name==method and description in (cls,cls+'.'+name)]
 assert len(candidates)==1;actual_methods.append(candidates[0])
assert len(actual_methods)==len(set(actual_methods))==26 and set(actual_methods)==expected_methods
'''
s=s.replace(needle,replacement).replace('root-reader-intent.json','root-reader-v2-intent.json').replace('root-reader.stdout.private.json','root-reader-v2.stdout.private.json').replace('root-reader.stderr.private.txt','root-reader-v2.stderr.private.txt').replace('root-reader-result.json','root-reader-v2-result.json')
ast.parse(s);(D/'read_original_terminal_v2.py').write_bytes(s.encode())
correction=dict(first_reader_transport_rc=1,first_reader_receipt_SHA256=hashlib.sha256((D/'root-reader-result.json').read_bytes()).hexdigest(),first_stderr_SHA256=hashlib.sha256((D/'root-reader.stderr.private.txt').read_bytes()).hexdigest(),first_failed_predicate='Unittest class-dot string counts assumed Python3.12 description format;actualPython3.10 has class-only parentheses.',actual_success_log_records=26,actual_original_CPU_passed=True,change='Resolve exact8+18testmethodinventory from353frozenAST,accept only exact class or class.method unittest representations with matching methodname,require26uniqueallmethods.',original_tests_replayed=False,original_Slurm_step=original['value']['accounting'][0][0],original_raw_logs_unchanged=True)
with (D/'root-reader-format-correction.json').open('x') as f:json.dump(correction,f,indent=2)
print(json.dumps(dict(prepared_reader_v2=True,original_test_records=26,new_CPU_runs=0,GPU_cards=0)))
