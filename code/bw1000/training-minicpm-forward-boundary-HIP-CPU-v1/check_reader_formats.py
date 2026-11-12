"""Exercise the actual reader parser against observed3.10/3.12 test methods."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;B=D.parent
tree=ast.parse((D/'read_original_terminal_v2_retry1.py').read_bytes())
raw=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,bytes) and b'expected_methods=set()' in n.value)
text=raw.decode();begin=text.index("assert 'Ran 26 tests' in tests");end=text.index("assert result['plugin_registered",begin);block=text[begin:end]
hip=json.loads((D/'transport-outcome.json').read_bytes())['value']['CPU-result.json']['value']['test_log']
reference=json.loads((B/'training-minicpm-forward-boundary-CPU-v1/root-reader-result.json').read_bytes())['value']['test_methods']
original_rows_format='\n'.join(reference)+'\nRan 26 tests\nOK\n'
cases=[]
def run(name,log,expected):
 rows=[r for r in log.splitlines() if r.startswith('test_') and r.endswith(' ... ok')]
 success=True
 try:exec(compile(block,'actual_readonly_reader_parser','exec'),dict(W=D/'local-source-install',tests=log,ok=rows,Path=Path))
 except (AssertionError,ValueError):success=False
 assert success==expected
 cases.append(dict(case=name,expected_acceptance=expected,actual_acceptance=success,passed=True,methods=len(rows)))
run('actual-HIP-Python310-log',hip,True)
run('observed-A100-Python312-methods-with-fixture-footer',original_rows_format,True)
rows=hip.splitlines();indexes=[i for i,r in enumerate(rows) if r.startswith('test_') and r.endswith(' ... ok')]
bad=rows.copy();bad[indexes[0]]=bad[indexes[1]];run('duplicate-method', '\n'.join(bad),False)
bad=rows.copy();bad[indexes[0]]=bad[indexes[0]].replace('vlm.tests.','foreign.tests.');run('foreign-provider-description','\n'.join(bad),False)
bad=rows.copy();bad.pop(indexes[0]);run('missing-method','\n'.join(bad),False)
v=dict(passed=True,actual_remote_parser_block_SHA256=hashlib.sha256(block.encode()).hexdigest(),cases=cases,actual_CPU182_replayed=False,new_GPU_runs=0,scope='Local evidence parser checks only;3.12descriptions use observed originalCPUmethodrows with explicitfixturefooter,not a new or complete originaltestlog proof.')
with (D/'reader-format-CPU-checks.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(dict(passed=True,cases=len(cases),accepted_formats=2,rejected_bad_evidence=3,CPU182_replayed=False)))
