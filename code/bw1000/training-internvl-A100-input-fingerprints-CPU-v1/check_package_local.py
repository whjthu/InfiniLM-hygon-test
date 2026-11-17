"""Local AST/bytes/new budget boundaries only; no producer, Torch, SDK or SSH."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes())
for p in D.rglob('*.py'):ast.parse(p.read_bytes())
canonical=(D/'canonical/first_update_audit.py').read_bytes()
assert len(canonical)==6939 and hashlib.sha256(canonical).hexdigest()==P['fingerprint_source_SHA256']
assert P['case']=='T03.sft' and len(P['source_receipts'])==8 and P['file_denominator']==64
assert P['work_offset_seconds']+P['reserve_for_cleanup_seconds']<P['whole_seconds']
spec=importlib.util.spec_from_file_location('budget_only',D/'common.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
with patch.object(M.time,'time',return_value=1000),patch.object(M.time,'monotonic',return_value=10):b=M.Budget(1000)
with patch.object(M.time,'time',return_value=1010),patch.object(M.time,'monotonic',return_value=20):
    assert b.remaining(work=True)==770 and b.remaining()==890
with patch.object(M.time,'time',return_value=900),patch.object(M.time,'monotonic',return_value=790):
    try:b.remaining(work=True)
    except ValueError:pass
    else:raise AssertionError('rollback extended work budget')
with patch.object(M.time,'time',return_value=1000),patch.object(M.time,'monotonic',return_value=910):
    try:b.remaining()
    except ValueError:pass
    else:raise AssertionError('rollback extended whole epoch')
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'small.json';M.save(p,dict(schema='local-budget-check'))
    assert json.loads(p.read_bytes())==dict(schema='local-budget-check') and p.read_bytes().endswith(b'\n')
    try:M.unique(b'{"x":1,"x":2}')
    except ValueError:pass
    else:raise AssertionError('duplicate metadata accepted')
worker=ast.parse((D/'derive_inputs.py').read_bytes())
main=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='main')
text=ast.unparse(main)
assert text.index('resolve_all(')<text.index('import torch')<text.index('load_original(')
assert 'weights_only=True' in (D/'derive_inputs.py').read_text() and "map_location='cpu'" in (D/'derive_inputs.py').read_text()
launcher=ast.parse((D/'launch_once.py').read_bytes())
assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and '\\nREQUEST=' in n.value for n in ast.walk(launcher))
spec=importlib.util.spec_from_file_location('new_client_only',D/'launch_once.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
args=L.cli(['--publication','a'*40,'--source-manifest-sha','b'*64,'--claim-sha256','c'*64])
assert args.publication=='a'*40
for bad in ('x'*40,'a'*39):
    try:L.cli(['--publication',bad,'--source-manifest-sha','b'*64,'--claim-sha256','c'*64])
    except ValueError:pass
    else:raise AssertionError('bad publication accepted')
claim=json.loads((D/'CLAIM.json').read_bytes());L.check_claim(claim)
for key,value in [('CPU_GO',False),('GPU_GO',True),('target_GO',True),('maximum_CPU_invocations',2),('maximum_CPU_invocations',True),('case','T04.sft'),('LLaVA_included',True)]:
    bad=dict(claim);bad[key]=value
    try:L.check_claim(bad)
    except ValueError:pass
    else:raise AssertionError('bad claim accepted: '+key)
client=(D/'launch_once.py').read_text()
assert "git('show',args.publication+':'+relative+'/SOURCE_FILES.json')" in client
assert "git('show',args.publication+':'+relative+'/CLAIM.json')" in client
assert "git('status','--porcelain=v1','--untracked-files=all','--',relative)" in client
assert 'source-publication-receipt' not in client and "save(D/" not in client
assert "remote='/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -'" in client
assert L.LOCAL.parent.name=='local-only' and L.LOCAL!=D
assert 'torch' not in sys.modules and 'swift' not in sys.modules
print(json.dumps(dict(passed=True,canonical_exactbytes=True,all64pins_before_Torch_and_load=True,
    new_budget_rollback_expiry_boundaries=True,metadata_save_and_duplicate_gate=True,
    new_CLI_pin_and_claim_negative_boundaries=True,Git_calls=0,Torch_imports=0,SSH_calls=0,actual_artifact_reads=0,actual_CPU_derivations=0)))
