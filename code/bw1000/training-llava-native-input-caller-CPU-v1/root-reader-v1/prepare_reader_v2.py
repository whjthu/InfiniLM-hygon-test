"""Correct only the independent reader's fingerprint schema assumption."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
s=(D/'read_original_terminal.py').read_text(encoding='utf8')
old="assert collated['input_ids']['shape']==collated['labels']['shape']==[1,length] and collated['input_ids']['elements']==collated['labels']['elements']==length"
new="assert all(collated[n]['tensors']==1 and collated[n]['elements']==length and collated[n]['floating_elements']==0 for n in ('input_ids','labels'))"
assert old in s
s=s.replace(old,new).replace('root-reader','root-reader-v2');ast.parse(s)
assert not (D/'read_original_terminal_v2.py').exists();(D/'read_original_terminal_v2.py').write_bytes(s.encode())
v=dict(original_reader_SHA256=hashlib.sha256((D/'read_original_terminal.py').read_bytes()).hexdigest(),original_reader_transport_rc=1,original_reader_stderr_SHA256=hashlib.sha256((D/'root-reader.stderr.private.txt').read_bytes()).hexdigest(),cause='Reader expected per-tensor shape field,but shared fingerprint schema records SHA/tensor count/elements/finite. Original CPU source separately asserts exact1xlength shapes before fingerprinting.',fix='Use actual aggregate fingerprint schema: one integer tensor,length elements,zero floating elements,valid finiteSHA. Do not fabricate missing shape fields or repeat native encoding.',model_rerun=False,GPU_launch=False,local_first_preparation_shell_quoting_failed=True,local_first_v2_invocation_missing_file=True,remote_work_executed_by_failed_local_preparation=False)
(D/'root-reader-schema-correction.json').write_bytes((json.dumps(v,indent=2)+'\n').encode())
print(json.dumps(dict(v2_AST=True,failed_reader_retained=True,model_rerun=False)))
