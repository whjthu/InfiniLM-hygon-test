"""Static local closure only; never imports endpoint/controller main or reads payloads."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
def main():
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        assert Path(name).name==name and not (D/name).is_symlink()
        raw=(D/name).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
        if name.endswith('.py'):ast.parse(raw)
    p=json.loads((D/'PLAN.json').read_bytes());c=json.loads((D/'CLAIM.json').read_bytes())
    assert set(p['members'])=={'before.pt','inputs.pt'} and sum(x['bytes'] for x in p['members'].values())==9002784
    assert p['epoch_seconds']==300 and p['max_compressed_bytes']==10485760 and p['max_decompressed_bytes']==9068328
    assert p['original_request_id']==c['original_request_id']=='694070604bdb409db0eb9bf76f0e9f0c'
    assert p['target']==c['target']=='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-first-step-replay-a100-v1'
    flags=[c[x] for x in ('root_GO','CPU_GO','transfer_GO')]
    assert flags in ([False,False,False],[True,True,True]) and c['GPU_GO'] is c['Model_GO'] is False
    assert c['original_state']=='UNKNOWN' and c['original694_replay_allowed'] is False and c['max_invocations']==1
    if flags[0]:assert c['explicit_user_exception_received'] is True and c['exception_record_path'] and c['exception_record_SHA256']
    else:assert c['explicit_user_exception_received'] is False and c['exception_record_path'] is c['exception_record_SHA256'] is None
    proof=json.loads((D/'SOURCE_PROOF.json').read_bytes())
    tree=ast.parse((D/'reuse.py').read_bytes())
    for name,pin in proof['exact_reused_function_AST_SHA256'].items():
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==pin
    assert hashlib.sha256((D/'nice_policy.py').read_bytes()).hexdigest()=='c79a6e307ac1368a324a56abe8ba1d9dd67a036c222ca4d459ae21bc8b256c10'
    atomic=ast.parse((D/'atomic_files.py').read_bytes())
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='link' for n in ast.walk(atomic))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('replace','rename') for n in ast.walk(atomic))
    for source in ('endpoint.py','atomic_files.py','reuse.py','permission_gate.py','nice_policy.py'):
        parsed=ast.parse((D/source).read_bytes())
        for n in ast.walk(parsed):
            if isinstance(n,ast.Import):assert all(x.name.split('.')[0] not in ('torch','transformers','safetensors','pickle') for x in n.names)
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in ('torch','transformers','safetensors','pickle')
    print(json.dumps(dict(passed=True,members=len(mf['files']),exact_reused_AST_functions=len(proof['exact_reused_function_AST_SHA256']),
        user_exception_received=c['explicit_user_exception_received'],transfer_GO=c['transfer_GO'],network_calls=0,PT_reads=0)))
if __name__=='__main__':main()
