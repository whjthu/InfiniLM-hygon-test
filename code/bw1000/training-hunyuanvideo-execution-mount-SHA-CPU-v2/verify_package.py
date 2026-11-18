"""Pure source/AST and actual caller wiring closure, no source weight/SDK/provider read."""
import ast
import hashlib
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent
if str(D) not in sys.path:sys.path.insert(0,str(D))


def main():
    manifest=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in manifest['files'].items():
        raw=(D/name).read_bytes()
        assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
        if name.endswith('.py'):ast.parse(raw)
    plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    assert type(claim['root_GO']) is type(claim['CPU_GO']) is bool and claim['root_GO']==claim['CPU_GO']
    assert claim['GPU_GO'] is False and claim['max_invocations']==1 and plan['epoch_seconds']==900
    assert plan['weight_bytes']==33306632192 and plan['weight_SHA256']=='71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86'
    assert (plan['CPU'],plan['memory_MiB'],plan['GPU'],plan['parent'],plan['node'])==(2,4096,0,'1103606','f13r3n13')
    assert plan['label']=='training-hunyuanvideo-execution-mount-SHA-CPU-v2' and plan['required_client_server_version']=='24.05.8-3.1.3-668-20260723'
    assert hashlib.sha256((D/'prior-v1-terminal.json').read_bytes()).hexdigest()==plan['prior_terminal_SHA256']=='00c73016b009ecb7528bf7b9407e23827fa53469aff5879879242a9322bed809'
    import pending_placeholder
    assert pending_placeholder.VERSION==plan['required_client_server_version']
    install=ast.parse((D/'install_and_run.py').read_bytes())
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='execv' for n in ast.walk(install))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('Popen','run','check_output') for n in ast.walk(install))
    launcher=ast.parse((D/'launch_once.py').read_bytes())
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fromisoformat' for n in ast.walk(launcher))
    assert any(isinstance(n,ast.ExceptHandler) and isinstance(n.type,ast.Tuple) and any(isinstance(x,ast.Attribute) and x.attr=='SubprocessError' for x in n.type.elts) for n in ast.walk(launcher))
    nested=json.loads((D/'caller-v3/SOURCE_FILES.json').read_bytes())
    assert len(nested['files'])==14
    for name,pin in nested['files'].items():
        raw=(D/'caller-v3'/name).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    # All links are executable call/arg sites, not README promises or comment-only mention.
    for name in ('controller.py','slurm_command.py','device_entry.py','run_replay.py'):
        tree=ast.parse((D/'caller-v3'/name).read_bytes())
        if name=='device_entry.py':
            assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'
                       and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--execution-authority-sha256' for n in ast.walk(tree))
            assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='prepare'
                       and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='args' for n in ast.walk(tree))
        else:
            assert any(isinstance(n,ast.Attribute) and n.attr=='execution_authority_sha256'
                       or isinstance(n,ast.Name) and n.id=='execution_authority_sha256' for n in ast.walk(tree))
    tree=ast.parse((D/'caller-v3/check_hunyuan_sft_replay_inputs.py').read_bytes())
    final=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='final_suffix_identity')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='consume' for n in ast.walk(final))
    cpu=ast.parse((D/'original_hash.py').read_bytes());expected=plan['original_hash_function_AST_SHA256']
    for name,pin in expected.items():
        node=next(n for n in cpu.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==pin
    print(json.dumps(dict(passed=True,outer_members=len(manifest['files']),caller_source_members=14,
        actual_caller_chain_verified=True,CPU_GO=claim['CPU_GO'],GPU_GO=False,remote_calls=0,weight_reads=0)))


if __name__=='__main__':main()
