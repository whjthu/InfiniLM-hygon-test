"""Only the new basename/diagnostic delta: AST and bounded-source metadata."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
OLD = D.parent / 'training-USGS-T03-T04-native-template-CPU-v1'


def calls(tree, name):
    return [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def main():
    before = ast.parse((OLD / 'native_template_CPU.py').read_bytes())
    after = ast.parse((D / 'native_template_CPU.py').read_bytes())
    for name in ['get_model_processor', 'get_template']:
        assert len(calls(before, name)) == 1 and calls(before, name) == calls(after, name)
    assert calls(before, 'bind_encode') == calls(after, 'bind_encode')
    assert not any(isinstance(n, (ast.Assign, ast.AnnAssign)) and any(isinstance(t, ast.Attribute) and t.attr == 'template'
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])) for n in ast.walk(after))
    need_calls = [n for n in ast.walk(after) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'need']
    raw_guard = next(n for n in need_calls if len(n.args) > 1 and isinstance(n.args[1], ast.Constant)
                     and n.args[1].value == 'native raw processor template differs')
    resolved_guard = next(n for n in need_calls if len(n.args) > 1 and isinstance(n.args[1], ast.Constant)
                          and n.args[1].value == 'native final resolved template differs')
    assert isinstance(raw_guard.args[0], ast.Compare) and isinstance(raw_guard.args[0].comparators[-1], ast.Constant)
    assert raw_guard.args[0].comparators[-1].value == resolved_guard.args[0].comparators[-1].value == 'internvl_hf'
    saves = {n.args[0].value:n for n in ast.walk(after) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'save' and n.args and isinstance(n.args[0], ast.Constant)}
    template_call = next(n for n in ast.walk(after) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'get_template')
    assert saves['processor-declaration-private.json'].lineno < raw_guard.lineno < template_call.lineno
    assert template_call.lineno < saves['resolved-template-private.json'].lineno < resolved_guard.lineno
    plan = json.loads((D / 'PLAN.json').read_bytes()); old_plan = json.loads((OLD / 'PLAN.json').read_bytes())
    assert set(plan['cases']) == {'T03.sft'} and plan['cases']['T03.sft'] == old_plan['cases']['T03.sft']
    assert plan['media'] == old_plan['media'] and plan['inputs'] == old_plan['inputs'] and plan['train_labels'] == old_plan['train_labels']
    assert Path(plan['cases']['T03.sft']['model_metadata']['model_root']).name == 'InternVL3_5-8B-HF'
    installer = ast.parse((D / 'install_and_run.py').read_bytes())
    models = next(n.value for n in installer.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'MODELS' for t in n.targets))
    assert isinstance(models, ast.Dict) and [k.value for k in models.keys] == ['T03.sft']
    scope = next(n.value for n in ast.walk(installer) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'scope' for t in n.targets))
    assert isinstance(scope, ast.BinOp) and scope.right.value == 'training-USGS-T03-native-template-CPU-v2'
    for name in ['held_child.py','owned_processes.py','wrapper_binding.py','prepare_inputs.py','FRAMEWORK_BINDING.json']:
        assert (D/name).read_bytes() == (OLD/name).read_bytes()
    for p in D.glob('*.py'): ast.parse(p.read_bytes(), filename=p.name)
    return dict(passed=True, processor_and_template_call_AST_unchanged=True, original_raw_hf_gate_not_relaxed=True,
        final_resolved_hf_checked=True, actual_raw_and_class_record_precedes_guard=True,
        metadata13_and_all55_input_source_pins_unchanged=True, new_T03_namespace=True,
        helper_bytes_unchanged=True, actual_provider_import=False, actual_Native=False, actual_GPU=False,
        original_tests_not_rerun=True)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
