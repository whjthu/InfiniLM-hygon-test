"""New controller AST/epoch adaptation only; no join/provider/process tests repeated."""
import ast
import hashlib
import json
from pathlib import Path
import run_CPU_once as controller

D = Path(__file__).resolve().parent


def main():
    valid = []; rejected = []
    for now in [1000.0, 1001.0, 1250.0]:
        clock = controller.schedule(1000.0, now)
        assert clock['work_deadline'] == 1260 and clock['supervisor_stop'] == 1270
        assert clock['held_cleanup_deadline'] == 1282 and clock['registry_cleanup_deadline'] == 1285
        assert clock['alarm_deadline'] == 1290 and clock['client_deadline'] == 1295 and clock['deadline'] == 1300
        valid.append(now)
    for epoch, now in [(1000.0, 1260.0), (1000.0, 999.0), (float('nan'), 1001.0)]:
        try: controller.schedule(epoch, now)
        except ValueError: rejected.append([str(epoch), now])
        else: raise AssertionError('expired or invalid original epoch accepted')
    tree = ast.parse((D / 'run_CPU_once.py').read_bytes())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(x and x.startswith(('torch', 'swift', 'safetensors')) for x in imports)
    assert not any(alias.name.startswith(('torch', 'swift', 'safetensors')) for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names)
    popens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen']
    assert len(popens) == 1 and any(k.arg == 'start_new_session' and isinstance(k.value, ast.Constant) and k.value.value is True for k in popens[0].keywords)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    anchor_save = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'save'
                       and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'held-child-anchor.json')
    snapshots = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'snapshot']
    assert anchor_save.lineno < min(n.lineno for n in snapshots)
    helpers = {name:hashlib.sha256((D / name).read_bytes()).hexdigest() for name in ['owned_processes.py', 'held_child.py']}
    return dict(passed=True, valid_epoch_cases=len(valid), rejected_epoch_cases=len(rejected),
                held_anchor_persisted_before_snapshot=True, one_new_session_Popen=True,
                no_controller_provider_import=True, helpers=helpers,
                actual_child_started=False, actual_Torch_import=False, actual_GPU=False,
                original_join_and_cleanup_tests_rerun=False)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
