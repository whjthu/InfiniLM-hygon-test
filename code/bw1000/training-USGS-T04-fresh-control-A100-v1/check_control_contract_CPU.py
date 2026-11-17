"""Only the new generic-arm initial/data binding branches; no native/provider imports."""
import ast
import copy
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    tree = ast.parse((D/'install_and_run.py').read_bytes())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'need', 'control_template'}]
    space = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'actual-control-contract', 'exec'), space)
    call = space['control_template']
    config = json.loads((D/'CONFIG.template.json').read_bytes())
    plan = json.loads((D/'INSTALL_PLAN.json').read_bytes())
    call(config, plan, config['pair_id'])
    rejected = []
    mutations = {
        'old-domain-arm': lambda c: c.update(arm='domain'),
        'old-checkpoint-resume': lambda c: c.update(mode='resume20to25'),
        'seed-only-no-initial': lambda c: c.update(initial_state=None),
        'foreign-initial-body': lambda c: c['initial_state'].update(SHA256='0'*64),
        'missing-pair-reference': lambda c: c.update(pair_initial_reference=None),
        'wrong-pair': lambda c: c.update(pair_id='0'*32),
        'partial800': lambda c: c.update(physical_rows=799),
        'domain-input-reused': lambda c: c['training800'].update(SHA256='57640ac8c8fd9a7d0614df276af37e0c92201ec293a3ea722fd0b6252ac01570'),
    }
    for label, mutate in mutations.items():
        value = copy.deepcopy(config); mutate(value)
        try:
            call(value, plan, config['pair_id'])
        except (ValueError, TypeError):
            rejected.append(label)
        else:
            raise AssertionError(label)
    print(json.dumps(dict(passed=True, valid=1, rejected=rejected, rejected_count=len(rejected),
                          actual_native_calls=0, Torch_import=False, remote_calls=0)))


if __name__ == '__main__':
    main()
