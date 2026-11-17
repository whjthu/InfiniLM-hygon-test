"""Only new source-review/adopted and mixed-GO boundaries, no previous fixtures."""
import ast
import copy
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    tree = ast.parse((D/'verify_package.py').read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'adoption_gate']
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-adoption-gate', 'exec'), scope)
    gate = scope['adoption_gate']
    rows = [json.loads((D/name).read_bytes()) for name in ('CONFIG.template.json', 'CLAIM.json', 'supervisor/CLAIM.json')]
    assert gate(*rows) == 'source_review'
    adopted = copy.deepcopy(rows)
    for row in adopted:
        row.update(root_GO=True, GPU_GO=True)
    assert gate(*adopted) == 'root_adopted'
    rejected = []
    for index, key in ((0, 'GPU_GO'), (1, 'root_GO'), (2, 'GPU_GO')):
        bad = copy.deepcopy(adopted); bad[index][key] = False
        try:
            gate(*bad)
        except AssertionError:
            rejected.append('mixed-'+str(index)+'-'+key)
        else:
            raise AssertionError('mixedGO allowed')
    print(json.dumps(dict(passed=True, valid=2, rejected=rejected, rejected_count=len(rejected), remote_calls=0)))


if __name__ == '__main__':
    main()
