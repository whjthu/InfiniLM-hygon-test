"""Only strict zero-domain/case canonical equivalence and invalid domain rejection."""
import ast
import json
from pathlib import Path
import re

D = Path(__file__).resolve().parent


def main():
    tree = ast.parse((D/'read_original.py').read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'need','canonical_PCI'}]
    scope = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-strict-PCI', 'exec'), scope)
    canonical = scope['canonical_PCI']
    assert canonical('00000000:A2:00.0') == canonical('0000:a2:00.0') == '0000:a2:00.0'
    try:
        canonical('00010000:A2:00.0')
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range-domain-accepted')
    print(json.dumps(dict(passed=True, equivalent_pair=1, invalid_domain_rejected=1, remote_calls=0)))


if __name__ == '__main__':
    main()
