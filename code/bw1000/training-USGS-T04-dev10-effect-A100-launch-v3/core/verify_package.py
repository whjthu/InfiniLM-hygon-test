"""Static own-package verifier; no device/provider imports or original data reads."""
import ast
import json
from pathlib import Path
from effect_contract import need, pinned, sha

D = Path(__file__).resolve().parent


def verify():
    raw = (D / 'SOURCE_FILES.json').read_bytes()
    manifest = json.loads(raw)
    for name, pin in manifest['files'].items():
        body = pinned(D / name, pin)
        if name.endswith('.py'):
            ast.parse(body)
    actual = {p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    need(actual == set(manifest['files']) | {'SOURCE_FILES.json'}, 'EXACT_SOURCE_INVENTORY')
    plan = json.loads((D / 'PLAN.json').read_bytes()); claim = json.loads((D / 'CLAIM.json').read_bytes())
    need(type(plan['root_GO']) is bool and plan['root_GO'] == plan['GPU_GO'] == claim['root_GO'] == claim['GPU_GO'], 'ADOPTION_FLAGS')
    need(not plan['Model_GO'] and not claim['Model_GO'] and not plan['formal17'] and not claim['formal17'], 'NO_FORMAL_MODEL_CLAIM')
    need(plan['generation']['max_new_tokens'] == 256 and plan['dev_inputs']['rows'] == 10, 'FROZEN_INPUT_AND_CAP')
    need(plan['generic25'] is not None and plan['generic25']['initial_named_SHA256'] == plan['initial_named_SHA256'], 'ACTUAL_PAIRED_GENERIC_INITIAL')
    return dict(passed=True, files=len(manifest['files']), source_manifest_SHA256=sha(raw),
        root_GO=plan['root_GO'], GPU_GO=plan['GPU_GO'], actual_encode=0, actual_model=0)


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
