"""Only package bytes/AST/adoption; no model, device, original data or network."""
import ast
import hashlib
import json
from pathlib import Path
import re

D = Path(__file__).resolve().parent


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pin(path, expected):
    need(path.is_file() and not path.is_symlink(), 'REGULAR_SOURCE')
    raw = path.read_bytes(); need(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['SHA256'], 'SOURCE_PIN')
    if path.suffix == '.py':
        ast.parse(raw)
    return raw


def verify(budget=lambda: None):
    mf = json.loads((D / 'SOURCE_FILES.json').read_bytes())['files']
    for name, expected in mf.items():
        budget()
        pin(D / name, expected)
    actual = {p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    need(actual == set(mf) | {'SOURCE_FILES.json'}, 'EXACT_OUTER_INVENTORY')
    original = (D / 'CORE_ORIGINAL_FILES.json').read_bytes()
    plan = json.loads((D / 'INSTALL_PLAN.json').read_bytes()); claim = json.loads((D / 'CLAIM.json').read_bytes())
    need(hashlib.sha256(original).hexdigest() == plan['core_original_manifest_SHA256'] ==
         '722cf0ea6e2fec32bbd94b58121b8cf20477a5459096971ba5043025051a3410', 'ORIGINAL_FROZEN_CORE')
    core_mf = (D / 'core/SOURCE_FILES.json').read_bytes()
    need(hashlib.sha256(core_mf).hexdigest() == plan['core_manifest_SHA256'], 'NESTED_MANIFEST_PIN')
    for name, expected in json.loads(core_mf)['files'].items():
        budget()
        pin(D / 'core' / name, expected)
    for name, expected in json.loads(original)['files'].items():
        budget()
        if name not in ['PLAN.json','CLAIM.json']:
            pin(D / 'core' / name, expected)
        else:
            restored = (D / 'core' / name).read_bytes()
            # Compare the original exact source, including its Windows CRLF bytes.
            # Only these two boolean lexemes may differ after root adoption.
            for key in [b'root_GO', b'GPU_GO']:
                restored, count = re.subn(rb'("' + key + rb'"\s*:\s*)true\b', rb'\1false', restored)
                need(count in (0, 1), 'ONE_CORE_GO_FIELD')
            need(len(restored) == expected['bytes'] and hashlib.sha256(restored).hexdigest() == expected['SHA256'],
                 'CORE_ONLY_GO_ADOPTION_DELTA_ALLOWED')
    flags = []
    for x in [plan, claim, json.loads((D / 'core/PLAN.json').read_bytes()), json.loads((D / 'core/CLAIM.json').read_bytes())]:
        flags += [x['root_GO'], x['GPU_GO']]
    need(all(type(x) is bool and x == flags[0] for x in flags), 'EIGHT_MATCHING_ADOPTION_FLAGS')
    need(not claim['Model_GO'] and not claim['formal17'] and claim['max_invocations'] == 1, 'ONE_INDEPENDENT_DIAGNOSTIC')
    q = D / 'qualified/bounded_GPU_supervisor.py'
    need(hashlib.sha256(q.read_bytes()).hexdigest() == plan['qualified_driver_SHA256'] ==
         '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'ACTUAL_RSS_V2_IMPORTED_BODY')
    need(plan['expected_GPU']['total_MiB'] == 40960 and plan['expected_GPU']['minimum_free_MiB'] == 38000,
         'ACTUAL_A100_40GB_ADMISSION')
    return dict(passed=True, files=len(mf), root_GO=flags[0], GPU_GO=flags[0],
        source_manifest_SHA256=hashlib.sha256((D / 'SOURCE_FILES.json').read_bytes()).hexdigest(),
        actual_model=0, actual_remote=0, actual_GPU=0)


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
