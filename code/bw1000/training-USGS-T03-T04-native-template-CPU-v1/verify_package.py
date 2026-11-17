"""Pure-stdlib source closure; no model/data/library imports."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    m = json.loads((D / 'SOURCE_FILES.json').read_bytes())
    actual = {p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and p.name != 'SOURCE_FILES.json'}
    assert set(m['files']) == actual
    for name, pin in m['files'].items():
        p = D / name; raw = p.read_bytes()
        assert not p.is_symlink() and len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256']
        if p.suffix == '.py': ast.parse(raw, filename=name)
    p = json.loads((D / 'PLAN.json').read_bytes()); c = json.loads((D / 'CLAIM.json').read_bytes())
    assert len(json.loads((D / 'FRAMEWORK_BINDING.json').read_bytes())['files']) == 396
    assert c['budgets_seconds']['single_origin_total'] == 300 and c['budgets_seconds']['native_work_deadline_offset'] == 260
    assert set(p['cases']) == {'T03.sft', 'T04.sft'} and len(p['media']) == 11
    assert p['cases']['T03.sft']['metadata_files'] == 13 and p['cases']['T04.sft']['metadata_files'] == 14
    assert c['input']['stage11_PNG_bytes'] == sum(pin['bytes'] for pin in p['media'].values()) == 375733
    return dict(passed=True, source_files=len(actual), existing_framework_members=396,
                source_only=True, actual_processor=False, actual_Torch_import=False, actual_GPU=False)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
