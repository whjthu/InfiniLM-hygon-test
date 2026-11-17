"""Source-only closure check; no provider import or checkpoint access."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    manifest = json.loads((D / 'SOURCE_FILES.json').read_bytes())
    actual = {p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and p.name != 'SOURCE_FILES.json'}
    assert actual == set(manifest['files'])
    for name, pin in manifest['files'].items():
        p = D / name; raw = p.read_bytes()
        assert not p.is_symlink() and len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256']
        if p.suffix == '.py':
            ast.parse(raw, filename=name)
    base = json.loads((D / 'BASE_FRAMEWORK_FILES.json').read_bytes())
    merged = json.loads((D / 'MERGED_FRAMEWORK_FILES.json').read_bytes())
    assert len(base['files']) == 393 and len(merged['files']) == 396
    deltas = {p.relative_to(D / 'deltas').as_posix(): p for p in (D / 'deltas').rglob('*.py')}
    assert len(deltas) == 4
    for name, pin in merged['files'].items():
        if name in deltas:
            raw = deltas[name].read_bytes()
            assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256']
        else:
            assert pin == base['files'][name]
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    assert claim['training']['planned_global_microbatches'] == 160 and claim['budget']['single_origin_seconds'] == 900
    for case in ('T03.sft', 'T04.sft'):
        cfg = json.loads((D / 'cases' / (case + '-config-template.json')).read_bytes())
        assert cfg['root_GO'] is cfg['GPU_GO'] is False and cfg['supervisor'] is None
        assert cfg['case'] == case and len(cfg['expected_GPU_inventory']) == 8
        assert len(cfg['checkpoint_files']) == len(cfg['comparison_files']) == 14
        raw = (D / 'cases' / (case + '-native-binding.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == cfg['binding_SHA256']
    return dict(passed=True, source_members=len(actual), complete_framework_members=396,
                actual_Torch_import=False, actual_checkpoint_read=False, actual_GPU=False)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
