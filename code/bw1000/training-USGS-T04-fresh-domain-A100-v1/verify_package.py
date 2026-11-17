"""Static actual first-domain source/config closure; no provider or model imports."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    mf = json.loads((D / 'SOURCE_FILES.json').read_bytes())
    actual = {p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and p.name != 'SOURCE_FILES.json'}
    # The nested supervisor manifest is itself an outer package member.
    actual.add('supervisor/SOURCE_FILES.json')
    assert set(mf['files']) == actual
    for name, pin in mf['files'].items():
        p = D / name
        raw = p.read_bytes()
        assert not p.is_symlink() and len(raw) == pin['bytes'] and digest(raw) == pin['SHA256']
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
    base = json.loads((D / 'BASE_FRAMEWORK_FILES.json').read_bytes())['files']
    merged = json.loads((D / 'MERGED_FRAMEWORK_FILES.json').read_bytes())['files']
    overlay = json.loads((D / 'OVERLAYS.json').read_bytes())['files']
    assert len(base) == 396 and len(merged) == 400 and len(overlay) == 5
    for name, pin in base.items():
        if name not in overlay:
            assert merged[name] == pin
    for name, pin in overlay.items():
        raw = (D / 'overlay' / name).read_bytes()
        assert merged[name] == dict(bytes=len(raw), SHA256=digest(raw))
        assert pin['bytes'] == len(raw) and pin['SHA256'] == digest(raw)
    source = json.loads((D / 'supervisor/SOURCE_FILES.json').read_bytes())['files']
    assert len(source) == 8
    for name, pin in source.items():
        raw = (D / 'supervisor' / name).read_bytes()
        assert len(raw) == pin['bytes'] and digest(raw) == pin['SHA256']
    c = json.loads((D / 'CONFIG.template.json').read_bytes())
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    assert c['case'] == claim['case'] == 'T04.sft' and c['arm'] == claim['arm'] == 'domain'
    assert c['mode'] == claim['mode'] == 'fresh25' and c['pair_id'] == claim['pair_id']
    assert claim['max_invocations'] == 1 and claim['replay_allowed'] is False
    assert c['root_GO'] is c['GPU_GO'] is True and c['formal17'] is False
    assert c['initial_state'] is c['pair_initial_reference'] is None
    assert c['world_size'] == 8 and c['physical_rows'] == 800 and c['unique_rows'] == 45 and c['max_steps'] == 25
    assert c['native45']['SHA256'] == '07f0073299aa3ba5f82b359affcc09704f9e8515205f7f4b87ae9c835884a65c'
    assert c['training800']['SHA256'] == '57640ac8c8fd9a7d0614df276af37e0c92201ec293a3ea722fd0b6252ac01570'
    assert c['occurrences']['SHA256'] == '1b15a612ec1beb014c11e7e2beb86e54aae6876c7e5cb17c9466758a47f62338'
    assert len(c['providers']) == 15 and len(c['expected_GPU_inventory']) == 8
    for field, name in [('binding', 'T04-native-binding.json'), ('model_admission', 'T04-model-files.json'),
                        ('framework_manifest', 'MERGED_FRAMEWORK_FILES.json')]:
        raw = (D / name).read_bytes()
        assert c[field]['bytes'] == len(raw) and c[field]['SHA256'] == digest(raw)
    assert claim['budgets_seconds'] == dict(whole_original=900, native=820, stop=850, cleanup=880, alarm=890, client=895)
    return dict(passed=True, outer_members=len(mf['files']), common_framework_members=400,
                unchanged_base_members=395, new_or_replaced_overlay_members=5,
                actual_remote_calls=0, Torch_import=False, GPU_calls=0, actual_model_calls=0)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
