"""Static generic-control source/config closure; no provider or model imports."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def adoption_gate(c, claim, nested):
    flags = [row[key] for row in (c, claim, nested) for key in ('root_GO', 'GPU_GO')]
    assert all(type(value) is bool for value in flags) and len(set(flags)) == 1
    assert claim['Model_GO'] is claim['formal17'] is claim['effect_qualified'] is False
    assert c['formal17'] is False and nested['formal17'] is nested['effect_qualified'] is False
    return 'root_adopted' if flags[0] else 'source_review'


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
    assert len(source) == 9
    for name, pin in source.items():
        raw = (D / 'supervisor' / name).read_bytes()
        assert len(raw) == pin['bytes'] and digest(raw) == pin['SHA256']
    c = json.loads((D / 'CONFIG.template.json').read_bytes())
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    assert c['case'] == claim['case'] == 'T04.sft' and c['arm'] == claim['arm'] == 'generic'
    assert c['mode'] == claim['mode'] == 'fresh25' and c['pair_id'] == claim['pair_id']
    assert claim['max_invocations'] == 1 and claim['replay_allowed'] is False
    adoption = adoption_gate(c, claim, json.loads((D / 'supervisor/CLAIM.json').read_bytes()))
    assert c['initial_state']['bytes'] == 174808783 and c['initial_state']['SHA256'] == 'b574c04a3bdaaff7f061d7285dc0fc73f2cbadce4801fca97dbd00ad8772fa0d'
    assert c['pair_initial_reference']['bytes'] == 551 and c['pair_initial_reference']['SHA256'] == 'c5967f3e88134923b98cd596ac0fccea899000963b82ceb569ae689662c0746e'
    assert c['world_size'] == 8 and c['physical_rows'] == 800 and c['unique_rows'] == 45 and c['max_steps'] == 25
    assert c['native45']['SHA256'] == '12e9f892c0dbac5354a612dc1cfdb8fc65bee3de333ee619f48d6ac0fc3a68f0'
    assert c['training800']['SHA256'] == '564e7439a41de9d6c8b76af2ae0392f154e5157c60687aaff12e970f593986ab'
    assert c['occurrences']['SHA256'] == 'c9c69a23be8ba91ead6125d33b132f90e86a48ae3b7c1f611fcb5aaab394b79b'
    assert digest((D / 'supervisor/bounded_GPU_supervisor.py').read_bytes()) == '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f'
    assert len(c['providers']) == 15 and len(c['expected_GPU_inventory']) == 8
    for field, name in [('binding', 'T04-native-binding.json'), ('model_admission', 'T04-model-files.json'),
                        ('framework_manifest', 'MERGED_FRAMEWORK_FILES.json')]:
        raw = (D / name).read_bytes()
        assert c[field]['bytes'] == len(raw) and c[field]['SHA256'] == digest(raw)
    assert claim['budgets_seconds'] == dict(whole_original=900, native=820, stop=850, cleanup=880, alarm=890, client=895)
    return dict(passed=True, adoption=adoption, outer_members=len(mf['files']), common_framework_members=400,
                unchanged_base_members=395, new_or_replaced_overlay_members=5,
                actual_remote_calls=0, Torch_import=False, GPU_calls=0, actual_model_calls=0)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
