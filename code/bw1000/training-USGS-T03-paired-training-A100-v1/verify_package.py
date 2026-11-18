"""Stdlib package closure before any SSH, model import, or prospective training."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

D = Path(__file__).resolve().parent


def need(value, code):
    if not value:
        raise ValueError(code)


def verify(root=D):
    def read(name):
        return json.loads((root / name).read_bytes())
    def pin(name, expected):
        p = root / name
        raw = p.read_bytes()
        need(not p.is_symlink() and len(raw) == expected['bytes'] and
             hashlib.sha256(raw).hexdigest() == expected['SHA256'], 'EXACT_SOURCE_PIN:' + name)
        return raw
    mf = read('SOURCE_FILES.json')['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root / 'SOURCE_FILES.json'}
    need(set(mf) == actual, 'EXACT_PACKAGE_MEMBERS')
    for name, expected in mf.items():
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts, 'PACKAGE_PATH')
        raw = pin(name, expected)
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
    base, merged, overlays = (read(n)['files'] for n in
                             ('BASE_FRAMEWORK_FILES.json', 'MERGED_FRAMEWORK_FILES.json', 'OVERLAYS.json'))
    expected_overlays = {'tools/hygon_training_recipe.py', 'tools/usgs_t03_launch.py',
                         'vlm/plugins/usgs_t03_training.py', 'vlm/runtime/usgs_t03_contract.py',
                         'vlm/runtime/usgs_t03_witness.py'}
    need(len(base) == 400 and len(merged) == 404 and set(overlays) == expected_overlays,
         'COMMON404_EXACT5_DELTA')
    need(merged == dict(base, **overlays), 'MERGED_COMMON_FRAMEWORK')
    for name, expected in overlays.items():
        pin('overlay/' + name, expected)
    plan, claim, config = (read(n) for n in ('INSTALL_PLAN.json', 'CLAIM.json', 'CONFIG.template.json'))
    sub = read('supervisor/SOURCE_FILES.json')['files']
    for name, expected in sub.items():
        pin('supervisor/' + name, expected)
    pin('supervisor/SOURCE_FILES.json', plan['supervisor_manifest'])
    pin('supervisor/CLAIM.json', plan['supervisor_claim'])
    pin('MERGED_FRAMEWORK_FILES.json', plan['framework_manifest'])
    need(sub['qualified_driver.py']['SHA256'] ==
         '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'ACTUAL_QUALIFIED_RSS_MODULE')
    need(sub['held_child.py']['SHA256'] == '06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351' and
         sub['owned_processes.py']['SHA256'] == '5a3dd9b7530526f324b230ef0fe5be6ff2953f1331718d6520a1362bca229817',
         'QUALIFIED_CLEANUP_HELPERS')
    need(all(v['root_GO'] is True and v['GPU_GO'] is True for v in
             (plan, claim, config, read('supervisor/CLAIM.json'))), 'ROOT_ADOPTED_SCOPE')
    need(claim['formal17_GO'] is False and claim['Model_GO'] is False and config['formal17'] is False,
         'ENGINEERING_SCOPE_ONLY')
    need(claim['pair_id'] == plan['pair_id'] == config['pair_id'] and claim['max_invocations_per_arm'] == 1,
         'PAIR_ONCE_ONLY')
    need(config['initial_states'] == read('INITIAL_REFERENCES.json')['ranks'] and
         set(config['initial_states']) == {str(i) for i in range(8)}, 'ORIGINAL_ALL8_INITIALS')
    need(config['initial_tensors'] == 504 and config['initial_elements'] == 43646976 and
         config['world_size'] == 8 and config['accumulation'] == 4 and config['physical_rows'] == 800 and
         config['max_steps'] == 25 and config['framework_members'] == 404, 'FIXED_ORIGINAL_RECIPE')
    for arm in ('domain', 'generic'):
        pin(plan['input_arms'][arm]['qualification']['file'], plan['input_arms'][arm]['qualification'])
    need(read('CONTROL_NATIVE_QUALIFICATION.json')['independent_passed'] is True and
         read('DOMAIN_NATIVE_QUALIFICATION.json')['native_preprocessing_qualified'] is True, 'ACTUAL_INPUT_QUALIFICATION')
    return dict(source_members=len(mf), framework_members=404, overlay_members=5,
                stdlib_only=True, model_imported=False, remote=False, GPU=False)


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
