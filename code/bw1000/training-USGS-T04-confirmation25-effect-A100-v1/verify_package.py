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
    changed={'PLAN.json','CLAIM.json','SOURCE_PROOF.json','README.zh-CN.md','infer_dev10.py','effect_contract.py','score_after.py','verify_package.py','check_records_CPU.py'}
    for name,expected in json.loads(original)['files'].items():
        budget()
        if name not in changed:pin(D/'core'/name,expected)
    need((D/'core/infer_confirmation25.py').is_file() and not (D/'core/infer_dev10.py').exists(),'NEW_CONFIRMATION_WORKER_NOT_DEV_RENAMED_INPUT')
    core_plan=json.loads((D/'core/PLAN.json').read_bytes())
    need(core_plan['confirmation_inputs']['rows']==25 and len(core_plan['confirmation_media'])==5
         and core_plan['generation']['max_new_tokens']==256 and core_plan['confirmation_metadata']['closure_groups']==5,'FROZEN_CONFIRMATION25_CAP256_GROUPS')
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
    prior_pin = plan['prior_v1_failure_file']; prior = json.loads(pin(D / prior_pin['name'], prior_pin))
    need(prior_pin['SHA256'] == 'f374c88a3055e78478b1369c7b0dbeaaf93bb754eabe95e3ad155e2e5310d62e'
         and plan['prior_v1_failure_publication'] == '6949a3efb01bd85834afb467e1355659308e67e7'
         and prior['original_native_released'] is False and prior['independent_device_post'] == 'UNKNOWN'
         and plan['prior_failure_is_resource_release_proof'] is False, 'PRESERVE_FAILED_V1_AND_UNKNOWN_POST')
    prior2_pin = plan['prior_v2_failure_file']; prior2 = json.loads(pin(D / prior2_pin['name'], prior2_pin))
    need(prior2_pin['SHA256'] == 'b8dc5ebef39d2404feb589da0b31f6a61ee12bd882b69da1adc623c9c5027a2b'
         and plan['prior_v2_failure_publication'] == 'cb49b3509c6c2ed0153d7d19b0bab23bc223fc38'
         and prior2['native_released'] is False and prior2['independent_device_post'] == 'UNKNOWN'
         and plan['prior_v2_failure_is_resource_release_proof'] is False, 'PRESERVE_FAILED_V2_AND_UNKNOWN_POST')
    prior3_pin = plan['prior_v3_failure_file']; prior3 = json.loads(pin(D / prior3_pin['name'], prior3_pin))
    need(prior3_pin['SHA256'] == '56d0d599d5d3a34ce44f088cfd89a2d57abc412f5caab10bd20ce9f04c96da4d'
         and plan['prior_v3_failure_publication'] == 'acd573d390e3bff20a2abe1440f4bafdf4988551'
         and prior3['native_released'] is False and prior3['independent_device_post'] == 'UNKNOWN'
         and plan['prior_v3_failure_is_resource_release_proof'] is False, 'PRESERVE_FAILED_V3_AND_UNKNOWN_POST')
    need(not (D / 'transport.py').exists() and b'import transport' not in (D / 'gpu_admission.py').read_bytes(),
         'NO_TRANSPORT_ANCESTRY_GATE_EXECUTED')
    return dict(passed=True, files=len(mf), root_GO=flags[0], GPU_GO=flags[0],
        source_manifest_SHA256=hashlib.sha256((D / 'SOURCE_FILES.json').read_bytes()).hexdigest(),
        actual_model=0, actual_remote=0, actual_GPU=0)


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
