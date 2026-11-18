"""Install one published source-only bundle, then exec its same-epoch supervisor."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import time

HOME = Path('/home/whj/nvidia-baseline/alignment-assets')
LABEL = 'training-USGS-T04-dev10-effect-A100-launch-v4'
PYTHON = '/home/whj/nvidia-baseline/env-vlm/bin/python'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def parse_process_identity(raw, expected_pid):
    need(isinstance(raw, str) and type(expected_pid) is int and expected_pid > 0, 'INSTALLER_STAT_TYPES')
    fields = raw.rsplit(') ', 1)[1].split()
    need(len(fields) >= 20 and int(raw.split(' ', 1)[0]) == expected_pid
         and int(fields[19]) > 0 and int(fields[1]) >= 0 and int(fields[2]) > 0, 'INSTALLER_EXACT_PROC_STAT')
    return dict(PID=expected_pid, PPID=int(fields[1]), PGID=int(fields[2]), startticks=int(fields[19]))


def write(path, body):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())


def main():
    need(isinstance(INSTALL_PAYLOAD_B64, str) and len(INSTALL_PAYLOAD_B64) <= 4 * 1024**2, 'BOUNDED_SOURCE_PAYLOAD')
    payload = json.loads(base64.b64decode(INSTALL_PAYLOAD_B64, validate=True)); epoch = payload['epoch']
    need(type(epoch) in (int, float) and math.isfinite(epoch) and 0 < epoch <= time.time() < epoch + 760, 'ORIGINAL_EPOCH_ADMISSION')
    need(re.fullmatch('[0-9a-f]{32}', payload['request_id']) and re.fullmatch('[0-9a-f]{40}', payload['publication']), 'ORIGINAL_CONTEXT')
    def expired(*_):
        raise TimeoutError('SAME_ORIGINAL900_EPOCH')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, max(.001, epoch + 890-time.time()))
    package = {name: base64.b64decode(body, validate=True) for name, body in payload['package'].items()}
    mf = package['SOURCE_FILES.json']; need(sha(mf) == payload['source_manifest_SHA256'], 'PUBLISHED_MANIFEST')
    pins = json.loads(mf)['files']; need(set(package) == set(pins) | {'SOURCE_FILES.json'}, 'EXACT_PACKAGE_MEMBERS')
    for name, body in package.items():
        parts = PurePosixPath(name)
        need(not parts.is_absolute() and '..' not in parts.parts and '.' not in parts.parts, 'RELATIVE_SOURCE_PATH')
        if name != 'SOURCE_FILES.json':
            need(len(body) == pins[name]['bytes'] and sha(body) == pins[name]['SHA256'], 'EXACT_SOURCE_BYTES')
    claim = json.loads(package['CLAIM.json']); plan = json.loads(package['INSTALL_PLAN.json'])
    need(sha(package['CLAIM.json']) == payload['claim_SHA256'] and claim['root_GO'] is claim['GPU_GO'] is True
         and claim['max_invocations'] == 1 and claim['Model_GO'] is False, 'ONE_PUBLISHED_ROOT_ADOPTION')
    prior_pin = plan['prior_v1_failure_file']; prior_raw = package[prior_pin['name']]
    need(len(prior_raw) == prior_pin['bytes'] and sha(prior_raw) == prior_pin['SHA256'] ==
         'f374c88a3055e78478b1369c7b0dbeaaf93bb754eabe95e3ad155e2e5310d62e', 'PUBLISHED_ORIGINAL_V1_FAILURE')
    prior = json.loads(prior_raw)
    need(prior['state'] == 'ORIGINAL_TERMINAL_ADMISSION_FAILURE' and prior['original_native_released'] is False
         and prior['original_engineering_completed'] is False and prior['independent_device_post'] == 'UNKNOWN'
         and plan['prior_failure_is_resource_release_proof'] is False, 'PRIOR_FAILURE_NEVER_FRESH_RESOURCE_ADMISSION')
    prior2_pin = plan['prior_v2_failure_file']; prior2_raw = package[prior2_pin['name']]
    need(len(prior2_raw) == prior2_pin['bytes'] and sha(prior2_raw) == prior2_pin['SHA256'] ==
         'b8dc5ebef39d2404feb589da0b31f6a61ee12bd882b69da1adc623c9c5027a2b', 'PUBLISHED_ORIGINAL_V2_FAILURE')
    prior2 = json.loads(prior2_raw)
    need(prior2['native_released'] is False and prior2['observed_predictions'] == 0
         and prior2['independent_device_post'] == 'UNKNOWN'
         and plan['prior_v2_failure_is_resource_release_proof'] is False, 'V2_FAILURE_NOT_RESOURCE_ADMISSION')
    prior3_pin = plan['prior_v3_failure_file']; prior3_raw = package[prior3_pin['name']]
    need(len(prior3_raw) == prior3_pin['bytes'] and sha(prior3_raw) == prior3_pin['SHA256'] ==
         '56d0d599d5d3a34ce44f088cfd89a2d57abc412f5caab10bd20ce9f04c96da4d', 'PUBLISHED_ORIGINAL_V3_FAILURE')
    prior3 = json.loads(prior3_raw)
    need(prior3['native_released'] is False and prior3['observed_predictions'] == 0
         and prior3['independent_device_post'] == 'UNKNOWN'
         and plan['prior_v3_failure_is_resource_release_proof'] is False, 'V3_FAILURE_NOT_RESOURCE_ADMISSION')
    core_plan = json.loads(package['core/PLAN.json']); core_claim = json.loads(package['core/CLAIM.json'])
    need(all(value is True for value in [plan['root_GO'], plan['GPU_GO'], core_plan['root_GO'], core_plan['GPU_GO'],
         core_claim['root_GO'], core_claim['GPU_GO']]), 'ALL_EIGHT_ADOPTION_FLAGS')
    need(sha(package['core/SOURCE_FILES.json']) == plan['core_manifest_SHA256'], 'NESTED_CORE_MF')
    need(sha(package['qualified/bounded_GPU_supervisor.py']) == plan['qualified_driver_SHA256'] ==
         '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'EXECUTED_QUALIFIED_RSS_V2')
    root = HOME / LABEL
    need(str(root) == plan['namespace'] == claim['namespace'] and root.parent.resolve() == root.parent
         and not root.exists() and not root.is_symlink(), 'EXACT_NAMESPACE_EXISTS_RECONCILE_NO_REPLAY')
    root.mkdir(mode=0o700)
    installer = None
    try:
        installer = dict(**parse_process_identity(Path('/proc/self/stat').read_text(), os.getpid()),
            UID=os.getuid(), host=os.uname().nodename,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        write(root / 'install-intent.json', (json.dumps(dict(request_id=payload['request_id'], publication=payload['publication'],
            epoch=epoch, deadline=epoch+900, source_manifest_SHA256=sha(mf), claim_SHA256=payload['claim_SHA256'],
            **installer, GPU_initialized=False), sort_keys=True)+'\n').encode())
        for name, body in package.items():
            need(time.time() < epoch + 760, 'SAME_EPOCH_INSTALL_BUDGET')
            write(root / 'src' / name, body)
        need(not (root / 'run').exists(), 'UNIQUE_NEW_CONTROLLER_SCOPE')
        command = [PYTHON, '-B', str(root / 'src/bounded_eval.py'), '--scope', str(root / 'run'),
            '--request-id', payload['request_id'], '--publication', payload['publication'], '--epoch', str(epoch),
            '--source-manifest-sha256', sha(mf), '--claim-sha256', payload['claim_SHA256']]
        os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline',
            PYTHONPATH=str(root / 'src/core') + ':' + plan['framework_root'])
        os.execv(PYTHON, command)
    except BaseException as error:
        result = dict(request_id=payload['request_id'], epoch=epoch, deadline=epoch+900,
            state='KNOWN_INSTALL_FAILURE_NO_REPLAY', error_type=type(error).__name__, error_SHA256=sha(str(error).encode()),
            GPU_launched=False, child_processes=0, installer_identity=installer)
        write(root / 'install-failure.json', (json.dumps(result, sort_keys=True)+'\n').encode())
        print(json.dumps(result), flush=True); return 1


if __name__ == '__main__':
    raise SystemExit(main())
