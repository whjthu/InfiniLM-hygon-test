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
LABEL = 'training-USGS-T03-dev10-effect-A100-v1'
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
