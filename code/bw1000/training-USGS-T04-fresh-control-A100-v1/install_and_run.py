"""Install exact private400 common source, then exec one identity-owned supervisor."""
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
LABEL = 'training-USGS-T04-fresh-control-A100-v1'
PYTHON = '/home/whj/nvidia-baseline/env-vlm/bin/python'


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def original_file(path, pin):
    need(path.is_file() and not path.is_symlink() and 0 <= pin['bytes'] <= 4 * 1024**2, 'BOUNDED_SOURCE_MEMBER')
    before = path.stat()
    with path.open('rb') as stream:
        raw = stream.read(4 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
         and len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'ACTUAL_ORIGINAL_SOURCE_PIN')
    return raw


def control_template(template, plan, pair_id):
    need(template['case'] == 'T04.sft' and template['arm'] == 'generic'
         and template['mode'] == 'fresh25' and template['pair_id'] == pair_id,
         'ONE_FRESH_GENERIC_ARM')
    need(template['initial_state'] == plan['actual_domain_initial']
         and template['pair_initial_reference'] == plan['actual_domain_initial_reference']
         and template['initial_state']['bytes'] == 174808783
         and template['initial_state']['SHA256'] == 'b574c04a3bdaaff7f061d7285dc0fc73f2cbadce4801fca97dbd00ad8772fa0d'
         and template['pair_initial_reference']['bytes'] == 551
         and template['pair_initial_reference']['SHA256'] == 'c5967f3e88134923b98cd596ac0fccea899000963b82ceb569ae689662c0746e',
         'GENERIC_REQUIRES_PINNED_ACTUAL_DOMAIN_INITIAL')
    need(template['training800']['bytes'] == 314277
         and template['training800']['SHA256'] == '564e7439a41de9d6c8b76af2ae0392f154e5157c60687aaff12e970f593986ab'
         and template['world_size'] == 8 and template['physical_rows'] == 800
         and template['max_steps'] == 25 and template['accumulation'] == 4, 'CONTROL_800_DENOMINATOR')


def main():
    need(isinstance(INSTALL_PAYLOAD_B64, str) and len(INSTALL_PAYLOAD_B64) <= 4 * 1024**2, 'SMALL_ROOT_PACKAGE')
    payload = json.loads(base64.b64decode(INSTALL_PAYLOAD_B64, validate=True))
    epoch = payload['epoch']
    need(type(epoch) in (int, float) and math.isfinite(epoch) and 0 < epoch <= time.time() < epoch + 760,
         'ORIGINAL900_EPOCH_ADMISSION_RESERVE')
    need(re.fullmatch('[0-9a-f]{32}', payload['request_id'])
         and re.fullmatch('[0-9a-f]{40}', payload['publication']), 'ORIGINAL_CONTEXT')
    def check():
        need(epoch <= time.time() < epoch + 760, 'ORIGINAL_INSTALL_BUDGET')
    def expired(*_):
        raise TimeoutError('ORIGINAL900_EPOCH')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, epoch + 890 - time.time()))
    package = {name: base64.b64decode(b, validate=True) for name, b in payload['package'].items()}
    mf = package['SOURCE_FILES.json']
    need(sha(mf) == payload['source_manifest_SHA256'], 'PUBLISHED_COMPOSITE_MANIFEST')
    pins = json.loads(mf)['files']
    need(set(package) == set(pins) | {'SOURCE_FILES.json'}, 'EXACT_COMPOSITE_MEMBERS')
    for name, raw in package.items():
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts and '.' not in rel.parts, 'SOURCE_PATH')
        if name != 'SOURCE_FILES.json':
            need(len(raw) == pins[name]['bytes'] and sha(raw) == pins[name]['SHA256'], 'COMPOSITE_SOURCE_PIN')
    claim = json.loads(package['CLAIM.json'])
    need(sha(package['CLAIM.json']) == payload['claim_SHA256'] and claim['root_GO'] is True
         and claim['GPU_GO'] is True and claim['max_invocations'] == 1
         and claim['arm'] == 'generic' and claim['mode'] == 'fresh25', 'ONE_PUBLISHED_GENERIC_SCOPE')
    root = HOME / LABEL
    need(root.parent.resolve() == root.parent and not root.exists() and not root.is_symlink(),
         'ORIGINAL_NAMESPACE_EXISTS_RECONCILE_NO_REPLAY')
    root.mkdir(mode=0o700)
    try:
        write(root / 'install-intent.json', canonical(dict(request_id=payload['request_id'], publication=payload['publication'],
            epoch=epoch, deadline=epoch + 900, source_manifest_SHA256=sha(mf), claim_SHA256=payload['claim_SHA256'],
            phase='SOURCE_INSTALL_ONLY_NO_GPU', PID=os.getpid(), UID=os.getuid(), host=os.uname().nodename,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))
        for name, raw in package.items():
            check()
            write(root / 'src' / name, raw)
        src = root / 'src'
        plan = json.loads(package['INSTALL_PLAN.json'])
        need(sha(package['supervisor/bounded_GPU_supervisor.py']) ==
             plan['required_RSS_driver_SHA256'] == '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f',
             'EXACT_RSS_V2_DRIVER_ACTUALLY_EXECUTED')
        need(plan['namespace'] == str(root) == claim['namespace'], 'OWNED_NAMESPACE')
        allowed = sorted(set(os.sched_getaffinity(0)) - {0, 1})
        need(len(allowed) >= 32, 'MINIMUM32_CPU_AFFINITY')
        actual = allowed[:64]
        os.sched_setaffinity(0, set(actual))
        need(os.sched_getaffinity(0) == set(actual), 'ACTUAL_CPU_MASK')
        old = Path(plan['existing_framework'])
        base = json.loads(package['BASE_FRAMEWORK_FILES.json'])['files']
        need(len(base) == 396, 'ORIGINAL396_SOURCE')
        framework = root / 'framework'
        overlays = json.loads(package['OVERLAYS.json'])['files']
        need(len(overlays) == 5 and set(overlays).issubset(json.loads(package['MERGED_FRAMEWORK_FILES.json'])['files']),
             'EXACT_FIVE_OVERLAYS')
        # Reuse each qualified original source byte. Never modify the old installation.
        for name, pin in base.items():
            check()
            rel = PurePosixPath(name)
            need(not rel.is_absolute() and '..' not in rel.parts, 'BASE_PATH')
            raw = original_file(old / name, pin)
            if name not in overlays:
                write(framework / name, raw)
        for name, pin in overlays.items():
            check()
            raw = package['overlay/' + name]
            need(len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'OVERLAY_PIN')
            write(framework / name, raw)
        merged = json.loads(package['MERGED_FRAMEWORK_FILES.json'])['files']
        need(len(merged) == 400, 'MERGED400')
        for name, pin in merged.items():
            check()
            original_file(framework / name, pin)
        template = json.loads(package['CONFIG.template.json'])
        control_template(template, plan, claim['pair_id'])
        template.update(epoch=epoch, deadline=epoch + 820, maximum_seconds=820,
                        request_id=payload['request_id'], source_publication=payload['publication'])
        runtime_template = root / 'actual-template.json'
        runtime_raw = canonical(template)
        write(runtime_template, runtime_raw)
        write(root / 'install-ready.json', canonical(dict(request_id=payload['request_id'], epoch=epoch,
            deadline=epoch + 900, framework_members=400, CPU_affinity=actual, GPU_initialized=False,
            runtime_template=dict(bytes=len(runtime_raw), SHA256=sha(runtime_raw)))))
        command = [PYTHON, '-B', str(src / 'supervisor/bounded_GPU_supervisor.py'),
            '--template', str(runtime_template), '--template-sha256', sha(runtime_raw),
            '--framework', str(framework), '--scope', str(root / 'run'),
            '--request-id', payload['request_id'], '--publication', payload['publication'], '--epoch', str(epoch),
            '--a100-launch', str(src / 'T04-original-launch.json'), '--swift', plan['swift'],
            '--models', plan['model_parent'], '--plugins', plan['plugins'], '--master-port', str(plan['master_port']),
            '--source-manifest-sha256', plan['supervisor_manifest']['SHA256'],
            '--claim-sha256', plan['supervisor_claim']['SHA256']]
        check()
        os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline')
        os.execv(PYTHON, command)
    except BaseException as error:
        # No model/native subprocess exists before this process execs the owned supervisor.
        result = dict(request_id=payload['request_id'], epoch=epoch, deadline=epoch + 900,
            state='KNOWN_INSTALL_FAILURE_NO_REPLAY', error_type=type(error).__name__,
            error_SHA256=sha(str(error).encode()), GPU_launched=False, child_processes=0)
        write(root / 'install-failure.json', canonical(result))
        print(json.dumps(result), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
