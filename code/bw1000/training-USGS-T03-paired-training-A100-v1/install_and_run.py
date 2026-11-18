"""One prospective T03 arm; merge common404, stage800 once, exec qualified owned driver."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import time

HOME = Path('/home/whj/nvidia-baseline/alignment-assets')
PYTHON = '/home/whj/nvidia-baseline/env-vlm/bin/python'


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def original_file(path, pin):
    need(path.is_file() and not path.is_symlink() and 0 <= pin['bytes'] <= 4 * 1024**2, 'BOUNDED_ORIGINAL_SOURCE')
    before = path.stat()
    with path.open('rb') as f:
        raw = f.read(4 * 1024**2 + 1); after = os.fstat(f.fileno())
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and
         len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'ACTUAL_ORIGINAL_SOURCE_PIN')
    return raw


def main():
    need(isinstance(INSTALL_PAYLOAD_B64, str) and len(INSTALL_PAYLOAD_B64) <= 4 * 1024**2, 'SMALL_ROOT_PAYLOAD')
    payload = json.loads(base64.b64decode(INSTALL_PAYLOAD_B64, validate=True))
    epoch, arm = payload['epoch'], payload['arm']
    need(type(epoch) in (int, float) and math.isfinite(epoch) and 0 < epoch <= time.time() < epoch + 760,
         'ONE900_EPOCH_INCLUDES_PREPARATION')
    need(arm in ('domain', 'generic') and re.fullmatch('[0-9a-f]{32}', payload['request_id']) and
         re.fullmatch('[0-9a-f]{40}', payload['publication']), 'ONE_ARM_CONTEXT')
    end = time.monotonic() + epoch + 760 - time.time()
    def check():
        need(epoch <= time.time() < epoch + 760 and time.monotonic() < end, 'INSTALLATION_ORIGINAL_EPOCH_EXHAUSTED')
    def expired(*_):
        raise TimeoutError('ORIGINAL900_EPOCH')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, epoch + 890 - time.time())
    package = {name: base64.b64decode(raw, validate=True) for name, raw in payload['package'].items()}
    mf = package['SOURCE_FILES.json']; need(sha(mf) == payload['source_manifest_SHA256'], 'PUBLISHED_SOURCE_MF')
    pins = json.loads(mf)['files']
    need(set(package) == set(pins) | {'SOURCE_FILES.json'}, 'EXACT_SOURCE_PACKAGE')
    for name, raw in package.items():
        check(); rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts and '.' not in rel.parts, 'SOURCE_PATH')
        if name != 'SOURCE_FILES.json':
            need(len(raw) == pins[name]['bytes'] and sha(raw) == pins[name]['SHA256'], 'SOURCE_MEMBER_PIN')
    claim = json.loads(package['CLAIM.json']); plan = json.loads(package['INSTALL_PLAN.json'])
    need(sha(package['CLAIM.json']) == payload['claim_SHA256'] and claim['root_GO'] is True and
         claim['GPU_GO'] is True and claim['max_invocations_per_arm'] == 1 and
         claim['formal17_GO'] is False and claim['Model_GO'] is False, 'PUBLISHED_ACTUAL_ARM_ADOPTION')
    root = Path(plan['namespaces'][arm])
    need(root.parent == HOME and not root.exists() and not root.is_symlink(), 'EXACT_NAMESPACE_EXISTS_NO_REPLAY')
    # Generic is admitted only after an actual independent, clean, complete domain25 terminal.
    domain_qualification_raw = None
    if arm == 'generic':
        domain_qualification_raw = base64.b64decode(payload['domain_qualification_b64'], validate=True)
        need(0 < len(domain_qualification_raw) <= 16384, 'BOUNDED_REAL_DOMAIN_QUALIFICATION')
        q = json.loads(domain_qualification_raw)
        need(q['schema'] == 'T03_PAIRED_DOMAIN25_INDEPENDENT_TERMINAL_V1' and q['case'] == 'T03.sft' and
             q['arm'] == 'domain' and q['pair_id'] == claim['pair_id'] and q['namespace'] == plan['namespaces']['domain'] and
             q['source_manifest_SHA256'] == payload['source_manifest_SHA256'] and q['independent_passed'] is True and
             q['engineering_completed'] is True and q['release_permitted'] is True and q['completed_ranks'] == 8 and
             q['completed_updates'] == 25 and q['global_micro_exposures'] == 800, 'DOMAIN_C0_INDEPENDENT_FIRST_CONTROL_LATER')
    root.mkdir(mode=0o700)
    try:
        fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
        write(root / 'install-intent.json', canonical(dict(arm=arm, request_id=payload['request_id'], publication=payload['publication'],
            pair_id=claim['pair_id'], epoch=epoch, deadline=epoch + 900, source_manifest_SHA256=sha(mf),
            claim_SHA256=payload['claim_SHA256'], PID=os.getpid(), startticks=int(fields[19]), UID=os.getuid(),
            host=os.uname().nodename, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))
        for name, raw in package.items():
            check(); write(root / 'src' / name, raw)
        src = root / 'src'
        if domain_qualification_raw is not None:
            write(root / 'domain-qualification-private.json', domain_qualification_raw)
        need(sha(package['supervisor/qualified_driver.py']) == '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f',
             'QUALIFIED_RSS_V2_ACTUAL_MODULE')
        allowed = sorted(set(os.sched_getaffinity(0)) - {0, 1})
        need(len(allowed) >= 32, 'MINIMUM32_CPU_AFFINITY'); actual = allowed[:64]
        os.sched_setaffinity(0, set(actual)); need(os.sched_getaffinity(0) == set(actual), 'ACTUAL_CPU_MASK')
        base = json.loads(package['BASE_FRAMEWORK_FILES.json'])['files']
        overlays = json.loads(package['OVERLAYS.json'])['files']
        need(len(base) == 400 and len(overlays) == 5, '400_PLUS_EXPLICIT_T03_DELTA')
        framework = root / 'framework'; old = Path(plan['existing_framework'])
        for name, pin in base.items():
            check(); raw = original_file(old / name, pin)
            if name not in overlays:
                write(framework / name, raw)
        for name, pin in overlays.items():
            check(); raw = package['overlay/' + name]
            need(len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'T03_OVERLAY_PIN')
            write(framework / name, raw)
        merged = json.loads(package['MERGED_FRAMEWORK_FILES.json'])['files']
        need(len(merged) == 404, 'UNIFIED404')
        for name, pin in merged.items():
            check(); original_file(framework / name, pin)
        sys.path.insert(0, str(src))
        if arm == 'domain':
            from prepare_views import build_views
            views = build_views(root / 'views', src, plan, check)
            view_path = root / 'views/pair-views-private.json'; view_raw = view_path.read_bytes()
            views_pin = dict(path=str(view_path), bytes=len(view_raw), SHA256=sha(view_raw))
        else:
            from prepare_views import pinned
            views_pin = q['views_manifest']
            need(views_pin['path'] == plan['namespaces']['domain'] + '/views/pair-views-private.json', 'ACTUAL_DOMAIN_VIEW_LOCATION')
            view_raw, _ = pinned(views_pin, check); views = json.loads(view_raw)
        v = views[arm]
        need(v['arm'] == arm and v['unique_rows'] == 45 and v['physical_rows'] == 800 and v['new_encodes'] == 0,
             'REAL_QUALIFIED45_FIXED800_VIEW')
        template = json.loads(package['CONFIG.template.json'])
        need(template['root_GO'] is True and template['GPU_GO'] is True and template['case'] == 'T03.sft', 'RUNTIME_TEMPLATE_ADOPTION')
        template.update(arm=arm, epoch=epoch, deadline=epoch + 820, maximum_seconds=820, request_id=payload['request_id'],
            source_publication=payload['publication'], output=str(root / 'run/witness'), train_output=str(root / 'run/train'),
            binding=dict(path=str(src / 'T03-native-binding.json'), **pins['T03-native-binding.json']), native45=v['native45'],
            training800=v['training800'], occurrences=v['occurrences'],
            framework_manifest=dict(path=str(src / 'MERGED_FRAMEWORK_FILES.json'), **pins['MERGED_FRAMEWORK_FILES.json']),
            model_admission=dict(path=str(src / 'T03-model-files.json'), **pins['T03-model-files.json']))
        need(template['initial_states'] == json.loads(package['INITIAL_REFERENCES.json'])['ranks'], 'ALL8_ORIGINAL_INITIAL_REFERENCES')
        runtime_raw = canonical(template); runtime = root / 'actual-template.json'; write(runtime, runtime_raw)
        write(root / 'install-ready.json', canonical(dict(request_id=payload['request_id'], arm=arm, epoch=epoch,
            deadline=epoch + 900, framework_members=404, CPU_affinity=actual, GPU_initialized=False,
            pair_views=views_pin, view_binding=v, runtime_template=dict(bytes=len(runtime_raw), SHA256=sha(runtime_raw)))))
        command = [PYTHON, '-B', str(src / 'supervisor/bounded_GPU_supervisor.py'),
            '--template', str(runtime), '--template-sha256', sha(runtime_raw), '--framework', str(framework),
            '--scope', str(root / 'run'), '--request-id', payload['request_id'], '--publication', payload['publication'],
            '--epoch', str(epoch), '--a100-launch', str(src / 'T03-original-launch.json'), '--swift', plan['swift'],
            '--models', plan['model_parent'], '--plugins', plan['plugins'], '--master-port', str(plan['master_ports'][arm]),
            '--source-manifest-sha256', plan['supervisor_manifest']['SHA256'], '--claim-sha256', plan['supervisor_claim']['SHA256']]
        check()
        os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline')
        os.execv(PYTHON, command)
    except BaseException as error:
        result = dict(request_id=payload['request_id'], arm=arm, epoch=epoch, deadline=epoch + 900,
            state='KNOWN_INSTALL_FAILURE_NO_REPLAY', error_type=type(error).__name__,
            error_SHA256=sha(str(error).encode()), GPU_launched=False, child_processes=0)
        write(root / 'install-failure.json', canonical(result)); print(json.dumps(result), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
