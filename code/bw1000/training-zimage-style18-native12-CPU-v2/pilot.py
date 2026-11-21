"""Submit and audit the single adopted Style18 native CPU input scope."""

import argparse
import base64
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
HERE = Path(globals().get('__file__', '<stdin>')).resolve().parent
PARENT, NODE, ZONE = '1087656', 'f11r4n14', 'T07'
LABEL = 'training-zimage-style18-native12-CPU-v2'
REQUEST = '1268f832fa01473a8f523dc9b80d51ae'


RUN_NAME = LABEL
SLURM_SECONDS, WORKER_SECONDS, CONTROLLER_SECONDS = 240, 220, 300
APPTAINER = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
HOST_PYTHON = '/usr/local/python3.12/bin/python3'
SOURCE_NAMES = set(['CLAIM.json', 'ORIGINAL_PROVIDER_FILES.json', 'PLAN.json', 'README.zh-CN.md', 'SOURCE_PROOF.json', 'SOURCE_FILES.json', 'admission.py', 'ledger.json', 'metadata.csv', 'pilot.py', 'launch.py', 'prepare_inputs.py', 'run_native.py', 'source/coordination/training/zimage-style18-target-train-inputs-v1.json', 'source/tools/preflight_zimage_style18_train_input.py', 'sync_dispatch.py', 'qualified_sampler.py', 'check_staging_import.py'])
FROZEN = {'sync_dispatch.py': '3c13176628661ddc2474fa96b15f37da1c33cba49ad5ecf1493a8711587b7aaa', 'qualified_sampler.py': '09fc94b7f527c159bd489bacbf7718dad8615ba61cd3f361a44f441da14a0430', 'admission.py': 'fc89becc1b18bbab71e9a0c1dcdb05482b00643f7dff8cdfcbe1f2dd265291e5', 'ledger.json': '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9', 'metadata.csv': 'f71b51feb39af32e1d79b7750c9c4c437d930a35f69b60e52f80389ba6084350'}
HEX = re.compile(r'[a-f0-9]{64}')
PCI = re.compile(r'[a-f0-9]{4}:[a-f0-9]{2}:[01][a-f0-9]\.[0-7]')
HIP_SHA256 = 'b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b'
SIZES, ARMS = (1024**2, 32 * 1024**2), ('serial', 'overlap', 'overlap', 'serial')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return sha256(raw).hexdigest()


def source_identity(files):
    pins = {name: sha(raw) for name, raw in sorted(files.items())}
    return sha(json.dumps(pins, separators=(',', ':')).encode())


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def read(path, limit=128 * 1024):
    need(path.resolve(strict=True) == path and path.is_file(), 'original receipt path is not canonical')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit, 'original receipt is empty or too large')
    return raw


def source_verify(files):
    need(set(files) == SOURCE_NAMES and all(0 < len(raw) <= 32 * 1024 for raw in files.values()), 'source pack bounds')
    for name, expected in FROZEN.items():
        need(sha(files[name]) == expected, f'{name} differs from fixed scope')
    manifest = json.loads(files['SOURCE_FILES.json'])['files']
    need(set(manifest) == SOURCE_NAMES - {'SOURCE_FILES.json', 'metadata.csv'}, 'closed delivered source inventory differs')
    for name, pin in manifest.items():
        need(len(files[name]) == pin['bytes'] and sha(files[name]) == pin['SHA256'], f'{name} differs from source manifest')
    claim = json.loads(files['CLAIM.json'])
    need(claim['request_id'] == REQUEST and claim['root_GO'] is claim['CPU_GO'] is True
         and claim['GPU_GO'] is claim['training_GO'] is False
         and claim['candidate_node'] == NODE and claim['parent'] == PARENT, 'new native CPU grant differs')


def scope(plan):
    allowed = {'request', 'epoch', 'PCI', 'before_CPU_step', 'source_contract_SHA256', 'before_snapshot_SHA256',
               'publication', 'handoff_SHA256'}
    need(set(plan) == allowed and plan['request'] == REQUEST and type(plan['epoch']) is float
         and math.isfinite(plan['epoch']) and re.fullmatch(PARENT + r'\.[0-9]+', plan['before_CPU_step'])
         and re.fullmatch(r'[a-f0-9]{40}', plan['publication'])
         and all(HEX.fullmatch(plan[key]) for key in ('before_snapshot_SHA256', 'source_contract_SHA256', 'handoff_SHA256')),
         'original pilot request or CPU admission receipt differs')
    need(set(plan['PCI']) == {NODE} and len(set(plan['PCI'][NODE])) == len(plan['PCI'][NODE]) == 8
         and all(PCI.fullmatch(value) for value in plan['PCI'][NODE]), 'exact O01 card set differs')


def load_scheduler(target):
    source_verify({name: read(target / name, 32 * 1024) for name in SOURCE_NAMES})
    sys.path.insert(0, str(target))
    import sync_dispatch
    for name in ('sync_dispatch', 'qualified_sampler', 'admission'):
        provider = sys.modules[name]
        need(Path(provider.__file__).resolve(strict=True) == target / (name + '.py'),
             'scheduler imported a provider outside delivered source')
    return sync_dispatch


def command(target, run, pci_file, epoch):
    plan = json.loads(read(target / 'PLAN.json'))
    need(plan['request_id'] == REQUEST and plan['namespace'] == str(run), 'frozen plan differs')
    need(Path('/usr/local/hyhal').resolve(strict=True) == Path('/usr/local/hyhal') and
         Path('/opt/hyhal').resolve(strict=True) == Path('/usr/local/hyhal'), 'original hyhal bind differs')
    data = run / 'data/train'
    (data / 'images').mkdir(parents=True, mode=0o700)
    (run / 'evidence').mkdir(mode=0o700)
    binding_name = 'source/coordination/training/zimage-style18-target-train-inputs-v1.json'
    binding = json.loads(read(target / binding_name))
    need(len(binding['rows']) == 12, 'training image count differs')
    for i, row in enumerate(binding['rows']):
        leaf = f'{i:03d}.png'
        need(row['ordinal'] == i and row['path'] == 'images/' + leaf, 'training image order differs')
        expected = row['image']
        raw = read(Path(plan['source_images']) / leaf, expected['bytes'])
        need(len(raw) == expected['bytes'] and sha(raw) == expected['SHA256'], 'target training image differs')
    with (data / 'metadata.csv').open('xb') as stream:
        stream.write(read(target / 'metadata.csv', 3601))
    for member in ('CLAIM.json', 'run_native.py', 'ORIGINAL_PROVIDER_FILES.json', binding_name,
                   'source/tools/preflight_zimage_style18_train_input.py'):
        dest = run / 'source' / member
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(read(target / member))
    left = min(SLURM_SECONDS, int(epoch + 250 - time.time()))
    need(left >= 60, 'insufficient original work budget')
    minutes, seconds = divmod(left, 60)
    prefix = plan['CPU_srun_prefix'][:-1] + [f'--time=00:{minutes:02d}:{seconds:02d}']
    runtime = plan['container_command']
    runtime[-1] = runtime[-1].replace('"$ORIGINAL_EPOCH"', repr(epoch)).replace('"$ORIGINAL_DEADLINE"', repr(epoch+300))
    return [*prefix, '--output='+str(run/'stdout.private'), '--error='+str(run/'stderr.private'), *runtime]


def observation(row):
    if not row:
        return dict(state='UNKNOWN', terminal=False)
    need(len(row) == 7 and re.fullmatch(PARENT + r'\.[0-9]+', row[0]) and row[1] == LABEL
         and row[5] == NODE, 'accounting row differs from original node or label')
    state = row[2].split()[0].rstrip('+')
    terminal = state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
                         'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')
    return dict(state=state, terminal=terminal, accounting=row,
                accounting_columns=['JobIDRaw','JobName%100','State','ExitCode','ElapsedRaw','NodeList','AllocTRES'])


def reconcile(target, scheduler):
    run = ROOT / 'results' / RUN_NAME
    plan_raw = read(target / 'plan.json')
    plan = json.loads(plan_raw)
    scope(plan)
    need(source_identity({name: read(target / name, 32 * 1024) for name in SOURCE_NAMES}) ==
         plan['source_contract_SHA256'], 'original source pack differs')
    result = dict(request=REQUEST, label=LABEL, node=NODE, planned_rows=12, passed_rows=0,
                  post_required=True, replay_allowed=False, native_qualified=False,
                  actual_model_loads=None, actual_optimizer_updates=None)
    result.update(observation(scheduler.account({LABEL}).get(LABEL)))
    if not result['terminal']:
        return result
    row = result['accounting']
    tres = dict(field.split('=', 1) for field in row[6].split(',') if '=' in field)
    need(tres.get('node') == '1' and tres.get('cpu') == '4' and tres.get('mem') in ('8G', '8192M')
         and not any(k.startswith('gres/') for k in tres), 'original CPU resources differ')
    source = run / 'evidence/native'
    try:
        raw = read(source / 'result.json')
        d = json.loads(raw)
        owner_raw = read(source / 'owner.json')
        owner = json.loads(owner_raw)
        need(d['request_id'] == owner['request_id'] == REQUEST and d['owner'] == owner
             and owner['step'] == row[0] and owner['epoch'] == plan['epoch'] and owner['deadline'] == plan['epoch']+300
             and owner['host'].split('.')[0] == NODE and len(set(owner['CPU_affinity'])) == 4,
             'original worker identity or epoch differs')
        result.update(result_SHA256=sha(raw), owner_SHA256=sha(owner_raw), passed_rows=d['passed_rows'],
                      native_state=d.get('native_state'), worker_state=d['state'], worker_phase=d.get('phase'),
                      worker_error_type=d.get('error_type'), native_entry_calls=d['native_entry_calls'],
                      actual_model_loads=d['model_loads'], actual_optimizer_updates=d['optimizer_updates'])
        if row[2:4] == ['COMPLETED','0:0'] and d['passed'] is True:
            native_raw = read(source / 'native-result.json')
            native = json.loads(native_raw)
            need(d['passed_rows'] == native['passed'] == native['native_dataset_rows'] == 12
                 and native['failed'] == d['forbidden_tensor_file_attempts'] == d['model_loads'] == d['optimizer_updates'] == 0
                 and d['native_entry_calls'] == 1 and native['state'] == 'NATIVE_TRAIN_INPUT_PASS',
                 'native twelve-input denominator differs')
            result.update(native_result_SHA256=sha(native_raw), native_qualified=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['read_failure'] = type(error).__name__
    return result


def dispatch(target, scheduler):
    plan = json.loads(read(target / 'plan.json'))
    scope(plan)
    epoch = plan['epoch']
    need(0 <= time.time() - epoch < 25, 'admission not within the original epoch')
    ledger = json.loads(read(target / 'ledger.json'))
    owners = scheduler.node_map(ledger)
    need(owners[NODE] == (PARENT, ZONE), 'exact parent O01 membership differs')
    all_nodes = sorted(node for node, (job, _) in owners.items() if job == PARENT)
    need(NODE not in scheduler.allocation(PARENT, all_nodes), 'target now has a numbered Slurm step')
    need(not scheduler.account({LABEL}), 'original label already exists; reconcile instead')
    run = ROOT / 'results' / RUN_NAME
    need((ROOT / 'containers/dtk26-sandbox-r1').is_dir() and os.access(APPTAINER, os.X_OK),
         'qualified DTK sandbox or Apptainer is absent')
    need(time.time() < epoch + 30, 'insufficient original dispatch time')
    run.mkdir(mode=0o700, exist_ok=False)
    (run / 'nodes').mkdir(mode=0o700)
    pci_file = run / 'pci.json'
    save(pci_file, plan['PCI'])
    argv = command(target, run, pci_file, epoch)
    save(run / 'intent.json', dict(plan=plan, argv=argv, replay_allowed=False,
                                  absolute_worker_deadline=epoch+WORKER_SECONDS,
                                  absolute_controller_deadline=epoch+CONTROLLER_SECONDS))
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
    save(run / 'submitted.json', dict(request=REQUEST, srun_PID=child.pid, attempted=True))
    try:
        rc = child.wait(timeout=max(0.01, epoch + CONTROLLER_SECONDS - 12 - time.time()))
    except subprocess.TimeoutExpired:
        rc = None
    result = observation(scheduler.account({LABEL}).get(LABEL))
    result.update(request=REQUEST, srun_client_rc=rc, owned_srun_pid=child.pid, replay_allowed=False)
    save(run / 'first-accounting.json', result)
    return result


def receive(payload):
    os.umask(0o077)
    plan = payload['plan']
    scope(plan)
    need(0 <= time.time() - plan['epoch'] < 25, 'shared host plan expired')
    packed = payload['files']
    need(set(packed) == SOURCE_NAMES, 'incorrect staged member names')
    files = {}
    for name, (pin, encoded) in packed.items():
        need(len(encoded) <= 46000, 'encoded source outside fixed bounds')
        raw = base64.b64decode(encoded, validate=True)
        need(sha(raw) == pin, 'envelope source differs')
        files[name] = raw
    source_verify(files)
    need(source_identity(files) == plan['source_contract_SHA256'], 'finite source envelope is not the original')
    target = ROOT / 'incoming' / RUN_NAME
    target.mkdir(mode=0o700, exist_ok=False)
    for name, raw in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    save(target / 'plan.json', plan)
    worker = subprocess.run(
        [HOST_PYTHON, '-I','-S','-B','-u',str(target / 'pilot.py'),'--dispatch'],
        stdin=subprocess.DEVNULL, timeout=max(0.01, plan['epoch'] + CONTROLLER_SECONDS - time.time()))
    return worker.returncode


def main():
    if 'PAYLOAD' in globals():
        return receive(globals()['PAYLOAD'])
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dispatch', action='store_true')
    modes.add_argument('--reconcile', action='store_true')
    args = parser.parse_args()
    need(HERE == ROOT / 'incoming' / RUN_NAME, 'expected unique staged pilot path')
    scheduler = load_scheduler(HERE)
    result = dispatch(HERE, scheduler) if args.dispatch else reconcile(HERE, scheduler)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
