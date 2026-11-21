"""Read-only target preflight for the original eight-DCU Qwen training request."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import signal
import stat
import time

from reconcile_qwen_usgs_dev80_original import (
    NODES as DEV_NODES, PARENT, REPO, RESULTS, live_accounting, live_membership,
    namespace, need, raw_file, reconcile, scan_accounting,
)


LABEL = 'training-qwen-usgs-exposure25-resume20-hip-v1'
REQUEST, NODE, PORT = 'f572fda6784140cbbdcae7bb91a916b9', 'f13r1n10', 29918
SOURCE = Path('code/bw1000') / LABEL
MANIFEST_SHA = 'e379c65c61bdb17680b7481d413d985f12e6a11dccc24cc658242c90e5812335'
CLAIM_SHA = '79bcd2e7efcb1bbbc49b683923719e0b0824e24eb13dbd9ef9aa26196d1489bc'
FRAMEWORK_SHA = 'e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41'
CPU_LABEL = 'training-qwen-usgs-table-native-input-CPU-v1'
CPU_LOG_SHA = 'efa9646e09c03f939030df137e4c7a8b6750b659a4baa59fcc91fdea3e63d917'
TRAIN_SHA = '0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62'
BASE = Path('training-qwen-a68952c-swift-initial-CPU-v1')
DATA = Path('training-domain-usgs-table-reading-v1')
MODEL = Path('training-qwen-original-public-model-v1/model')
INITIAL = Path('training-qwen-initial-trainables-v1')
PHASE_LABELS = {LABEL+'-'+phase for phase in ('pre', 'gpu', 'post')}


def exact(root, name, pin, bound):
    path = root / name
    need(path.resolve(strict=True) == path and path.is_relative_to(root), 'asset path differs: '+str(name))
    raw = raw_file(path, bound, wanted=pin['SHA256'])
    need(len(raw) == pin['bytes'], 'asset size differs: '+str(name))
    return raw


def source_pins(repo):
    source = repo / SOURCE
    raw = raw_file(source/'SOURCE_FILES.json', 32768, wanted=MANIFEST_SHA)
    members = json.loads(raw)['files']
    need(len(members) == 21 and members['CLAIM.json']['SHA256'] == CLAIM_SHA, 'source manifest scope differs')
    all_bytes = {name: exact(source, name, pin, 131072) for name, pin in members.items()}
    claim = json.loads(all_bytes['CLAIM.json'])
    need(claim['request'] == REQUEST and claim['parent'] == PARENT and claim['node'] == NODE
         and claim['master_port'] == PORT and claim['optimizer_updates'] == 30
         and all(claim[key] is False for key in ('root_GO', 'training_GO', 'GPU_GO', 'formal_training_GO')),
         'expected the frozen source-only Qwen packet')
    return json.loads(all_bytes['MODEL_ASSETS.json']), json.loads(all_bytes['TRAIN_TABLE_PINS.json'])


def accounted(raw):
    previous = scan_accounting(raw)
    rows = [line.split('|') for line in raw.decode().splitlines() if line.strip()]
    cpu = [row for row in rows if row[1] == CPU_LABEL]
    need(len(cpu) == 1 and cpu[0][:4] == ['1103606.490', CPU_LABEL, 'COMPLETED', '0:0']
         and cpu[0][5] == NODE and 'gres/dcu' not in cpu[0][6], 'original CPU input step not verified')
    selected = [row for row in rows if row[1] in PHASE_LABELS]
    need(len({row[1] for row in selected}) == len(selected)
         and all(re.fullmatch(PARENT+r'\.\d+', row[0]) and row[5] == NODE for row in selected),
         'duplicate or foreign current training step')
    return dict(SHA256=previous['raw_SHA256'], predecessor_rows=previous['rows'], current_rows=selected)


def model_metadata(path, size):
    info = path.lstat()
    need(path.resolve(strict=True) == path and stat.S_ISREG(info.st_mode) and info.st_size == size,
         'model or initial-state metadata differs: '+path.name)


def assets(base, models, tables):
    visit = raw_file(base/CPU_LABEL/'step.log', 4*1024**2, wanted=CPU_LOG_SHA)
    manifest = raw_file(base/BASE/'source/FRAMEWORK_FILES.json', 256*1024, wanted=FRAMEWORK_SHA)
    files = json.loads(manifest)['files']
    need(len(files) == 393, 'common framework denominator differs')
    for name, pin in files.items():
        exact(base/BASE/'framework', name, pin, 4*1024**2)
        need(stat.S_IMODE((base/BASE/'framework'/name).stat().st_mode) == pin['mode'], 'framework mode differs')
    originals = models['staged_source']
    raw_file(base/BASE/'source/original-launch.json', 131072, wanted=originals['original-launch.json'])
    binding = json.loads(raw_file(base/BASE/'source/QWEN_TARGET_NATIVE_BINDING.json', 131072,
                                  wanted=originals['QWEN_TARGET_NATIVE_BINDING.json']))
    initial = binding['initial_state']
    init_path = Path(initial['path'])
    need(init_path.is_relative_to(base/INITIAL), 'initial-state path differs from the qualified tree')
    model_metadata(init_path, initial['bytes'])
    train = raw_file(base/DATA/'data/train/swift.jsonl', 2*1024**2, wanted=TRAIN_SHA)
    need(len(train.splitlines()) == 88, 'original train row count differs')
    stage = json.loads(raw_file(base/DATA/'stage-receipt.json', 128*1024))
    need(stage['request'] == '72e8a7066f73425cbf5a7e0921baf09c' and stage['staged'] is True
         and stage['output_identity_SHA256'] == 'd7c5cfc9b2b00982ebd25fb03c341807fcc272e7f989886c33118316bd8edd8b'
         and stage['blind_and_dev_gold_uploaded'] is False, 'original train-only stage differs')
    need(len(tables) == 22 and all(re.fullmatch(r'table-\d{3}\.png', name) for name in tables),
         'original training media names or count differ')
    for name, pin in tables.items():
        exact(base/DATA/'data/train/media', name, pin, 4*1024**2)
    shards = 0
    for name, pin in models['model'].items():
        if name.endswith('.safetensors'):
            model_metadata(base/MODEL/name, pin['bytes'])
            shards += 1
        else:
            exact(base/MODEL, name, pin, 32*1024**2)
    need(shards == 4, 'original model shard denominator differs')
    return dict(framework_files=len(files), train_rows=88, train_media_files=len(tables),
                CPU_input_log_bytes=len(visit), model_shard_metadata=shards,
                small_model_files=len(models['model'])-shards, model_weight_body_bytes_read=0,
                initial_state_metadata=True, initial_state_body_bytes_read=0)


def preflight(repo, base, *, account=live_accounting, membership=live_membership, verify=assets):
    predecessor = reconcile(repo, base, account=account, membership=membership)
    result = dict(request=REQUEST, parent=PARENT, node=NODE, required_port=PORT,
                  source_manifest_SHA256=MANIFEST_SHA, claim_SHA256=CLAIM_SHA, predecessor=predecessor,
                  actual_slurm_submissions=0, device_admission=False, training_READY=False)
    if predecessor['original_hold_eligible_for_removal'] is not True:
        return dict(result, state='DEV80_RECONCILIATION_REQUIRED')
    base = base.resolve(strict=True)
    anchor = base.stat()
    folder = base/LABEL

    def inspect():
        value = accounted(account())
        value['current_namespace_absent'] = not folder.exists() and not folder.is_symlink()
        value['predecessor_namespace_absent'] = namespace(base)['present'] is False
        value['clear'] = (not value['predecessor_rows'] and not value['current_rows']
                          and value['current_namespace_absent'] and value['predecessor_namespace_absent'])
        return value

    first = inspect()
    if first['clear'] is not True:
        return dict(result, state='ORIGINAL_REQUEST_RECONCILIATION_REQUIRED', first=first)
    facts = verify(base, *source_pins(repo))
    placement = membership(repo)
    last = inspect()
    after = base.stat()
    need((anchor.st_dev, anchor.st_ino) == (after.st_dev, after.st_ino), 'original results directory changed')
    covered = placement['node_membership']
    clear = last['clear'] and set(covered) == set(DEV_NODES.values()) and all(
        value == 'NO_NUMBERED_STEP' for value in covered.values())
    return dict(result, state='TRAINING_ASSETS_VERIFIED_DEVICE_PENDING' if clear else 'ORIGINAL_REQUEST_RECONCILIATION_REQUIRED',
                first=first, last=last, assets=facts, membership=placement, read_only=True, returned_epoch=time.time())


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    def expire(*_):
        raise TimeoutError('original Qwen preflight exceeded its 90-second budget')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(90)
    result = preflight(REPO, RESULTS)
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(result['state'] != 'TRAINING_ASSETS_VERIFIED_DEVICE_PENDING')


if __name__ == '__main__':
    raise SystemExit(main())
