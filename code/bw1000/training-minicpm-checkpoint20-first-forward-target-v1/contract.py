"""Frozen scope for one native MiniCPM checkpoint-20 microbatch on eight ranks."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys


SOURCE = 'training-minicpm-checkpoint20-first-forward-source-v1'
LABEL = 'training-minicpm-checkpoint20-first-forward-target-v1'
REQUEST = '09c51b48d46b45c6aeabbaa117160ac2'
ORIGINAL_LABEL = 'training-minicpm-usgs-exposure25-resume20-target-v2'
ORIGINAL_REQUEST = 'fa68d03ea04f4588ac9c21459097d50e'
PARENT, NODE, PORT = '1103606', 'f13r1n19', 29939
RESULTS = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results'
ORIGINAL = PurePosixPath(RESULTS) / ORIGINAL_LABEL
RUN = PurePosixPath(RESULTS) / LABEL
WHOLE, NATIVE, CLEANUP, GPU_CLIENT = 1200, 720, 840, 900
PHASES = {'pre': (0, 2, 2048, 60), 'gpu': (8, 64, 224000, GPU_CLIENT), 'post': (0, 2, 2048, 90)}
STAGE_SHA = 'ac4ca58990c955ec99db48b2e4cc8be4dac6e0fdefa59fbee3f5f3bf6784cd6d'
CONTROL_MANIFEST = 'coordination/training/minicpm-usgs-target-control-files-v3.json'
CONTROL_SHA = 'eb4e2de22895f7ede8273e5693829bfe9d025958b3fe7abcaf96586b574060cb'
SOURCE_SHA = 'b78196e08b364f64b4755474185b9d0118931ab01f074944af67bf9078b73046'
PUBLIC = 'results/bw1000/2026-11-22/training-minicpm-native88-structured-v3-1171-paired-recovery-terminal.json'
PUBLIC_SHA = '816e3561638b1625aa18038b8dfb50fc2e9c343570501cbde6a7c29839d5be86'
CHECKPOINT = 'continuous/train/checkpoint-20'
INVENTORY = 'continuous-checkpoint-20-files.json'
INVENTORY_SHA = 'c596dd8368f9798c0555e721bd63512f0087dc2546a22588adf128245c96a752'
INVENTORY_BYTES = 2131
MEMBERS = {'trainer_state.json', 'args.json', 'adapter_model.safetensors', 'optimizer.pt',
           'scheduler.pt', 'adapter_config.json'} | {f'rng_state_{rank}.pth' for rank in range(8)}
HEX = re.compile('[0-9a-f]{64}')
WORKER_FILES = ('contract.py', 'worker.py', 'native_observer.py', 'probe.py',
                'compare_first.py', 'original_worker.py', 'snapshot.py', 'owned_processes.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def read_pin(path, expected=None, size=None, cap=2*1024**2):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and path.resolve(strict=True) == path.absolute(),
         'expected regular canonical file: ' + path.name)
    need(0 < before.st_size <= cap and (size is None or size == before.st_size),
         'invalid bounded file size: ' + path.name)
    with path.open('rb') as stream:
        raw = stream.read(before.st_size + 1)
    after = path.lstat()
    names = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    need(len(raw) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in names)
         and (expected is None or expected == digest(raw)), 'input changed or wrong digest: ' + path.name)
    return raw


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_member(root, record, name, cap=2*1024**2):
    path = root / name
    need(record['path'] == str(path) and HEX.fullmatch(record['SHA256']), 'invalid original member identity')
    return read_pin(path, record['SHA256'], record['bytes'], cap)


def checkpoint(root, pin):
    """Structurally close exact archived identities without opening the tensors."""
    inventory = json.loads(exact_member(root, pin, INVENTORY, cap=16*1024))
    need(inventory['arm'] == 'continuous' and inventory['global_step'] == 20
         and inventory['member_count'] == 14 and set(inventory['files']) == MEMBERS
         and inventory['actual_tensor_deserialization_here'] is False
         and inventory['accumulation_boundary'] == 'native witness complete-four-micro-update',
         'original checkpoint-20 identity differs')
    for name, record in inventory['files'].items():
        path = root / CHECKPOINT / name
        meta = path.lstat()
        need(type(record['bytes']) is int and record['bytes'] > 0 and HEX.fullmatch(record['SHA256'])
             and stat.S_ISREG(meta.st_mode) and path.resolve(strict=True) == path.absolute()
             and meta.st_size == record['bytes'], 'original checkpoint member identity differs: ' + name)
    return inventory['files']


def admission(document):
    need(re.fullmatch('[0-9a-f]{32}', REQUEST) and REQUEST != ORIGINAL_REQUEST, 'bad new request')
    need(document['schema'] == 'MINICPM_CHECKPOINT20_FIRST_FORWARD_ADMISSION_V1'
         and document['request'] == REQUEST and document['original_step'] == '1103606.1171'
         and (document['parent'], document['node'], document['port']) == (PARENT, NODE, PORT)
         and document['root_GO'] is True and document['GPU_GO'] is True
         and re.fullmatch('[0-9a-f]{40}', document['publication'])
         and document['stage_SHA256'] == STAGE_SHA and document['planned_optimizer_updates'] == 0,
         'missing actual root adoption or original source binding')
    pin = document['checkpoint20_inventory']
    need(pin == dict(path=str(ORIGINAL / INVENTORY), bytes=INVENTORY_BYTES, SHA256=INVENTORY_SHA),
         'missing original checkpoint inventory pin')
    return pin


def render(plan, original=ORIGINAL, run=RUN):
    """Change only observer asset, output directory and local port."""
    argv = list(plan['command'])
    env = dict(plan['environment_overrides'])
    need(plan['arm'] == 'resumed' and argv[0] == str(original / 'framework/vlm/runtime/swift_entry.py')
         and env['NPROC_PER_NODE'] == '8' and env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
         and env['MASTER_PORT'] == '29929', 'wrong original resumed native launch')

    def option(flag, value, end=None):
        need(argv.count(flag) == 1, 'ambiguous original Swift argument: ' + flag)
        index = argv.index(flag) + 1
        need(index < len(argv) and argv[index] == value
             and (end is None or index < end), 'wrong original Swift argument: ' + flag)
        return index

    option('--resume_from_checkpoint', str(original / CHECKPOINT))
    option('--callbacks', 'minicpm_usgs_resume_witness')
    option('--max_steps', '25')
    witness = str(original / 'src/minicpm_usgs_witness.py')
    need(argv.count(witness) == 1 and argv.count('--external_plugins') == 1, 'ambiguous original witness')
    plugins = argv.index('--external_plugins') + 1
    end = next((i for i in range(plugins, len(argv)) if argv[i].startswith('--')), len(argv))
    witness_index = argv.index(witness)
    need(plugins <= witness_index < end, 'original witness is outside the Swift plugin list')
    argv[witness_index] = str(run / 'target-worker/native_observer.py')
    index = option('--output_dir', str(original / 'resumed/train'))
    argv[index] = str(run / 'resumed/train')
    env['MASTER_PORT'] = str(PORT)
    return dict(command=argv, environment_overrides=env)


def ready(root, config_sha, epoch, deadline, actual_step):
    from compare_first import probe_rows
    rows = probe_rows(root)
    for rank in range(8):
        marker = root / 'minicpm-usgs-witness' / f'rank-{rank:02d}' / 'first-forward-ready.json'
        info = json.loads(read_pin(marker, cap=4096))
        identity = dict(config_SHA256=config_sha, epoch=epoch, deadline=deadline, actual_step=actual_step)
        need(info == dict(rank=rank, world_size=8, checkpoint_step=20, run=identity,
                          state='FORWARD_RETURNED'), 'eight-rank ready identity differs')
    return rows
