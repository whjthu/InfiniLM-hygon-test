"""Read exactly sixteen original T03 JSON witnesses; never open a tensor file."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import time

ROOT = Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture')
ORIGINAL_EPOCH = 1794887170.6908448
ORIGINAL_DEADLINE = 1794887990.6908448  # Native capture gets epoch+820, not supervisor epoch+900.
BINDING = '2ff3b19fc433f38b2b9053787f1b458e70ada2457e1d1092f5beab3ae769a076'
MAX_FILE = 65536
MAX_TOTAL = 16 * MAX_FILE
MAX_OUTPUT = 16384


def need(condition, code):
    if not condition:
        raise ValueError(code)


def sha(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'INVALID_SHA256')
    return value


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def clock(origin):
    now = time.time()
    need(math.isfinite(origin) and 0 < origin <= now < origin + 45, 'ORIGIN_EPOCH_EXHAUSTED')
    return origin + 45 - now


def read_json(path, origin):
    clock(origin)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_FILE, 'METADATA_SIZE_OR_TYPE')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            body = stream.read(MAX_FILE + 1)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(identity(before) == identity(after) and len(body) == before.st_size, 'METADATA_CHANGED')
        need(identity(after) == identity(path.lstat()), 'METADATA_PATH_CHANGED')
        value = json.loads(body, object_pairs_hook=no_duplicates,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
        need(isinstance(value, dict), 'METADATA_OBJECT_REQUIRED')
        return value, dict(path=str(path), bytes=len(body), SHA256=hashlib.sha256(body).hexdigest())
    finally:
        os.close(fd)


def inspect_rank(rank, owner, progress):
    need(type(rank) is int and 0 <= rank < 8, 'RANK')
    need(owner.get('rank') == rank and owner.get('local_rank') == rank, 'OWNER_RANK')
    need(owner.get('accelerator') == 'cuda' and owner.get('binding_SHA256') == BINDING, 'ORIGINAL_BINDING')
    need(owner.get('epoch') == ORIGINAL_EPOCH and owner.get('deadline') == ORIGINAL_DEADLINE, 'ORIGINAL_SCOPE')
    for key in ('PID', 'PGID', 'startticks'):
        need(type(owner.get(key)) is int and owner[key] > 0, 'HISTORICAL_OWNER_IDENTITY')
    named = sha(owner.get('initial_trainables_SHA256'))
    expected = dict(rank=rank, world_size=8, planned_updates=2, planned_micro_forwards=8,
                    planned_native_global_step=25, completed_updates=2, inputs_saved=8,
                    outputs_saved=8, clip_calls=2, failed=False)
    for key, value in expected.items():
        need(type(progress.get(key)) is type(value) and progress[key] == value, 'CAPTURE_DENOMINATOR_' + key)
    files = progress.get('files')
    need(isinstance(files, list) and len(files) == 25, 'CAPTURE_FILES_25')
    names = []
    for item in files:
        need(isinstance(item, dict), 'CAPTURE_FILE_RECORD')
        name = item.get('name')
        need(isinstance(name, str) and name not in ('.', '..') and '/' not in name and '\\' not in name,
             'CAPTURE_BASENAME')
        need(type(item.get('bytes')) is int and 0 < item['bytes'] <= 2147483647, 'CAPTURE_FILE_BYTES')
        sha(item.get('SHA256'))
        names.append(name)
    need(len(set(names)) == 25 and names.count('initial.pt') == 1, 'INITIAL_MEMBER_UNIQUE')
    initial = files[names.index('initial.pt')]
    need(174587904 <= initial['bytes'] <= 256 * 1024**2, 'INITIAL_REFERENCE_BYTES')
    return dict(rank=rank, initial_trainables_named_SHA256=named,
                initial=dict(path=str(ROOT / ('rank-%02d' % rank) / 'initial.pt'),
                             bytes=initial['bytes'], SHA256=initial['SHA256']),
                historical_owner={k: owner[k] for k in ('PID', 'PGID', 'startticks')},
                capture_artifacts=25, captured_optimizer_updates=2, captured_micro_forwards=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', type=float, required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--root-go', action='store_true')
    args = parser.parse_args()
    need(args.root_go and re.fullmatch('[0-9a-f]{32}', args.request), 'ROOT_ONCE_CONTEXT_REQUIRED')
    remaining = clock(args.origin)
    available = sorted(cpu for cpu in os.sched_getaffinity(0) if cpu not in (0, 1))
    need(len(available) >= 2, 'TWO_CPUS_EXCLUDING_ZERO_ONE_REQUIRED')
    os.sched_setaffinity(0, set(available[:2]))
    need(os.sched_getaffinity(0) == set(available[:2]), 'CPU_AFFINITY_FAILED')
    def expired(*_):
        raise TimeoutError('ORIGINAL_45_SECOND_EPOCH_EXHAUSTED')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, clock(args.origin))
    ranks, pins = [], []
    try:
        for rank in range(8):
            folder = ROOT / ('rank-%02d' % rank)
            owner, op = read_json(folder / 'owner.json', args.origin)
            progress, pp = read_json(folder / 'progress.json', args.origin)
            pins.extend((op, pp))
            ranks.append(inspect_rank(rank, owner, progress))
        need(len(pins) == 16 and sum(p['bytes'] for p in pins) <= MAX_TOTAL, 'EXACT_16_METADATA_BUDGET')
        need(len({r['initial_trainables_named_SHA256'] for r in ranks}) == 1, 'ALL_EIGHT_INITIAL_VALUES_DIFFER')
        clock(args.origin)
        result = dict(passed=True, state='ORIGINAL_T03_INITIAL_METADATA_PINS_CLOSED',
                      request_id=args.request, origin=args.origin, deadline=args.origin + 45,
                      finished_epoch=time.time(), CPU_affinity=sorted(os.sched_getaffinity(0)),
                      metadata_files=16, metadata_bytes=sum(p['bytes'] for p in pins),
                      all_eight_named_SHA256_equal=True, ranks=ranks, files=pins,
                      tensor_body_bytes_read=0, tensor_body_SHA_recomputed=False,
                      original_values_qualification='Reuse published original200 CPU audit; current read only binds its historical metadata.',
                      historical_owner_liveness_checked=False, model_loaded=False,
                      GPU_queries=0, GPU_cards=0, training_updates=0,
                      formal17_GO=False, domain_training_GO=False)
        encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        need(len(encoded) <= MAX_OUTPUT, 'OUTPUT_CAP')
        print(encoded.decode())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
