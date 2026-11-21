"""Eight-card, guard-checked pinned HIP copy overlap on one Slurm node."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes as C
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
HIP_SHA256 = 'b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b'
HIP_PATH = '/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0'
SOURCE = (('hip_runtime.py', 'e551c1faa38e6472be7861a572354c72311a9b21d4ad2e73d7bb0a3fa403d534'), ('node_sample.py', 'e10a17f576da010cd50937a8f91eac1c8f66098dde6d887c8373d00a615c8931'),
          ('qualified_sampler.py', '09fc94b7f527c159bd489bacbf7718dad8615ba61cd3f361a44f441da14a0430'))
KFD = Path('/sys/class/kfd/kfd/proc')
SIZES = (1024**2, 32 * 1024**2)
ARMS = ('serial', 'overlap', 'overlap', 'serial')
WARM, SAMPLES, COPIES = 3, 12, 8
GUARD = 256
GUARDS = (b'\xa5' * GUARD, b'\x5a' * GUARD)
H2D, D2H = 1, 2


def need(condition, name):
    if not condition:
        raise ValueError(name)


for dependency, expected_pin in SOURCE:
    path = HERE / dependency
    need(path.resolve(strict=True) == path and sha256(path.read_bytes()).hexdigest() == expected_pin,
         'REUSED_SOURCE_SHA256')
sys.path.insert(0, str(HERE))
from hip_runtime import DlInfo, Hip, I, S, U, V, add, digest, file_pin, pci_identity, symbol
from node_sample import fresh_inventory


def save(path, value):
    with path.open('x', encoding='utf8') as output:
        json.dump(value, output, allow_nan=False, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())


def bind():
    path, pin = file_pin(dict(path=HIP_PATH, sha256=HIP_SHA256), exact_dtk_hip=True)
    loader = C.CDLL('libdl.so.2')
    lookup = loader.dladdr
    lookup.argtypes, lookup.restype = [V, C.POINTER(DlInfo)], I
    library = C.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    hip = Hip(library, lookup, path)
    extra = {'HostMalloc': [C.POINTER(V), S, U], 'HostFree': [V], 'MemcpyAsync': [V, V, S, I, V]}
    hip.functions.update({name: symbol(library, lookup, 'hip' + name, args, path) for name, args in extra.items()})
    need(digest(path) == pin, 'MAPPED_HIP_IDENTITY_CHANGED')
    return hip


def pci_at(hip, index):
    value = C.create_string_buffer(32)
    hip.call('DeviceGetPCIBusId', value, len(value), index)
    return pci_identity(value.value)


def original_affinity(pcis):
    available = set(os.sched_getaffinity(0))
    need(len(available) >= 8, 'EIGHT_DISJOINT_OWNER_CPUS_REQUIRED')
    placements = []
    for pci in pcis:
        path = Path('/sys/bus/pci/devices') / pci
        local = set()
        for entry in (path / 'local_cpulist').read_text().strip().split(','):
            bounds = entry.split('-')
            if entry:
                local.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        matching = local & available
        cpu = min(matching or available)
        placements.append((cpu, bool(matching), int((path / 'numa_node').read_text())))
        available.remove(cpu)
    return placements


def new_buffer(hip, owners, host, size):
    first = V()
    operation = 'HostMalloc' if host else 'Malloc'
    hip.call(operation, C.byref(first), size + 2 * GUARD, *((0,) if host else ()))
    owners.callback(hip.call, 'HostFree' if host else 'Free', first)
    data = add(first, GUARD)
    return first, data, add(data, size)


def prime(hip, src, dst, gpu_out, gpu_in, size, seed):
    generator = random.Random(seed)
    patterns = tuple(generator.randbytes(65536) * (size // 65536) for _ in range(2))
    C.memmove(src[1], patterns[0], size)
    C.memset(dst[1], 0, size)
    hip.call('Memcpy', gpu_in[1], patterns[1], size, H2D)
    for owner in (src, dst, gpu_out, gpu_in):
        for ptr, guard in zip((owner[0], owner[2]), GUARDS, strict=True):
            if owner is src or owner is dst:
                C.memmove(ptr, guard, GUARD)
            else:
                hip.call('Memcpy', ptr, guard, GUARD, H2D)
    return [sha256(pattern).hexdigest() for pattern in patterns]


def audit(hip, src, dst, gpu_out, gpu_in, size, expected):
    staging = C.create_string_buffer(size)
    for buffer, pin, host in zip((src, gpu_out, dst, gpu_in),
                                 (expected[0], expected[0], expected[1], expected[1]),
                                 (True, False, True, False), strict=True):
        if host:
            payload = buffer[1]
        else:
            hip.call('Memcpy', staging, buffer[1], size, D2H)
            payload = staging
        need(sha256(C.string_at(payload, size)).hexdigest() == pin, 'COPY_PAYLOAD_SHA256')
        for ptr, value in zip((buffer[0], buffer[2]), GUARDS, strict=True):
            if host:
                guard = C.string_at(ptr, GUARD)
            else:
                hip.call('Memcpy', staging, ptr, GUARD, D2H)
                guard = C.string_at(staging, GUARD)
            need(guard == value, 'COPY_REDZONE')


def elapsed_ms(hip, stream):
    value = C.c_float()
    hip.call('EventElapsedTime', C.byref(value), stream[1], stream[2])
    need(math.isfinite(value.value) and value.value > 0, 'EVENT_INTERVAL')
    return value.value


def batch(hip, stream_up, stream_down, src, dst, gpu_out, gpu_in, size, mode):
    def enqueue(stream, target, source, direction):
        owner, first, last = stream
        hip.call('EventRecord', first, owner)
        for _ in range(COPIES):
            hip.call('MemcpyAsync', target, source, size, direction, owner)
        hip.call('EventRecord', last, owner)

    require_serial = mode == 'serial'
    start = time.perf_counter_ns()
    enqueue(stream_up, gpu_out[1], src[1], H2D)
    if require_serial:
        hip.call('StreamSynchronize', stream_up[0])
    enqueue(stream_down, dst[1], gpu_in[1], D2H)
    if not require_serial:
        hip.call('StreamSynchronize', stream_up[0])
    hip.call('StreamSynchronize', stream_down[0])
    end = time.perf_counter_ns()
    need(end > start, 'COPY_WALL_INTERVAL')
    return dict(begin_ns=start, end_ns=end, combined_bytes=2 * COPIES * size,
                h2d_stream_ms=elapsed_ms(hip, stream_up), d2h_stream_ms=elapsed_ms(hip, stream_down))


def run_case(hip, size, seed, rendezvous, report, progress):
    with ExitStack() as owners:
        src, dst = (new_buffer(hip, owners, True, size) for _ in range(2))
        gpu_out, gpu_in = (new_buffer(hip, owners, False, size) for _ in range(2))
        report['memory_after_alloc'] = hip.mem()
        pins = prime(hip, src, dst, gpu_out, gpu_in, size, seed)
        report['source_SHA256'] = pins
        with hip.stream() as up, hip.stream() as down:
            for ordinal, mode in enumerate(ARMS):
                arm = dict(mode=mode, samples=[], warm_completed=0, payload_and_guards_passed=False)
                report['arms'].append(arm)
                for attempt in range(WARM + SAMPLES):
                    if attempt == WARM:
                        C.memset(dst[1], 0, size)
                        hip.call('MemsetAsync', gpu_out[1], 0, size, up[0])
                        hip.call('StreamSynchronize', up[0])
                    rendezvous()
                    row = batch(hip, up, down, src, dst, gpu_out, gpu_in, size, mode)
                    if attempt < WARM:
                        arm['warm_completed'] += 1
                    else:
                        arm['samples'].append(row)
                rendezvous()
                audit(hip, src, dst, gpu_out, gpu_in, size, pins)
                arm['payload_and_guards_passed'] = True
                progress(dict(bytes=size, arm_index=ordinal, result=arm))
        report['passed'] = True


def node_rates(owners):
    rates = []
    for index, size in enumerate(SIZES):
        cell = dict(bytes_per_copy=size, combined_GB_s={'serial': [], 'overlap': []})
        for arm_index, mode in enumerate(ARMS):
            values = []
            for occurrence in range(SAMPLES):
                records = [row['cases'][index]['arms'][arm_index]['samples'][occurrence] for row in owners]
                combined = sum(row['combined_bytes'] for row in records)
                span_ns = max(row['end_ns'] for row in records) - min(row['begin_ns'] for row in records)
                values.append(combined / span_ns)
            cell['combined_GB_s'][mode].append(values)
        medians = {mode: statistics.median(val for series in groups for val in series)
                   for mode, groups in cell['combined_GB_s'].items()}
        cell['median_node_GB_s'] = medians
        cell['overlap_over_serial'] = medians['overlap'] / medians['serial']
        rates.append(cell)
    return rates


def main(output, deadline, expected):
    need(math.isfinite(deadline) and 0 < deadline - time.time() <= 240, 'ORIGINAL_SINGLE_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    node = socket.gethostname().split('.', 1)[0]
    parent, step = os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID']
    step_id = parent + '.' + step
    need(parent in ('1087656', '1103606') and step.isdecimal(), 'NUMBERED_OWNED_STEP_REQUIRED')
    expected_pci = expected[node]
    inventory = fresh_inventory()
    need(len(set(expected_pci)) == 8 and set(expected_pci) == set(inventory)
         and not any(KFD.iterdir()), 'FRESH_PCI_KFD_IS_NOT_CLEAR')
    output = output / node
    output.mkdir(mode=0o700)
    halt = threading.Event()
    barrier = threading.Barrier(8)
    signal.signal(signal.SIGUSR1, lambda *_: halt.set())

    def rendezvous():
        if halt.is_set() or time.time() >= deadline - 8:
            raise InterruptedError('COOPERATIVE_HANDOFF_OR_DEADLINE')
        barrier.wait(timeout=max(0.001, min(25, deadline - time.time() - 8)))

    hip = bind()
    need(hip.scalar('GetDeviceCount') == 8, 'EIGHT_ACTUAL_HIP_DEVICES_REQUIRED')
    mapping = [pci_at(hip, card) for card in range(8)]
    need(len(set(mapping)) == 8 and set(mapping) == set(expected_pci), 'HIP_PCI_BIJECTION')
    affinity = original_affinity(mapping)
    save(output / 'intent.json', dict(node=node, step=step_id, PCI=mapping, HIP_SHA256=HIP_SHA256,
                                     deadline=deadline, planned_cards=8, sizes=SIZES, arm_order=ARMS,
                                     warm=WARM, samples=SAMPLES, copies_per_direction=COPIES))

    def work(card):
        cpu, local, numa_node = affinity[card]
        pci = mapping[card]
        row = dict(node=node, step=step_id, card=card, PCI=pci, owner_cpu=cpu, PCI_numa_node=numa_node,
                   owner_cpu_in_PCI_local_cpulist=local, cases=[], normal_release=False, error=None)
        try:
            os.sched_setaffinity(0, {cpu})
            hip.call('SetDevice', card)
            need(hip.scalar('GetDevice') == card and pci_at(hip, card) == pci, 'GPU_OWNER_DEVICE_CHANGED')
            row['runtime_version'] = hip.scalar('RuntimeGetVersion')
            row['driver_version'] = hip.scalar('DriverGetVersion')
            row['memory_before'] = hip.mem()
            with (output / f'card-{card}.progress.jsonl').open('x', encoding='utf8') as journal:
                def progress(value):
                    journal.write(json.dumps(value, allow_nan=False, sort_keys=True) + '\n')
                    journal.flush()
                    os.fsync(journal.fileno())

                for size in SIZES:
                    case = dict(bytes=size, expected_warm_per_arm=WARM, expected_samples_per_arm=SAMPLES, arms=[],
                                passed=False)
                    row['cases'].append(case)
                    seed = int.from_bytes(sha256(f'{pci}:{size}'.encode()).digest()[:8], 'little')
                    run_case(hip, size, seed, rendezvous, case, progress)
            row['memory_after_release'] = hip.mem()
            row['normal_release'] = True
        except Exception as error:
            barrier.abort()
            row['error'] = dict(type=type(error).__name__, reason=str(error)[:128])
        save(output / f'card-{card}.json', row)
        return row

    with ThreadPoolExecutor(max_workers=8) as workers:
        owners = list(workers.map(work, range(8)))
    good = sum(row['normal_release'] for row in owners)
    result = dict(node=node, step=step_id, planned_cards=8, qualified_cards=good,
                  cooperative_stop_requested=halt.is_set(), normal_release_count=good)
    if good == 8:
        result['node_rates'] = node_rates(owners)
    save(output / 'terminal.json', result)
    signal.setitimer(signal.ITIMER_REAL, 0)
    return int(good != 8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('deadline', type=float)
    parser.add_argument('admitted_pci_json', type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.output, args.deadline, json.loads(args.admitted_pci_json.read_bytes())))
