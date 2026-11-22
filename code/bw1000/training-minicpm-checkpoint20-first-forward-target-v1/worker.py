"""Run the archived native MiniCPM resume through its first forward only."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from contract import (CHECKPOINT, CLEANUP, NATIVE, NODE, ORIGINAL, PARENT, PORT, PUBLIC_SHA,
                      REQUEST, RUN, STAGE_SHA, WHOLE, WORKER_FILES, checkpoint, digest, import_file,
                      need, read_pin, ready, render)


HERE = Path(__file__).resolve().parent


def gpu_order(pci):
    hip = ctypes.CDLL('libamdhip64.so')
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    count = ctypes.c_int()
    need(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 8, 'actual HIP count differs')
    order = []
    for rank in range(8):
        value = ctypes.create_string_buffer(32)
        need(hip.hipDeviceGetPCIBusId(value, len(value), rank) == 0, 'actual HIP PCI unavailable')
        name = value.value.decode().lower()
        if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', name):
            name = '0000:' + name
        need(re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', name), 'invalid HIP PCI')
        order.append(name)
    need(len(set(order)) == 8 and sorted(order) == pci, 'leased HIP PCI differs')
    return order


def observation(error):
    return dict(type=type(error).__name__, message=str(error)[:350])


def run_forward(root, original, save, plan, config, pins, budget, owned, actual_step):
    budget.left(NATIVE, reserve=30)
    directory = root / 'resumed'
    directory.mkdir(mode=0o700)
    cfg = root / 'resumed-witness-config.json'
    save(cfg, dict(asset_binding=str(original / 'src/ASSETS.json'),
                   checkpoint=str(original / CHECKPOINT), checkpoint_files=pins))
    cfg_sha = digest(read_pin(cfg))
    env = dict(os.environ, **plan['environment_overrides'])
    ranked = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'NODE_RANK', 'NNODES')
    need(not any(key in env for key in ranked), 'native parent must start unranked')
    env.update(MASTER_ADDR='127.0.0.1', MINICPM_USGS_CONFIG=str(cfg), MINICPM_USGS_CONFIG_SHA256=cfg_sha,
               MINICPM_USGS_EPOCH=str(config['epoch']), MINICPM_USGS_DEADLINE=str(config['native_deadline']),
               MINICPM_USGS_MAX_SECONDS=str(NATIVE),
               MINICPM_FIRST_FORWARD_FROZEN_WITNESS=str(original / 'src/minicpm_usgs_witness.py'))
    save(directory / 'native-intent.json', dict(request=REQUEST, arm='resumed', epoch=config['epoch'],
         deadline=config['native_deadline'], planned_optimizer_updates=0, planned_forwards_per_rank=1,
         native_plan_SHA256=digest(json.dumps(plan, sort_keys=True).encode())))
    registry = {}
    child = failure = rc = cleanup = None
    errors = []
    try:
        budget.left(NATIVE, reserve=2)
        with (directory / 'native.log').open('xb') as log:
            child = subprocess.Popen([sys.executable, '-B', *plan['command']], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            entry = owned.snapshot().get(child.pid)
            need(entry is not None, 'native Popen child strong identity unavailable')
            registry[child.pid] = {k: entry[k] for k in ('PID', 'PPID', 'PGID', 'startticks', 'state')}
            save(directory / 'child-owner.json', dict(request=REQUEST, boot_id=config['boot_id'],
                                                     identity=registry[child.pid]))
            while child.poll() is None:
                owned.refresh(registry, directory)
                budget.left(NATIVE, reserve=1)
                temporary = directory / 'owned-registry.tmp'
                temporary.write_text(json.dumps(registry), 'utf-8')
                temporary.replace(directory / 'owned-registry.json')
                time.sleep(.25)
            rc = child.returncode
            until = time.monotonic() + min(5, budget.left(NATIVE, reserve=1))
            while rc == 0 and owned.refresh(registry, directory) and time.monotonic() < until:
                time.sleep(.1)
    except BaseException as error:
        failure = observation(error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            available = budget.left(CLEANUP, reserve=4)
        except ValueError as error:
            available = 0
            errors.append(observation(error))
        try:
            cleanup = owned.cleanup(registry, directory, grace=min(5, available / 2))
        except BaseException as error:
            errors.append(observation(error))
        if child is not None:
            try:
                child.wait(timeout=min(3, budget.left(CLEANUP, reserve=1)))
                rc = child.returncode
            except BaseException as error:
                errors.append(observation(error))
        save(directory / 'owned-registry-final.json', registry)
    ranks = []
    for rank in range(8):
        path = directory / 'train/minicpm-usgs-witness' / f'rank-{rank:02d}'
        ranks.append(dict(rank=rank, planned_forwards=1,
                          forward_state_present=(path / 'first-forward-state.json').is_file(),
                          ready_present=(path / 'first-forward-ready.json').is_file(),
                          stop_present=(path / 'first-forward-stop.json').is_file()))
    result = dict(request=REQUEST, arm='resumed', rc=rc, error=failure, cleanup=cleanup,
                  cleanup_errors=errors, cleanup_known=cleanup is not None and not errors,
                  owners=registry, ranks=ranks, planned_forwards=8, planned_optimizer_updates=0,
                  attempted_child=child is not None, config_SHA256=cfg_sha, passed=False)
    try:
        need(rc == 0 and failure is None and not errors
             and cleanup == dict(owned_processes_remaining=0, termination_needed=False), 'native worker did not exit cleanly')
        records = ready(directory / 'train', cfg_sha, config['epoch'], config['native_deadline'], actual_step)
        from compare_first import compare, original_rows
        result['comparison'] = compare(records, original_rows(original, HERE / 'original1171.json'))
        result['passed'] = True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result['gate_error'] = observation(error)
    save(directory / 'native-outcome.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    root, original = Path(str(RUN)), Path(str(ORIGINAL))
    need(args.config == root / 'worker-config.json' and HERE == root / 'target-worker', 'wrong new worker namespace')
    cfg = json.loads(read_pin(args.config, args.config_sha256))
    need(cfg['request'] == REQUEST and cfg['root_GO'] is True and cfg['GPU_GO'] is True
         and cfg['stage']['asset_stage']['SHA256'] == STAGE_SHA
         and cfg['declared_optimizer_updates'] == 0, 'new worker config differs')
    need(set(cfg['workers']) == set(WORKER_FILES), 'worker bundle differs')
    for name, pin in cfg['workers'].items():
        read_pin(HERE / name, pin['SHA256'], pin['bytes'])
    read_pin(HERE / 'original1171.json', PUBLIC_SHA)
    old = import_file('old_minicpm_worker', HERE / 'original_worker.py')
    # Only the clock arithmetic is reused under the new budget. The archived
    # worker's source verification below retains its original root and request.
    old.WHOLE, old.NATIVE = WHOLE, NATIVE
    budget = old.Budget(cfg['epoch'], cfg['deadline'], cfg['native_deadline'])
    fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    owner = dict(request=REQUEST, host=socket.gethostname(), PID=os.getpid(), PGID=os.getpgrp(),
                 startticks=int(fields[19]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 epoch=cfg['epoch'], deadline=cfg['deadline'])
    old.save(root / 'worker-owner.json', owner)
    cfg['boot_id'] = owner['boot_id']

    def terminate(*_):
        raise RuntimeError('native first-forward worker received TERM')

    signal.signal(signal.SIGTERM, terminate)
    outcome = dict(request=REQUEST, epoch=cfg['epoch'], deadline=cfg['deadline'], planned_ranks=8,
                   planned_forwards_per_rank=1, planned_optimizer_updates=0, arms=[], passed=False)
    try:
        need(owner['host'].split('.')[0] == NODE and os.environ['SLURM_JOB_ID'] == PARENT
             and re.fullmatch(r'\d+', os.environ['SLURM_STEP_ID'])
             and os.environ['SLURM_CPUS_PER_TASK'] == '64', 'wrong owned Slurm shape')
        step = PARENT + '.' + os.environ['SLURM_STEP_ID']
        # Old source/data verification may take more than the snapshot age
        # allowance. Check the device lease first, and load no model yet.
        device = import_file('minicpm_probe_device', HERE / 'snapshot.py')
        owned = import_file('minicpm_probe_owned', HERE / 'owned_processes.py')
        before = old.checked_record(cfg['before'], root)
        current = device.snapshot(True)
        affinity = sorted(os.sched_getaffinity(0))
        old.check_snapshots(before, current, len(affinity))
        need(before['pci_to_kfd'] is None and sorted(current['pci_to_kfd']) == before['pci']
             and before['master_port'] == PORT, 'leased PCI/KFD or port binding differs')
        with socket.socket() as port:
            port.bind(('127.0.0.1', PORT))
        old.save(root / 'worker-assigned.json', dict(owner, actual_step=step, visible_PCI_by_index=gpu_order(current['pci']),
                 leased_snapshot=current, CPU_affinity=affinity))
        plans = old.source_and_data(cfg['stage'])
        pins = checkpoint(original, cfg['checkpoint20_inventory'])
        arm = run_forward(root, original, old.save, render(plans['resumed']), cfg, pins, budget, owned, step)
        outcome['arms'].append(arm)
        outcome['passed'] = arm['passed']
    except BaseException as error:
        outcome['error'] = observation(error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        old.save(root / 'worker-outcome.json', outcome)
    return int(not outcome['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
