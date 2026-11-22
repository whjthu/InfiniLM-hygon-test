"""Lease one Slurm step, then let three independent eight-rank native parents run."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from bracket_contract import (AB, ARM_CLEANUP, ARM_FLOOR, CLEANUP, INVENTORY, NATIVE, NODE, ORIGINAL, PARENT,
                              PLANNED, PORT, REQUEST, RUN, STAGED, OLD_OBSERVER, WHOLE, import_file,
                              need, observed_ranks, read, render)


HERE = Path(__file__).resolve().parent


def error_info(error):
    return dict(type=type(error).__name__, message=str(error)[:350])


def terminate(*_):
    raise RuntimeError('owned MiniCPM RNG bracket worker received TERM')


def finish_child(directory, owned, registry, child, budget, arm, rc):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    errors = []
    cleanup = None
    cap = min(AB[arm] + ARM_CLEANUP, CLEANUP)
    try:
        available = budget.left(cap, reserve=5)
    except ValueError as error:
        available = 0
        errors.append(error_info(error))
    try:
        cleanup = owned.cleanup(registry, directory, grace=min(5, available / 2))
    except BaseException as error:
        errors.append(error_info(error))
    if child is not None:
        try:
            child.wait(timeout=min(3, budget.left(cap, reserve=1)))
            rc = child.returncode
        except BaseException as error:
            errors.append(error_info(error))
    return rc, cleanup, errors


def run_arm(root, plan, config, witness, arm, budget, owned, save):
    budget.left(AB[arm], reserve=ARM_FLOOR)
    directory = root / arm
    directory.mkdir(mode=0o700)
    env = dict(os.environ, **plan['environment_overrides'])
    ranked = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'NODE_RANK', 'NNODES')
    need(not any(key in env for key in ranked), 'native Swift parent must start unranked')
    env.update(MASTER_ADDR='127.0.0.1', MINICPM_USGS_CONFIG=witness['path'],
               MINICPM_USGS_CONFIG_SHA256=witness['SHA256'], MINICPM_USGS_EPOCH=str(config['epoch']),
               MINICPM_USGS_DEADLINE=str(config['native_deadline']), MINICPM_USGS_MAX_SECONDS=str(NATIVE),
               MINICPM_FIRST_FORWARD_FROZEN_WITNESS=str(ORIGINAL / 'src/minicpm_usgs_witness.py'),
               MINICPM_FIRST_FORWARD_ORIGINAL_OBSERVER=str(HERE / 'original-observer'))
    need(env['MINICPM_RNG_TREATMENT'] == arm and env['MASTER_PORT'] == str(PORT), 'arm launch changed')
    with socket.socket() as port:
        port.bind(('127.0.0.1', PORT))
    save(directory / 'native-intent.json', dict(request=REQUEST, arm=arm, epoch=config['epoch'],
         arm_deadline=config['epoch'] + AB[arm], common_native_deadline=config['native_deadline'],
         planned_forwards=8, planned_optimizer_updates=0, witness_SHA256=witness['SHA256'],
         plan_SHA256=sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()))
    registry = {}
    child = failure = rc = None
    start = time.monotonic()
    try:
        budget.left(AB[arm], reserve=2)
        with (directory / 'native.log').open('xb') as log:
            child = subprocess.Popen([sys.executable, '-B', *plan['command']], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            entry = owned.snapshot().get(child.pid)
            need(entry is not None, 'native child strong identity unavailable')
            registry[child.pid] = {key: entry[key] for key in ('PID', 'PPID', 'PGID', 'startticks', 'state')}
            save(directory / 'child-owner.json', dict(request=REQUEST, boot_id=config['boot_id'],
                                                      identity=registry[child.pid]))
            while child.poll() is None:
                owned.refresh(registry, directory)
                budget.left(AB[arm], reserve=1)
                pending = directory / 'owned-registry.tmp'
                pending.write_text(json.dumps(registry), encoding='utf-8')
                pending.replace(directory / 'owned-registry.json')
                time.sleep(.25)
            rc = child.returncode
            until = time.monotonic() + min(5, budget.left(AB[arm], reserve=1))
            while rc == 0 and owned.refresh(registry, directory) and time.monotonic() < until:
                time.sleep(.1)
    except BaseException as error:
        failure = error_info(error)
    finally:
        rc, cleanup, cleanup_errors = finish_child(directory, owned, registry, child, budget, arm, rc)
        save(directory / 'owned-registry-final.json', registry)

    present = observed_ranks(directory)
    passed = (rc == 0 and not failure and not cleanup_errors and len(present) == 8
              and cleanup == dict(owned_processes_remaining=0, termination_needed=False))
    result = dict(request=REQUEST, arm=arm, rc=rc, error=failure, cleanup=cleanup, cleanup_errors=cleanup_errors,
                  cleanup_known=cleanup is not None and not cleanup_errors, owners=registry,
                  attempt_registered=True, planned_forwards=8, completed_receipt_ranks=present,
                  missing_receipt_ranks=sorted(set(range(8)) - set(present)), started_child=child is not None,
                  wall_seconds=time.monotonic() - start, passed=passed)
    save(directory / 'native-outcome.json', result)
    return result


def consume(root, step):
    saved_path = sys.path[:]
    sys.path.insert(0, str(HERE / 'original-observer'))
    try:
        from compare_first import original_rows
        from compare_bracket import consume as compare
        records = original_rows(Path(str(ORIGINAL)), HERE / 'original1171.json')
        return compare(root, step, HERE / 'original1189-route.json', records)
    finally:
        sys.path[:] = saved_path


def run(config, root, original, old, first, budget):
    fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    identity = dict(request=REQUEST, host=socket.gethostname(), PID=os.getpid(), PGID=os.getpgrp(),
                    startticks=int(fields[19]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    epoch=config['epoch'], deadline=config['deadline'])
    old.save(root / 'worker-owner.json', identity)
    config['boot_id'] = identity['boot_id']
    result = dict(request=REQUEST, epoch=config['epoch'], deadline=config['deadline'],
                  planned_arms=list(AB), planned_forwards=PLANNED, planned_optimizer_updates=0,
                  arms=[], planned_ranks_per_arm=8, passed=False, formal_recovery_qualified=False)
    step = None
    signal.signal(signal.SIGTERM, terminate)
    try:
        need(identity['host'].split('.')[0] == NODE and os.environ['SLURM_JOB_ID'] == PARENT
             and re.fullmatch(r'\d+', os.environ['SLURM_STEP_ID'])
             and os.environ['SLURM_CPUS_PER_TASK'] == '64', 'wrong eight-rank Slurm scope')
        step = PARENT + '.' + os.environ['SLURM_STEP_ID']
        device = import_file('bracket_snapshot', HERE / 'snapshot.py')
        owned = import_file('bracket_owned', HERE / 'owned_processes.py')
        before = old.checked_record(config['before'], root)
        leased = device.snapshot(True)
        affinity = sorted(os.sched_getaffinity(0))
        old.check_snapshots(before, leased, len(affinity))
        need(before['pci_to_kfd'] is None and sorted(leased['pci_to_kfd']) == before['pci']
             and before['master_port'] == PORT, 'leased KFD/PCI or port differs')
        order = first.gpu_order(leased['pci'])
        old.save(root / 'worker-assigned.json', dict(identity, actual_step=step, visible_PCI_by_index=order,
                 leased_snapshot=leased, CPU_affinity=affinity))
        plan = old.source_and_data(config['stage'])['resumed']
        pins = first.checkpoint(original, INVENTORY)
        witness_path = root / 'shared-witness-config.json'
        old.save(witness_path, dict(asset_binding=str(original / 'src/ASSETS.json'),
                 checkpoint=str(original / first.CHECKPOINT), checkpoint_files=pins))
        witness = dict(path=str(witness_path), SHA256=sha256(read(witness_path)).hexdigest())
        for arm in AB:
            signal.signal(signal.SIGTERM, terminate)
            arm_result = run_arm(root, render(first, plan, arm), config, witness, arm, budget, owned, old.save)
            result['arms'].append(arm_result)
            need(arm_result['passed'], 'native arm incomplete; later arms withheld')
            need(first.gpu_order(leased['pci']) == order, 'rank-to-PCI mapping changed after an arm')
    except BaseException as error:
        result['error'] = error_info(error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        result['not_attempted_arms'] = [arm for arm in AB if not (root / arm / 'native-intent.json').is_file()]
        result['receipt_presence_by_arm'] = {arm: observed_ranks(root / arm) for arm in AB}
        try:
            if step is not None:
                budget.left(NATIVE)
                evidence = consume(root, step)
                old.save(root / 'rng-bracket-comparison.json', evidence)
                result['passed'] = (not result.get('error') and evidence['complete'] and
                                    len(result['arms']) == len(AB) and all(arm['passed'] for arm in result['arms']))
        except Exception as error:
            result['comparison_error'] = error_info(error)
        old.save(root / 'worker-outcome.json', result)
    return int(not result['passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    root, original = Path(str(RUN)), Path(str(ORIGINAL))
    need(args.config == root / 'worker-config.json' and HERE == root / 'target-worker', 'wrong new worker namespace')
    config = json.loads(read(args.config, args.config_sha256))
    need(config['request'] == REQUEST and config['root_GO'] is True and config['GPU_GO'] is True
         and config['planned_forwards'] == PLANNED and config['planned_optimizer_updates'] == 0
         and config['arm_deadlines'] == {arm: config['epoch']+offset for arm, offset in AB.items()},
         'wrong new worker configuration')
    names = STAGED | {'original-observer/' + name for name in OLD_OBSERVER}
    need(set(config['workers']) == names, 'worker file set changed')
    for name, pin in config['workers'].items():
        read(HERE / name, pin['SHA256'], pin['bytes'])
    old = import_file('bracket_original_worker', HERE / 'original_worker.py')
    first = import_file('bracket_first_worker', HERE / 'first_worker.py')
    old.WHOLE, old.NATIVE = WHOLE, NATIVE
    budget = old.Budget(config['epoch'], config['deadline'], config['native_deadline'])
    return run(config, root, original, old, first, budget)


if __name__ == '__main__':
    raise SystemExit(main())
