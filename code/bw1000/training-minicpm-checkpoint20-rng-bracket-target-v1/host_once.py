"""Use the qualified MiniCPM HOST step control for one new RNG bracket."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bracket_contract import (AB, CLEANUP, FIRST, FIRST_SHA, GPU_CLIENT, INVENTORY, LABEL, NATIVE, NODE,
                              OLD_OBSERVER, ORIGINAL, PARENT, PHASES, PLANNED, PORT, REQUEST, ROUTE, ROUTE_SHA,
                              RUN, SOURCE, SOURCE_SHA, STAGE_SHA, STAGED, TARGET_FILES, TERMINAL, TERMINAL_SHA,
                              WHOLE, admission, frozen, import_file, need, read, render, require_original)


REPO = HERE.parents[2]


def verified(repo):
    first_files = frozen(repo / FIRST, FIRST_SHA)
    rng_source = frozen(repo / SOURCE, SOURCE_SHA)
    baseline = read(repo / TERMINAL, TERMINAL_SHA, 7710)
    route = read(repo / ROUTE, ROUTE_SHA, 1911)
    require_original(baseline, route)
    previous = import_file('minicpm_first_forward_host', repo / FIRST / 'host_once.py')
    archived = previous.verified_sources(repo)
    bundle = {('original-observer/' + name if name in OLD_OBSERVER else name): raw
              for name, raw in archived.items()}
    bundle['first_worker.py'] = first_files['worker.py']
    bundle['contract.py'] = first_files['contract.py']
    bundle['original1189-route.json'] = route
    for name in ('rng_treatment.py', 'native_rng_observer.py', 'compare_bracket.py'):
        bundle[name] = rng_source[name]
    return previous, bundle


def bind(previous, old):
    # Rebind the previously qualified container command renderer in its private
    # module. The original helper sources and archived run stay immutable.
    previous.RUN = RUN
    old.RUN = Path(str(RUN))
    old.LABEL, old.REQUEST, old.PORT = LABEL, REQUEST, PORT
    old.PHASES, old.WHOLE, old.NATIVE, old.WORKER_CLEANUP = PHASES, WHOLE, NATIVE, CLEANUP
    return old


def idle(old):
    old.original_membership()
    need(not old.RUN.exists() and all(old.accounting(LABEL + '-' + phase) is None for phase in PHASES),
         'RNG bracket namespace or Slurm label already exists')


def read_outcome(root):
    outcome = json.loads(read(root / 'worker-outcome.json'))
    arms = [row['arm'] for row in outcome['arms']]
    attempted = [arm for arm in AB if (root / arm / 'native-intent.json').is_file()]
    need(outcome['request'] == REQUEST and outcome['planned_forwards'] == PLANNED
         and outcome['planned_optimizer_updates'] == 0 and list(AB)[:len(attempted)] == attempted
         and arms == attempted and all(row['cleanup_known'] is True and
         row['cleanup']['owned_processes_remaining'] == 0 for row in outcome['arms']),
         'strong-owner cleanup is missing for a registered arm')
    return outcome


def workflow(old, previous, doc, bundle, source_sha):
    root = old.RUN
    idle(old)
    epoch = time.time()
    anchor = time.monotonic()

    def elapsed():
        return max(time.time() - epoch, time.monotonic() - anchor)

    root.mkdir(mode=0o700)
    intent = dict(request=REQUEST, publication=doc['publication'], epoch=epoch, deadline=epoch+WHOLE,
                  native_deadline=epoch+NATIVE, GPU_client_deadline=epoch+GPU_CLIENT,
                  arm_deadlines={arm: epoch+offset for arm, offset in AB.items()}, parent=PARENT, node=NODE, port=PORT,
                  source_manifest_SHA256=source_sha, original_terminal_SHA256=TERMINAL_SHA, original_post_step='1103606.1191',
                  planned_forwards=PLANNED, planned_optimizer_updates=0, replay_allowed=False)
    old.write(root / 'diagnostic-intent.json', intent)
    result = dict(intent, phases={}, clean_post=False, complete=False, independent_post_pending=True,
                  formal_recovery_qualified=False)
    phases = result['phases']
    try:
        stage = root / 'target-worker'
        stage.mkdir(mode=0o700)
        (stage / 'original-observer').mkdir(mode=0o700)
        members = STAGED | {'original-observer/' + name for name in OLD_OBSERVER}
        need(set(bundle) == members, 'staged bracket closure differs')
        workers = {name: old.write(stage / name, raw) for name, raw in bundle.items()}
        snapshot = [old.HOST_PYTHON, '-B', str(HERE / 'host_once.py'), '--snapshot']
        phases['pre'] = old.execute_step('pre', [*snapshot, 'pre'], elapsed, epoch)
        if not phases['pre']['passed']:
            return result
        before = old.read_cpu_inventory('pre', phases['pre'])
        bound = old.write(root / 'before.json', before)
        config = old.write(root / 'worker-config.json', dict(intent, stage=doc['stage'], before=bound, workers=workers,
                           root_GO=True, GPU_GO=True))
        phases['gpu'] = old.execute_step('gpu', previous.gpu_command(old, config, epoch), elapsed, epoch)
        if not phases['gpu']['terminal_visible']:
            result['terminal_outcome_unknown'] = True
            return result
        phases['post'] = old.execute_step('post', [*snapshot, 'post'], elapsed, epoch)
        if not phases['post']['passed']:
            return result
        after = old.read_cpu_inventory('post', phases['post'])
        old.write(root / 'after.json', after)
        outcome = read_outcome(root)
        same_physical = all(before[k] == after[k] for k in ('boot_id', 'pci', 'pci_to_kfd', 'master_port'))
        result['clean_post'] = (same_physical and not after['owned_processes_remaining']
                                and after['owned_registry_complete'] is True)
        if phases['gpu']['passed'] and result['clean_post']:
            assigned = json.loads(read(root / 'worker-assigned.json'))
            summary = json.loads(read(root / 'rng-bracket-comparison.json'))
            need(outcome['epoch'] == assigned['epoch'] == epoch and assigned['request'] == REQUEST
                 and phases['gpu']['row'][0] == assigned['actual_step'] == summary['actual_step']
                 and assigned['boot_id'] == before['boot_id'] and sorted(assigned['visible_PCI_by_index']) == before['pci']
                 and outcome['passed'] is True and summary['complete'] is True
                 and summary['planned_forwards'] == summary['verified_individual_forwards'] == PLANNED
                 and summary['comparable_ranks'] == 8, 'incomplete owned bracket or physical identity differs')
            result['complete'] = True
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error)[:350])
    finally:
        old.write(root / 'diagnostic-outcome.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--launch-once', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--snapshot', choices=('pre', 'post'))
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    previous, bundle = verified(REPO)
    old = previous.controller(REPO)
    if args.snapshot:
        need(socket.gethostname().split('.')[0] == NODE and os.environ.get('SLURM_JOB_ID') == PARENT
             and os.environ.get('SLURM_STEP_ID', '').isdigit() and os.environ.get('SLURM_CPUS_PER_TASK') == '2',
             'wrong zero-GRES CPU observer')
        bind(previous, old).cpu_snapshot(args.snapshot)
        return 0
    if args.status:
        print(json.dumps(dict(request=REQUEST, controller_namespace_exists=Path(str(RUN)).exists(),
                             accounting={phase: old.accounting(LABEL + '-' + phase) for phase in PHASES})))
        return 0
    need(args.admission is not None and args.admission_sha256 and args.source_sha256, 'missing exact source and adoption pins')
    doc = json.loads(read(args.admission, args.admission_sha256))
    admission(doc)
    current = frozen(HERE, args.source_sha256, TARGET_FILES)
    bundle.update({name: current[name] for name in ('bracket_contract.py', 'worker.py')})
    archive = Path(str(ORIGINAL))
    doc['stage'] = old.accepted_stage(archive, STAGE_SHA)
    from contract import checkpoint
    checkpoint(archive, INVENTORY)
    plan = doc['stage']['plans']['resumed']
    native = json.loads(read(Path(plan['path']), plan['SHA256'], plan['bytes']))
    from contract import render as old_render
    previous.render = old_render
    for arm in AB:
        render(previous, native, arm)
    bind(previous, old)
    if args.preflight:
        idle(old)
        print(json.dumps(dict(request=REQUEST, planned_forwards=PLANNED, checkpoint_members=14,
                             installed_worker_files=len(bundle), actual_GPU_steps=0, source_only=True)))
        return 0
    result = workflow(old, previous, doc, bundle, args.source_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(not result['complete'])


if __name__ == '__main__':
    raise SystemExit(main())
