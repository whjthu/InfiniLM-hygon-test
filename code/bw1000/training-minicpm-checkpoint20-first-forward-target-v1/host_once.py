"""Own one new MiniCPM diagnostic pre / eight-DCU / post sequence."""

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from contract import (CLEANUP, CONTROL_MANIFEST, CONTROL_SHA, GPU_CLIENT, LABEL, NATIVE, NODE,
                      ORIGINAL, ORIGINAL_LABEL, PARENT, PHASES, PORT, PUBLIC, PUBLIC_SHA, REQUEST,
                      RUN, SOURCE, SOURCE_SHA, STAGE_SHA, WHOLE, WORKER_FILES, admission, checkpoint,
                      digest, import_file, need, read_pin)


REPO = HERE.parents[2]


def verified_sources(repo):
    original = json.loads(read_pin(repo / CONTROL_MANIFEST, CONTROL_SHA))
    archived = {relative: read_pin(repo / relative, sha) for relative, sha in original.items()}
    source = repo / 'code/bw1000' / SOURCE
    manifest = json.loads(read_pin(source / 'SOURCE_FILES.json', SOURCE_SHA))
    members = {name: read_pin(source / name, pin['SHA256'], pin['bytes'])
               for name, pin in manifest['files'].items()}
    folder = 'code/bw1000/' + ORIGINAL_LABEL + '/'
    files = {name: members[name] for name in ('native_observer.py', 'probe.py', 'compare_first.py')}
    for name in ('snapshot.py', 'owned_processes.py'):
        files[name] = archived[folder + name]
    files['original_worker.py'] = archived[folder + 'worker.py']
    files['original1171.json'] = read_pin(repo / PUBLIC, PUBLIC_SHA, 3271)
    return files


def verify_target(manifest_sha):
    manifest = json.loads(read_pin(HERE / 'SOURCE_FILES.json', manifest_sha))
    need(set(manifest['files']) == {'CLAIM.json', 'README.zh-CN.md', 'check_local.py', 'contract.py',
                                  'host_once.py', 'worker.py'}, 'new source closure differs')
    return {name: read_pin(HERE / name, pin['SHA256'], pin['bytes'])
            for name, pin in manifest['files'].items()}


def controller(repo):
    sys.path.insert(0, str(repo / 'tools'))
    try:
        old = import_file('archived_minicpm_host', repo / 'tools/control_minicpm_usgs_pair_target_v3.py')
    finally:
        sys.path.pop(0)
    need(old.REPO == repo and (old.PARENT, old.NODE, old.RUN.as_posix()) == (PARENT, NODE, str(ORIGINAL)),
         'wrong archived HOST controller')
    return old


def bind_new(old):
    old.RUN = Path(str(RUN))
    old.LABEL, old.REQUEST, old.PORT = LABEL, REQUEST, PORT
    old.PHASES, old.WHOLE, old.NATIVE, old.WORKER_CLEANUP = PHASES, WHOLE, NATIVE, CLEANUP
    return old


def gpu_command(old, config, epoch):
    argv = old.gpu_command(config, epoch)
    image = str(old.RESULTS.parent / 'containers/dtk26-sandbox-r1')
    position = argv.index(image)
    argv[position:position] = ['--bind', f'{ORIGINAL}:{ORIGINAL}:ro']
    command = argv[-1]
    current = shlex.quote(str(RUN / 'framework') + ':' + str(RUN / 'src'))
    archive = shlex.quote(str(ORIGINAL / 'framework') + ':' + str(ORIGINAL / 'src'))
    need(command.count(current) == 1, 'archived GPU bootstrap changed')
    argv[-1] = command.replace(current, archive, 1)
    return argv


def workflow(old, doc, available, source_sha):
    root = old.RUN
    old.original_membership()
    need(not root.exists() and all(old.accounting(LABEL + '-' + role) is None for role in PHASES),
         'new label or namespace already present')
    epoch = time.time()
    started = time.monotonic()

    def elapsed():
        return max(time.time() - epoch, time.monotonic() - started)

    root.mkdir(mode=0o700)
    phases = {}
    initial = dict(request=REQUEST, publication=doc['publication'], epoch=epoch, deadline=epoch + WHOLE,
                   native_deadline=epoch + NATIVE, GPU_client_deadline=epoch + GPU_CLIENT,
                   source_manifest_SHA256=source_sha, parent=PARENT, node=NODE, replay_allowed=False,
                   planned_ranks=8, planned_forwards_per_rank=1, planned_optimizer_updates=0)
    # Intent exists before any Slurm command. If bootstrap is interrupted, the
    # same request is recovered by its exact labels and existing namespace.
    old.write(root / 'diagnostic-intent.json', initial)
    result = dict(initial, phases=phases, complete=False, clean_post=False, physical_post_audited_independently=False,
                  formal_recovery_qualified=False, error=None)
    try:
        folder = root / 'target-worker'
        folder.mkdir(mode=0o700)
        workers = {}
        for name, raw in available.items():
            pin = old.write(folder / name, raw)
            if name in WORKER_FILES:
                workers[name] = pin
        snap = [old.HOST_PYTHON, '-B', str(HERE / 'host_once.py'), '--snapshot']
        phases['pre'] = old.execute_step('pre', [*snap, 'pre'], elapsed, epoch)
        if not phases['pre']['passed']:
            return result
        before = old.read_cpu_inventory('pre', phases['pre'])
        bound = old.write(root / 'before.json', before)
        config = old.write(root / 'worker-config.json',
                           dict(initial, before=bound, workers=workers, stage=doc['stage'],
                                checkpoint20_inventory=doc['checkpoint20_inventory'], root_GO=True, GPU_GO=True,
                                declared_optimizer_updates=0))
        phases['gpu'] = old.execute_step('gpu', gpu_command(old, config, epoch), elapsed, epoch)
        if not phases['gpu']['terminal_visible']:
            return result
        phases['post'] = old.execute_step('post', [*snap, 'post'], elapsed, epoch)
        if not phases['post']['passed']:
            return result
        after = old.read_cpu_inventory('post', phases['post'])
        old.write(root / 'after.json', after)
        result['clean_post'] = (all(after[key] == before[key] for key in ('boot_id', 'pci', 'pci_to_kfd', 'master_port'))
                                and not after['owned_processes_remaining'] and after['owned_registry_complete'] is True)
        if phases['gpu']['passed'] and result['clean_post']:
            assigned = json.loads(read_pin(root / 'worker-assigned.json'))
            outcome = json.loads(read_pin(root / 'worker-outcome.json'))
            arm, = outcome['arms']
            check = arm['comparison']
            need(assigned['request'] == outcome['request'] == REQUEST and assigned['epoch'] == outcome['epoch'] == epoch
                 and assigned['actual_step'] == phases['gpu']['row'][0]
                 and assigned['boot_id'] == before['boot_id'] and sorted(assigned['visible_PCI_by_index']) == before['pci']
                 and outcome['passed'] is arm['passed'] is True
                 and check['ranks_planned'] == check['ranks_compared'] == len(check['rows']) == 8
                 and check['micro_forwards_per_rank'] == 1 and check['backward_calls'] == check['optimizer_updates'] == 0,
                 'terminal zero-update diagnostic or same-device binding differs')
            result['complete'] = True
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error)[:350])
    finally:
        old.write(root / 'diagnostic-outcome.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--launch-once', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--snapshot', choices=('pre', 'post'))
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    available = verified_sources(REPO)
    old = controller(REPO)
    if args.snapshot or args.status:
        bind_new(old)
        if args.snapshot:
            need(socket.gethostname().split('.')[0] == NODE and os.environ.get('SLURM_JOB_ID') == PARENT
                 and os.environ.get('SLURM_STEP_ID', '').isdigit()
                 and os.environ.get('SLURM_CPUS_PER_TASK') == '2', 'not the scoped zero-GRES CPU inventory step')
            old.cpu_snapshot(args.snapshot)
            return 0
        print(json.dumps(dict(request=REQUEST, controller_namespace_exists=old.RUN.exists(),
                             accounting={role: old.accounting(LABEL + '-' + role) for role in PHASES}), sort_keys=True))
        return 0
    need(args.admission is not None and args.admission_sha256 and args.source_sha256, 'exact adoption/source pins required')
    doc = json.loads(read_pin(args.admission, args.admission_sha256))
    pin = admission(doc)
    current = verify_target(args.source_sha256)
    available.update({name: current[name] for name in ('contract.py', 'worker.py')})
    original = Path(str(ORIGINAL))
    doc['stage'] = old.accepted_stage(original, STAGE_SHA)
    checkpoint(original, pin)
    render_pin = doc['stage']['plans']['resumed']
    from contract import render
    render(json.loads(read_pin(Path(render_pin['path']), render_pin['SHA256'], render_pin['bytes'])))
    bind_new(old)
    if args.preflight:
        old.original_membership()
        need(not old.RUN.exists() and all(old.accounting(LABEL + '-' + role) is None for role in PHASES),
             'new label or namespace already present')
        print(json.dumps(dict(request=REQUEST, verified_source_files=len(available), checkpoint_members=14,
                             checkpoint_tensors_read=0, new_steps=0, launch_attempted=False), sort_keys=True))
        return 0
    result = workflow(old, doc, available, args.source_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(not result['complete'])


if __name__ == '__main__':
    raise SystemExit(main())
