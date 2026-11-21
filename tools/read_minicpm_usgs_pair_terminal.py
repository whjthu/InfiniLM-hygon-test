"""Rebuild one MiniCPM domain pair from its original terminal evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import signal
import stat
import subprocess

from control_minicpm_usgs_pair_target import LABEL, NODE, PARENT, PHASES, PORT, REQUEST, RUN
from prepare_minicpm_usgs_pair_target import CORE, CORE_ASSETS_SHA, CORE_SHA, COLUMNS, stable_file, write
from read_training_slurm_membership import allocated_dcu


CHECKPOINTS = {'continuous': (10, 20, 25), 'resumed': (25,)}
FILES = {'adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json', 'args.json',
         'optimizer.pt', 'scheduler.pt', *(f'rng_state_{rank}.pth' for rank in range(8))}
PCI = re.compile(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')
STEP = re.compile(PARENT + r'\.\d+')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED',
            'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def token(value, size):
    return isinstance(value, str) and len(value) == size and re.fullmatch('[0-9a-f]+', value) is not None


def accounting_phases(raw):
    labels = {LABEL + '-' + phase: phase for phase in PHASES}
    rows = {phase: None for phase in PHASES}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = line.split('|')
        need(len(row) == 7, 'unexpected original accounting column layout')
        phase = labels.get(row[1])
        if phase is None:
            continue
        need(rows[phase] is None and STEP.fullmatch(row[0]) and row[5] == NODE,
             'duplicate or foreign original accounting row')
        rows[phase] = row
    found = [row[0] for row in rows.values() if row]
    need(len(set(found)) == len(found), 'same step selected for different phases')
    return rows


def proved_resources(phase, row):
    values = row[6].split(',')
    resources = dict(v.split('=', 1) for v in values)
    cards, cpus, memory, _ = PHASES[phase]
    actual_memory = resources.get('mem', '')
    size = re.fullmatch(r'(\d+)([MG])', actual_memory)
    memory_mib = int(size[1]) * (1024 if size[2] == 'G' else 1) if size else None
    return (len(values) == len(resources) and allocated_dcu(row[6]) == cards and resources.get('cpu') == str(cpus)
            and resources.get('node') == '1' and memory_mib == memory and row[4].isdecimal())


class OriginalFiles:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.meta = {}
        self.total_bytes = 0

    def raw(self, name, expected=None):
        path = self.root / name
        need(path.is_relative_to(self.root) and '..' not in path.parts, 'unexpected original receipt path')
        data = stable_file(path, digest=expected, ceiling=8 * 1024**2, collect=True)
        self.total_bytes += len(data)
        need(self.total_bytes <= 128 * 1024**2, 'original metadata read limit exceeded')
        digest = sha256(data).hexdigest()
        need(self.meta.setdefault(name, digest) == digest, 'original metadata changed')
        return data

    def read(self, name, expected=None):
        return json.loads(self.raw(name, expected))


def bind_snapshots(files, rows, epoch):
    before = files.read('before.json')
    after = files.read('after.json')
    worker = files.read('worker-assigned.json')
    expected = before['pci']
    need(len(expected) == len(set(expected)) == 8 and all(PCI.fullmatch(value) for value in expected),
         'expected eight distinct original PCI devices')
    for phase, receipt in [('pre', before), ('post', after)]:
        need(receipt['node'] == NODE and receipt['step'] == rows[phase][0] and receipt['pci'] == expected
             and receipt['full_fd_inventory_performed'] is True and receipt['kfd_holder_count'] == 0
             and receipt['boot_id'] == before['boot_id'] and receipt['master_port'] == PORT,
             'original HOST PCI/process/port snapshot differs')
    need(before['monotonic_ns'] < after['monotonic_ns'] and after['owned_registry_complete'] is True
         and after['owned_processes_remaining'] == [], 'independent post does not show original process cleanup')
    leased = worker['leased_snapshot']
    cards = leased['pci_to_kfd']
    need(worker['request'] == REQUEST and worker['epoch'] == epoch and worker['actual_step'] == rows['gpu'][0]
         and worker['boot_id'] == leased['boot_id'] == before['boot_id'] and leased['node'] == NODE
         and leased['step'] == rows['gpu'][0] and leased['pci'] == sorted(worker['visible_PCI_by_index']) == expected
         and leased['holder_inventory_basis'] == 'kernel_KFD_proc_only' and leased['kfd_holder_count'] == 0
         and after['monotonic_ns'] > leased['monotonic_ns']
         and 0 <= leased['monotonic_ns'] - before['monotonic_ns'] <= 45_000_000_000
         and set(cards) == set(expected) and len(set(cards.values())) == 8
         and all(type(value) is int and value > 0 for value in cards.values())
         and len(set(worker['actual_CPU_affinity'])) >= 64, 'original isolated GPU lease differs')


def checkpoints(files, arm):
    inventories = {}
    for step in CHECKPOINTS[arm]:
        value = files.read(f'{arm}-checkpoint-{step}-files.json')
        need(value['arm'] == arm and value['global_step'] == step and value['member_count'] == 14
             and set(value['files']) == FILES and value['actual_tensor_deserialization_here'] is False,
             'checkpoint manifest does not cover the original saved state')
        prefix = f'{arm}/train/checkpoint-{step}'
        pins = value['files']
        for name, pin in pins.items():
            path = files.root / prefix / name
            original = path.lstat()
            need(path.resolve(strict=True) == path.absolute() and stat.S_ISREG(original.st_mode)
                 and type(pin['bytes']) is int and original.st_size == pin['bytes'] > 0 and token(pin['SHA256'], 64),
                 'checkpoint member is missing or differs')
        state, args = [files.read(f'{prefix}/{name}', pins[name]['SHA256']) for name in ('trainer_state.json', 'args.json')]
        need(state['global_step'] == step and state['max_steps'] == 25 and state['epoch'] == step / 25
             and args['max_steps'] == 25 and args['gradient_accumulation_steps'] == 4
             and args['per_device_train_batch_size'] == 1, 'saved trainer context differs')
        inventories[step] = pins
    return inventories


def rank_runs(files, worker, epoch, arm, initial):
    receipt = files.read(f'{arm}/native-outcome.json')
    need(receipt == worker and receipt['request'] == REQUEST and receipt['arm'] == arm and receipt['passed'] is True
         and receipt['cleanup_known'] is True and receipt['cleanup'] == dict(owned_processes_remaining=0, termination_needed=False)
         and receipt['rc'] == 0 and len(receipt['ranks']) == 8, 'original GPU worker arm differs')
    pins = set()
    for rank, row in enumerate(receipt['ranks']):
        directory = f'{arm}/train/minicpm-usgs-witness/rank-{rank:02d}'
        ref = row['result_file']
        name = f'{directory}/result.json'
        need(row['rank'] == rank and row['complete'] is True and Path(ref['path']) == files.root / name,
             'rank evidence is outside its original path')
        raw = files.raw(name, ref['SHA256'])
        result = json.loads(raw)
        need(len(raw) == ref['bytes'] and result['rank'] == rank and result['world_size'] == 8,
             'rank or original receipt bytes differ')
        if arm == 'continuous':
            initialized = files.read(f'{directory}/initialization.json')
            need(initialized['named_parameters_SHA256'] == initial['named_SHA256']
                 and initialized['file_SHA256'] == initial['SHA256'] and initialized['file_bytes'] == initial['bytes']
                 and initialized['parameter_numel'] == 10822656 and initialized['global_step'] == 0
                 and initialized['optimizer_state_entries'] == 0, 'original trainable initialization differs')
            expected = list(range(1, 26))
            steps = result['steps']
            need([value['global_step'] for value in steps] == expected and
                 all(value['optimizer_step_min'] == value['optimizer_step_max'] == k
                     and value['optimizer_slot_count'] == initialized['parameter_tensors']
                     and initialized['parameter_tensors'] > 0 for k, value in zip(expected, steps))
                 and any(value['adapter_changed'] is True for value in steps), 'native optimizer updates were not proved')
            frozen = result['frozen']
        else:
            frozen = files.read(f'{directory}/frozen-boundaries.json')
        need(set(frozen) == {'model.vision_tower', 'model.merger'}
             and all(value['tensors'] > 0 and value['parameters'] > 0 for value in frozen.values()),
             'original frozen vision and aligner boundary differs')
        pins.add(name)
    intent = files.read(f'{arm}/native-intent.json')
    need(intent['request'] == REQUEST and intent['arm'] == arm and intent['epoch'] == epoch
         and intent['deadline'] == epoch+1980, 'native arm reset or changed its deadline')
    return pins


def original_recovery_inputs(root):
    manifests = {'continuous': ('result.json', 'checkpoint-20-rng.json', 'checkpoint-25-rng.json'),
                 'resumed': ('result.json', 'native-data-skip.json', 'restored-before-RNG.json', 'checkpoint-25-rng.json')}
    paths = set()
    total = 0
    for arm, names in manifests.items():
        for rank in range(8):
            for name in names:
                relative = f'{arm}/train/minicpm-usgs-witness/rank-{rank:02d}/{name}'
                path = root / relative
                info = path.lstat()
                need(path.resolve(strict=True) == path.absolute() and stat.S_ISREG(info.st_mode)
                     and 0 < info.st_size <= 8 * 1024**2, 'invalid original recovery metadata path')
                paths.add(relative)
                total += info.st_size
    need(total <= 128 * 1024**2, 'original recovery metadata read limit exceeded')
    return paths


def audit(root, raw_accounting, publication, intent_sha):
    need(token(publication, 40) and token(intent_sha, 64), 'exact published request and intent pins required')
    rows = accounting_phases(raw_accounting)
    report = dict(request=REQUEST, publication=publication, intent_SHA256=intent_sha,
                  accounting_SHA256=sha256(raw_accounting).hexdigest(), accounting=rows,
                  expected_rank_count=8, planned_updates=30, planned_rank_microbatches=960,
                  planned_checkpoints=4, optimizer_updates_verified=None, independent_recovery=False,
                  checkpoint_tensor_payload_read_bytes=0, formal17_qualified=False)
    terminal = {phase: row for phase, row in rows.items() if row and row[2].split()[0] in TERMINAL}
    if any(row[2:4] != ['COMPLETED', '0:0'] or not proved_resources(phase, row) for phase, row in terminal.items()):
        return dict(report, state='ORIGINAL_SLURM_ACCOUNTING_FAILED', passed=False)
    if terminal.keys() != rows.keys():
        return dict(report, state='ORIGINAL_ACCOUNTING_INCOMPLETE', passed=False)
    files = OriginalFiles(root)
    intent = files.read('training-intent.json', intent_sha)
    epoch = intent['epoch']
    need(intent['request'] == REQUEST and intent['publication'] == publication and intent['replay_allowed'] is False
         and intent['parent'] == PARENT and intent['node'] == NODE and intent['deadline'] == epoch+2400
         and intent['native_deadline'] == epoch+1980, 'original training intent differs')
    host = files.read('training-outcome.json')
    need(host['request'] == REQUEST and host['publication'] == publication and host['epoch'] == epoch
         and host['completed'] is True and host['clean_post'] is True, 'HOST original completion or cleanup differs')
    for phase, row in rows.items():
        stored = files.read(f'{phase}-accounting.json')
        need(stored['epoch'] == epoch and stored['row'] == row and stored['terminal_visible'] is True
             and stored['passed'] is True and host['phases'][phase]['row'] == row, 'HOST original and current Slurm rows differ')
    bind_snapshots(files, rows, epoch)
    worker = files.read('worker-outcome.json')
    need(worker['request'] == REQUEST and worker['epoch'] == epoch and worker['passed'] is True
         and worker['native_deadline'] == epoch+1980 and worker['arms_not_attempted'] == []
         and len(worker['arms']) == 2, 'original paired worker incomplete')
    source = json.loads(stable_file(CORE / 'SOURCE_FILES.json', digest=CORE_SHA, ceiling=2*1024**2, collect=True))
    comparator = CORE / 'compare_recovery.py'
    comparator_pin = source['files'][comparator.name]['SHA256']
    comparator_raw = stable_file(comparator, digest=comparator_pin, ceiling=2*1024**2, collect=True)
    context = {'__name__': 'minicpm_original_metadata_audit'}
    exec(compile(comparator_raw, str(comparator), 'exec'), context)
    recovery_paths = original_recovery_inputs(files.root)
    comparison = context['compare'](files.root)
    need(set(comparison['evidence_SHA256']) == recovery_paths, 'recovery did not consume the original file denominator')
    initial = files.read('src/ASSETS.json', CORE_ASSETS_SHA)['initial']
    for arm, outcome in zip(CHECKPOINTS, worker['arms']):
        manifest = checkpoints(files, arm)
        names = rank_runs(files, outcome, epoch, arm, initial)
        if arm == 'continuous':
            continuous20 = manifest[20]
        else:
            witness = files.read('resumed-witness-config.json')
            need(witness['checkpoint_files'] == continuous20 and Path(witness['checkpoint']) ==
                 files.root / 'continuous/train/checkpoint-20', 'resume witness missed its original checkpoint')
        need(all(comparison['evidence_SHA256'][name] == files.meta[name] for name in names), 'raw rank evidence changed')
    need(comparison == files.read('paired-recovery.json') and comparison['exact_paired_recovery'] is True,
         'independent recovery comparison differs')
    return dict(report, state='ORIGINAL_MINICPM_DOMAIN_PAIR_METADATA_PASS', passed=True,
                optimizer_updates_verified=30, independent_recovery=True, paired_rank_microbatches=160,
                original_checkpoints_verified=4, recovery_comparator_SHA256=comparator_pin, evidence_SHA256=files.meta,
                comparison=comparison, same_input_A100_numerics=False, heldout_effect=False, formal_performance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounting', type=Path, help='independent original seven-column save; otherwise query sacct once')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publication', required=True)
    parser.add_argument('--intent-sha', required=True)
    args = parser.parse_args()
    root = RUN.resolve(strict=True)
    output = args.output.absolute()
    need(output.parent.resolve(strict=True) == root
         and re.fullmatch(r'independent-terminal-metadata-[0-9a-f]{16}\.json', output.name),
         'expected a new private original metadata report path')
    need(not output.exists(), 'original independent report already exists')
    def timeout(*_):
        raise TimeoutError('independent terminal metadata budget expired')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    try:
        if args.accounting:
            raw = stable_file(args.accounting.absolute(), collect=True, ceiling=2*1024**2)
        else:
            raw = subprocess.check_output(['sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-19T00:00:00',
                                           '-o', COLUMNS], timeout=12)
            need(len(raw) <= 2 * 1024**2, 'independent original accounting exceeds its byte bound')
            # Keep the exact input next to its independent report, including incomplete observations.
            write(output.with_suffix('.psv'), raw)
        try:
            result = audit(root, raw, args.publication, args.intent_sha)
        except (OSError, KeyError, ValueError, TypeError) as error:
            result = dict(state='ORIGINAL_MINICPM_METADATA_REJECTED', passed=False, request=REQUEST,
                          publication=args.publication, intent_SHA256=args.intent_sha, planned_updates=30,
                          planned_rank_microbatches=960, optimizer_updates_verified=None,
                          accounting_SHA256=sha256(raw).hexdigest(), error_type=type(error).__name__,
                          error=str(error)[:256], formal17_qualified=False)
        result_pin = write(output, result)
    finally:
        signal.alarm(0)
    print(json.dumps(dict(state=result['state'], passed=result['passed'], report_SHA256=result_pin['SHA256'],
                          optimizer_updates_verified=result['optimizer_updates_verified']), flush=True))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
