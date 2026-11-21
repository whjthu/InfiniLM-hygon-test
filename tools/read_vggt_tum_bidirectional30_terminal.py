"""Independently consume the original VGGT four-view training and recovery run."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import signal

from read_training_slurm_membership import allocated_dcu


LABEL = 'training-vggt-tum-bidirectional30-v1'
REQUEST = 'ab8a312813524d949608defa7124c3bf'
PARENT, NODE = '1103606', 'f13r4n09'
ARMS = {'continuous': 0, 'resume20': 20}
PHASES = {'pre': (0, '2', {'2G', '2048M'}, 45), 'gpu': (1, '4', {'64G', '65536M'}, 1440),
          'post': (0, '2', {'2G', '2048M'}, 45)}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED',
            'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
PCI = re.compile(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')
HEX64 = re.compile(r'[0-9a-f]{64}')
BUNDLE = 'b79a591250f5ba367438efc4122dcfe4a4ec66d59b821252e83b70404399c665'
BACKEND = 'f53220165a356fec024505b65af2ac7f12b1c9272f6c231df22634c060f31ea5'
WEIGHT = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
COMPONENTS = ('model_SHA256', 'optimizer_SHA256', 'placements_SHA256', 'rng_SHA256', 'full_boundary_SHA256')
DIRECTIONAL = dict(source_SHA256='b08d278311cad3ffc44fed8985e68d3fe969735fdadaf2c16dba03eedccf64ad',
    weight=0.1, minimum_target_baseline=0.005, predicted_norm_floor=0.001, pose_source_SHA256={
        'pose_enc.py': '845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b',
        'rotation.py': 'ad3a904e243c6654f0aa5f1d0206ace5f2d61011bc1bf8f492b40e0b76cb2070'})


class ReadDeadline(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def hex64(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def accounting(raw):
    phases = dict.fromkeys(PHASES)
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = line.split('|')
        need(len(row) == 7, 'original accounting columns differ')
        phase = row[1].removeprefix(LABEL + '-')
        if row[1] != LABEL + '-' + phase or phase not in phases:
            continue
        need(phases[phase] is None and re.fullmatch(PARENT + r'\.\d+', row[0]) and row[5] == NODE
             and row[2].split(), 'foreign or duplicate original phase row')
        phases[phase] = row
    handles = [row[0] for row in phases.values() if row]
    need(len(set(handles)) == len(handles), 'original phase handles overlap')
    return phases


def terminal(row):
    return bool(row and row[2].split()[0] in TERMINAL)


def completed(phase, row):
    if not terminal(row) or row[2:4] != ['COMPLETED', '0:0']:
        return False
    cards, cpus, mem, limit = PHASES[phase]
    values = dict(field.split('=', 1) for field in row[6].split(','))
    return (allocated_dcu(row[6]) == cards and values.get('cpu') == cpus and values.get('mem') in mem
            and values.get('node') == '1' and row[4].isdecimal() and int(row[4]) <= limit)


class OriginalFiles:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.hashes = {}

    def inspect(self, name, *, metadata=True):
        path = self.root / name
        need(path.is_relative_to(self.root) and path.resolve(strict=True) == path and path.is_file(),
             'unexpected original file path: ' + name)
        before = path.stat()
        digest = sha256()
        with path.open('rb') as stream:
            if metadata:
                raw = stream.read(8*1024**2 + 1)
                need(0 < len(raw) <= 8*1024**2, 'original metadata exceeds bound: ' + name)
                digest.update(raw)
            else:
                need(before.st_size > 0, 'empty original checkpoint')
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    digest.update(chunk)
                raw = None
        after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), 'original file changed: ' + name)
        self.hashes[name] = digest.hexdigest()
        return raw, digest.hexdigest()

    def read(self, name):
        raw, digest = self.inspect(name)
        return json.loads(raw), digest


def clean_post(files, rows):
    if not (terminal(rows['gpu']) and completed('pre', rows['pre']) and completed('post', rows['post'])):
        return False, None
    before, _ = files.read('before.json')
    after, _ = files.read('after.json')
    for phase, item in (('pre', before), ('post', after)):
        need(item['step'] == rows[phase][0] and item['node'] == NODE and item['mode'] == 'host_complete'
             and item['full_fd_inventory_performed'] is True and item['kfd_holder_count'] == 0,
             'original HOST cleanup evidence differs')
    pci = before['pci']
    need(len(set(pci)) == len(pci) == 8 and all(PCI.fullmatch(value) for value in pci)
         and after['pci'] == pci and before['boot_id'] == after['boot_id'] and before['boot_id']
         and 0 < before['monotonic_ns'] < after['monotonic_ns'], 'original HOST inventory identity changed')
    return True, before


def process_id(row):
    need(isinstance(row['boot_id'], str) and row['boot_id'] and type(row['pid']) is int and row['pid'] > 1
         and type(row['startticks']) is int and row['startticks'] > 0, 'malformed original worker lifetime')
    return row['boot_id'], row['pid'], row['startticks']


def arm_result(files, arm, first, intent):
    base = f'arms/{arm}/'
    receipt, receipt_sha = files.read(base + 'receipt.json')
    invocation, _ = files.read(base + 'invocation.json')
    process = process_id(receipt['process'])
    need(all(item['arm'] == arm and item['epoch'] == intent['epoch'] and item['deadline'] == intent['deadline'] - 100
             for item in (invocation, receipt)) and process == process_id(invocation['process']),
         'arm differs from the original invocation')
    need(receipt['bundle_SHA256'] == BUNDLE and receipt['backend_SHA256'] == BACKEND
         and receipt['executed_optimizer_steps'] == 25-first and receipt['physical_training_pairs'] == 2
         and receipt['physical_scene_count'] == 2 and receipt['ordered_training_views'] == 4
         and receipt['formal_CO3D_eligible'] is False and receipt['gradient_accumulation'] == 1
         and receipt['scheduler'] == dict(type='constant_learning_rate_in_optimizer_param_groups',
                                          mutable_scheduler=False, learning_rate=2e-5), 'native arm contract changed')
    native, native_sha = files.read(base + 'native-result.json')
    metrics = native['metrics']
    need(receipt['result_SHA256'] == native_sha and metrics['global_optimizer_step'] == 25
         and metrics['executed_optimizer_steps'] == 25-first and metrics['completed_global_step_before_resume'] == first
         and metrics['cached_valid_batches'] == metrics['sequence_count'] == 4
         and metrics['measurement_steps'] == 20-max(0, first-5)
         and metrics['formal_CO3D_eligible'] is False and metrics['resume_source_sha256'] == receipt['resume_source_SHA256']
         and metrics['checkpoint_identity_evidence']['frozen_model']['weights_model_pt'] == WEIGHT
         and metrics['relative_translation_loss'] == DIRECTIONAL
         and metrics['checkpoint_identity'] == receipt['checkpoint_identity']
         and hex64(receipt['final_state_SHA256']) and receipt['final_state_SHA256'] == metrics['final_state_sha256']
         and all(native['checks'][name] is True for name in ('camera_head_weights_changed', 'frozen_aggregator_unchanged',
            'main_forward_device', 'nonzero_finite_camera_head_gradients', 'resume_boundary')), 'native metrics differ')
    raw, _ = files.inspect(base + 'train/steps.ndjson')
    steps = [json.loads(line) for line in raw.splitlines()]
    need(steps == receipt['steps'] and [row['global_optimizer_step'] for row in steps] == list(range(first+1, 26))
         and all(row['batch_cursor'] == (row['global_optimizer_step']-1)%4 for row in steps)
         and [sum(row['batch_cursor'] == view for row in steps) for view in range(4)] == receipt['ordered_view_exposures']
         and all(number(row[key]) for row in steps for key in ('loss','grad_norm','camera_head_grad_norm_before_clip',
                                                               'relative_translation_direction_cost'))
         and all(0 <= row['relative_translation_direction_cost'] <= 2 for row in steps), 'native step evidence differs')
    checkpoints = receipt['checkpoints']
    need(set(checkpoints) == {str(n) for n in (10, 20, 25) if n > first}, 'checkpoint denominator differs')
    for key, item in checkpoints.items():
        n = int(key)
        need((item['global_step'],item['next_global_step'],item['next_batch_cursor']) == (n,n+1,n%4)
             and item['optimizer_group_learning_rates'] and all(v == 2e-5 for v in item['optimizer_group_learning_rates'])
             and all(hex64(item[field]) for field in COMPONENTS + ('file_SHA256',)), 'checkpoint metadata differs')
        _, digest = files.inspect(base + f'train/checkpoint-{n}.pt', metadata=False)
        need(digest == item['file_SHA256'], 'original checkpoint content differs')
    need(first != 0 or receipt['resume_source_SHA256'] is None, 'continuous run restored a checkpoint')
    return dict(receipt=receipt, digest=receipt_sha, process=process, steps=steps)


def recovery(continuous, resumed):
    left, right = continuous['receipt'], resumed['receipt']
    need(continuous['process'] != resumed['process'] and continuous['process'][0] == resumed['process'][0]
         and right['resume_source_SHA256'] == left['checkpoints']['20']['file_SHA256']
         and left['checkpoint_identity'] == right['checkpoint_identity'], 'fresh resume did not bind checkpoint-20')
    pairs = list(zip(continuous['steps'][20:], resumed['steps'], strict=True))
    loss = max(abs(a['loss'] - b['loss']) for a,b in pairs)
    cost = max(abs(a['relative_translation_direction_cost'] - b['relative_translation_direction_cost']) for a,b in pairs)
    gradients = max(abs(a['grad_norm'] - b['grad_norm']) for a,b in pairs)
    checkpoint = {key: left['checkpoints']['25'][key] == right['checkpoints']['25'][key] for key in COMPONENTS}
    final = left['final_state_SHA256'] == right['final_state_SHA256']
    need(max(row['relative_translation_direction_cost'] for row in continuous['steps']) > 0, 'inactive directional loss')
    return dict(passed=loss <= 1e-6 and cost <= 1e-6 and final and all(checkpoint.values()),
                max_absolute_loss=loss, max_absolute_direction_cost=cost, max_absolute_grad_norm=gradients,
                checkpoint25_component_identity=checkpoint, final_state_identity=final)


def verify(root, raw, manifest, claim):
    rows = accounting(raw)
    result = dict(request=REQUEST, accounting_SHA256=sha256(raw).hexdigest(), phase_rows=rows,
                  gpu_terminal=terminal(rows['gpu']), independent_post_clean=False,
                  planned_updates=30, independently_verified_updates=0, formal_CO3D_eligible=False,
                  formal_performance_eligible=False, passed=False, errors={},
                  original_collector_reported_passed=None, original_collector_verified=False)
    if not result['gpu_terminal']:
        return result
    files = OriginalFiles(root)
    intent, _ = files.read('intent.json')
    need(intent['request'] == REQUEST and intent['parent'] == PARENT and intent['node'] == NODE
         and hex64(manifest) and hex64(claim) and intent['source_manifest_SHA256'] == manifest
         and intent['claim_SHA256'] == claim and re.fullmatch('[0-9a-f]{40}', intent['publication'])
         and number(intent['epoch']) and intent['epoch'] > 0 and intent['deadline'] == intent['epoch']+1800,
         'adopted original intent differs')
    host = None
    try:
        result['independent_post_clean'], host = clean_post(files, rows)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['errors']['post'] = str(error)[:200]
    arms = {}
    for arm, first in ARMS.items():
        try:
            arms[arm] = arm_result(files, arm, first, intent)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors'][arm] = str(error)[:200]
    result['independently_verified_updates'] = sum(len(a['steps']) for a in arms.values())
    if set(arms) == set(ARMS):
        try:
            assignment, _ = files.read('assignment.json')
            kfd = assignment['kfd']
            need(assignment['step'] == rows['gpu'][0] and assignment['node'] == NODE
                 and PCI.fullmatch(assignment['pci']) and assignment['pci'] == kfd['pci']
                 and type(kfd['gpu_id']) is int and kfd['gpu_id'] > 0 and hex64(kfd['properties_SHA256'])
                 and type(assignment['pid']) is int and assignment['pid'] > 1
                 and type(assignment['startticks']) is int and assignment['startticks'] > 0
                 and (host is None or (assignment['pci'] in host['pci'] and
                      all(a['process'][0] == host['boot_id'] for a in arms.values()))), 'original assigned PCI/KFD differs')
            result['recovery'] = recovery(arms['continuous'], arms['resume20'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors']['recovery'] = str(error)[:200]
    try:
        outcome, _ = files.read('outcome.json')
        need(outcome['request'] == REQUEST and outcome['publication'] == intent['publication'] and
             outcome['formal_acceptance'] is False and outcome['effect_qualified'] is False, 'original collector identity differs')
        need(type(outcome['passed']) is bool, 'original collector terminal flag differs')
        result['original_collector_reported_passed'] = outcome['passed']
        if outcome['passed']:
            original_native, native_sha = files.read('native-result.json')
            need(outcome['result_SHA256'] == native_sha and set(outcome['stages']) == set(PHASES)
                 and all(outcome['stages'][phase]['passed'] is True and outcome['stages'][phase]['row'] == rows[phase]
                         for phase in PHASES)
                 and set(arms) == set(ARMS) and all(
                     original_native['receipts'][arm] == dict(SHA256=arms[arm]['digest'], value=arms[arm]['receipt'])
                     for arm in ARMS), 'original collector does not bind independent rows and arm receipts')
            result['original_collector_verified'] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['errors']['outcome'] = str(error)[:200]
    result['passed'] = bool(result.get('recovery', {}).get('passed') and result['independent_post_clean']
                            and completed('gpu', rows['gpu']) and result['original_collector_verified'])
    result['original_file_SHA256'] = files.hashes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_root', type=Path)
    parser.add_argument('raw_accounting', type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--claim-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    need(args.original_root.name == LABEL and not args.output.resolve().is_relative_to(args.original_root.resolve()),
         'keep independent output outside the original run')
    def deadline(*_):
        raise ReadDeadline('independent read deadline elapsed')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    with args.raw_accounting.open('rb') as stream:
        raw = stream.read(2*1024**2 + 1)
    need(len(raw) <= 2*1024**2, 'original accounting exceeds bound')
    result = verify(args.original_root, raw, args.manifest_sha256, args.claim_sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('request','gpu_terminal','independent_post_clean',
                                                  'independently_verified_updates','passed')}), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
