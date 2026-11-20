"""Run the fixed VGGT development comparison once on its original scoring node."""

import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from read_training_slurm_membership import allocated_dcu, snapshot as membership
from stage_vggt_tum_cross_camera_dev import BUNDLE_SHA, COHORT, INPUTS_SHA, STAGE_NAME, SOURCES, write_json


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
RESULTS = ROOT / 'results'
PARENT, NODE = '1087656', 'f11r4n19'
LABEL = 'training-vggt-tum-cross-camera-dev24-v1'
HOST_PY = '/usr/local/python3.12/bin/python3'
APPTAINER = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
SHARED = 'code/bw1000/training-qwen-native25-hip-v3'
KFD = 'code/bw1000/GLM-C1-layer-observer-r474/TP16-FFN-f32-transport-pair-v1'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
STAGES = {'pre': (0, 2, 2048, 60), 'gpu': (1, 4, 65536, 1200), 'post': (0, 2, 2048, 60)}
SOURCE_FILES = (
    'tools/control_vggt_tum_cross_camera_dev.py', 'tools/stage_vggt_tum_cross_camera_dev.py',
    'tools/read_training_slurm_membership.py', 'tools/bundle_vggt_tum_cross_camera_dev.py',
    'tools/bind_vggt_tum_cross_camera_heads.py', 'tools/bind_vggt_tum_dev_rgb.py',
    'tools/join_vggt_tum_cross_camera_dev.py', 'tools/plan_vggt_tum_public_dev_trajectories.py',
    'tools/score_vggt_tum_cross_camera_dev.py', 'tools/audit_vggt_tum_cross_camera_dev.py',
    'coordination/training/training80-active-node-task-map-v1.json',
    'code/bw1000/training-domain-pilot-v1/score_vggt_tum_pair.py',
    SHARED + '/snapshot.py', SHARED + '/slurm_accounting.py', KFD + '/assigned_kfd.py',
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(path, wanted, limit=2 * 1024**2):
    require(path.is_file() and path.absolute() == path.resolve(), 'missing or aliased input')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit and sha256(raw).hexdigest() == wanted, 'input digest differs')
    return raw


def import_helpers():
    sys.path[:0] = [str(REPO / SHARED), str(REPO / KFD)]
    import snapshot
    snapshot.NODE = NODE
    snapshot.STEP = re.compile(PARENT + r'\.\d+')
    return snapshot


def slurm_row(label):
    text = subprocess.check_output(['sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-19T00:00:00',
                                    '-o', COLUMNS], text=True, timeout=8)
    parsed = [line.split('|') for line in text.splitlines() if line.strip()]
    require(all(len(row) == 7 for row in parsed), 'unexpected accounting column layout')
    rows = [row for row in parsed if row[1] == label]
    require(len(rows) <= 1, 'duplicate exact step label')
    return rows[0] if rows else None


def step_command(phase, args):
    devices, cpus, memory, seconds = STAGES[phase]
    return ['srun', '--jobid=' + PARENT, '--exact', '--immediate=5', '--cpu-bind=none', '-N1', '-n1',
            f'-c{cpus}', f'--mem={memory}M', '--gres=' + ('dcu:1' if devices else 'none'), '-w', NODE,
            '-t', str(seconds // 60), '--job-name=' + LABEL + '-' + phase, *args]


def run_step(root, phase, args, deadline):
    from slurm_accounting import wait_terminal

    label = LABEL + '-' + phase
    require(slurm_row(label) is None, 'the original phase already exists')
    maximum = STAGES[phase][3]
    seconds = min(maximum + 5, deadline - time.monotonic() - (110 if phase != 'post' else 26))
    require(seconds >= maximum, 'remaining controller budget cannot cover this phase')
    with (root / (phase + '.log')).open('xb') as log:
        try:
            rc = subprocess.run(step_command(phase, args), stdout=log, stderr=subprocess.STDOUT,
                                timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            rc = None
    try:
        result = wait_terminal(lambda: slurm_row(label), min(deadline - 9, time.monotonic() + 24), max_observations=7)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        write_json(root / (phase + '-accounting.json'), dict(rc=rc, terminal_visible=False, passed=False,
                   row=None, observation_error=type(error).__name__))
        raise
    row = result['row']
    ok = bool(rc == 0 and row and re.fullmatch(PARENT + r'\.\d+', row[0]) and row[2:4] == ['COMPLETED', '0:0']
              and row[5] == NODE and allocated_dcu(row[6]) == STAGES[phase][0])
    result.update(rc=rc, passed=ok)
    write_json(root / (phase + '-accounting.json'), result)
    return result


def read_object(path):
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2, 'unexpected receipt size')
    return json.loads(raw)


def clean_snapshot(root):
    snapshot = import_helpers()
    snap = snapshot.snapshot(False)
    require(snap['kfd_holder_count'] == 0 and snap['full_fd_inventory_performed'], 'node holds an accelerator')
    owner = root / 'assignment.json'
    if owner.exists():
        info = read_object(owner)
        try:
            fields = Path(f'/proc/{info["pid"]}/stat').read_text().rsplit(') ', 1)[1].split()
        except FileNotFoundError:
            pass
        else:
            require(fields[0] == 'Z' or int(fields[19]) != info['startticks'], 'original GPU process still active')
    print(json.dumps(snap), flush=True)


def native(root):
    snapshot = import_helpers()
    from assigned_kfd import assigned_kfd
    before = read_object(root / 'before.json')
    intent = read_object(root / 'score-intent.json')
    handle = os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', '')
    require(snapshot.STEP.fullmatch(handle) and socket.gethostname().split('.')[0] == NODE
            and len(os.sched_getaffinity(0)) >= 4, 'wrong scorer allocation')
    require(snapshot.drm_inventory() == before['pci'] and not snapshot.kernel_holders()
            and before['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and 0 <= time.monotonic_ns() - before['monotonic_ns'] <= snapshot.MAX_AGE_NS,
            'device admission no longer matches preflight')
    hip = ctypes.CDLL('libamdhip64.so')
    count, identifier = ctypes.c_int(), ctypes.create_string_buffer(32)
    require(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
            and hip.hipDeviceGetPCIBusId(identifier, len(identifier), 0) == 0, 'expected one assigned HIP device')
    pci = identifier.value.decode().lower()
    pci = '0000:' + pci if len(pci) == 7 else pci
    require(pci in before['pci'], 'assigned PCI differs from preflight')
    write_json(root / 'assignment.json', dict(step=handle, node=NODE, pci=pci, kfd=assigned_kfd(pci),
               pid=os.getpid(), startticks=int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])))
    sys.argv = ['score_vggt_tum_cross_camera_dev.py', '--inputs', str(root / 'inputs/inputs-private.json'),
                '--inputs-sha', INPUTS_SHA, '--head-authority', str(root / 'head-authority.json'),
                '--head-authority-sha', intent['head_authority_SHA256'], '--alternate', '/alternate',
                '--lowlr', '/lowlr', '--upstream', '/vggt-source', '--weights', '/weights/model.pt',
                '--output', str(root / 'score')]
    from score_vggt_tum_cross_camera_dev import main
    main()


def gpu_command(root):
    args = [APPTAINER, 'exec', '--no-home', '--pwd', '/']
    binds = ((REPO, '/source:ro'), (root, '/pilot:rw'),
             (RESULTS / SOURCES['alternate25'][0], '/alternate:ro'), (RESULTS / SOURCES['lowlr25'][0], '/lowlr:ro'),
             (RESULTS / 'training-vggt-native-provider-HIP-CPU-v1/upstream', '/vggt-source:ro'),
             (RESULTS / 'training-vggt-original-public-model-v1/model', '/weights:ro'),
             (RESULTS / 'training-qwen-a68952c-swift-initial-CPU-v1/framework', '/framework:ro'),
             (RESULTS / 'training-t06-sdxl-runtime-v1', '/training-runtime:ro'),
             (Path('/opt/hyhal'), '/opt/hyhal:ro'))
    for source, target in binds:
        args.extend(['--bind', f'{source}:{target}'])
    shell = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
             'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '
             'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site:/framework/generation/src:/framework; '
             'exec /usr/bin/timeout --kill-after=5s 1160s python3 -B -u '
             '/source/tools/control_vggt_tum_cross_camera_dev.py /pilot --worker native')
    return [*args, str(ROOT / 'containers/dtk26-sandbox-r1'), '/bin/bash', '--noprofile', '--norc', '-c', shell]


def controller(root, stage_sha, request, publication, manifest_sha):
    require(root == RESULTS / STAGE_NAME and root.resolve(strict=True) == root, 'expected the original staging directory')
    require(re.fullmatch(r'[a-f0-9]{32}', request or '') and re.fullmatch(r'[a-f0-9]{40}', publication or ''),
            'request identity differs')
    manifest = json.loads(pin(REPO / 'coordination/training/vggt-tum-cross-camera-controller-files-v1.json', manifest_sha))
    require(set(manifest) == set(SOURCE_FILES), 'unexpected controller source membership')
    for name, wanted in manifest.items():
        path = REPO / name
        require(path.is_relative_to(REPO) and path.absolute() == path.resolve(), 'unexpected source path')
        pin(path, wanted)
    staged = json.loads(pin(root / 'stage-complete.json', stage_sha))
    inputs = staged['input_staging']
    require(staged['state'] == 'INPUTS_STAGED_ORIGINAL_TRAINING_RECEIPTS_MATCHED' and staged['new_slurm_steps'] == 0
            and inputs['inputs_SHA256'] == INPUTS_SHA and inputs['bundle_SHA256'] == BUNDLE_SHA
            and tuple(inputs[key] for key in ('input_pairs', 'planned_windows', 'verified_members')) == COHORT,
            'expected the original staging receipt')
    pin(root / 'head-authority.json', staged['head_authority_SHA256'])
    ledger = read_object(REPO / 'coordination/training/training80-active-node-task-map-v1.json')
    groups = {parent: {key: group['allocations'][parent] for key, group in ledger['groups'].items()}
              for parent in ('1087656', '1103606')}
    fleet = membership(groups, owner_uid=os.getuid())
    parent = fleet['parents'][PARENT]
    require(fleet['membership_qualified'] and parent['coverage_by_group_and_node']['T12'][NODE]['state']
            == 'NO_NUMBERED_STEP', 'scorer node lacks current clean Slurm membership')
    require(all(slurm_row(LABEL + '-' + phase) is None for phase in STAGES), 'original labels already present')
    import_helpers()
    write_json(root / 'score-intent.json', dict(request=request, publication=publication, stage_SHA256=stage_sha,
               manifest_SHA256=manifest_sha, head_authority_SHA256=staged['head_authority_SHA256'],
               epoch=time.time(), max_seconds=1500, parent=PARENT, node=NODE))
    deadline = time.monotonic() + 1500
    command = [HOST_PY, '-B', str(Path(__file__).resolve()), str(root), '--worker', 'snapshot']
    stages = {}
    clean = False
    completed = False
    error = None
    try:
        stages['pre'] = run_step(root, 'pre', command, deadline)
        if stages['pre']['passed']:
            before = read_object(root / 'pre.log')
            require(before['step'] == stages['pre']['row'][0] and before['node'] == NODE, 'preflight step identity differs')
            write_json(root / 'before.json', before)
            stages['gpu'] = run_step(root, 'gpu', gpu_command(root), deadline)
            if stages['gpu']['terminal_visible']:
                stages['post'] = run_step(root, 'post', command, deadline)
                if stages['post']['passed']:
                    after = read_object(root / 'post.log')
                    clean = after['step'] == stages['post']['row'][0] and after['node'] == NODE and all(
                        after[key] == before[key] for key in ('pci', 'pci_to_kfd', 'boot_id'))
                    write_json(root / 'after.json', after)
        if all(stages.get(phase, {}).get('passed') for phase in STAGES) and clean:
            require(read_object(root / 'assignment.json')['step'] == stages['gpu']['row'][0], 'HIP step and accounting differ')
            completed = (root / 'score/progress-private.jsonl').is_file()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        error = type(exc).__name__
        raise
    finally:
        outcome = dict(request=request, publication=publication, stages=stages, clean_post=clean, error=error,
                       completed=completed, independent_prediction_audit=False)
        write_json(root / 'score-outcome.json', outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--worker', choices=('snapshot', 'native'))
    for flag in ('stage-sha', 'request', 'publication', 'manifest-sha'):
        parser.add_argument('--' + flag)
    args = parser.parse_args()
    if args.worker:
        return {'snapshot': clean_snapshot, 'native': native}[args.worker](args.root)
    result = controller(args.root, args.stage_sha, args.request, args.publication, args.manifest_sha)
    print(json.dumps(result), flush=True)
    return int(not result['completed'])


if __name__ == '__main__':
    raise SystemExit(main())
