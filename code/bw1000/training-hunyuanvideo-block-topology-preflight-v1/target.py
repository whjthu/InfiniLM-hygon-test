"""Run one bounded zero-GRES probe of the production T2V block topology."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
OLD = ROOT / 'results/t09-step6-block-de7576437787'
ORIGINAL_SOURCE = OLD / 'src'
PARENT, NODE = '1103606', 'f13r3n13'
MANAGED_PYTHON = '/usr/local/python3.12/bin/python3'
GUEST_NAMES = {'topology.py': '4302db233bf66143a607d3a295a27033ee1ede7159b2878facacd593020d2198',
               'backward_observer.py': '0043b4319d2a682e2d0a427e36c966b6ef5a7402419bab828000887d5c6c66f3'}
ACCOUNTING = 'JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES%180,Start,End'
TIMEOUT = 125


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + '\n').encode()


def write_new(path, data):
    with path.open('xb') as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def read_query(argv, deadline):
    left = min(6, deadline - time.monotonic())
    require(left > 0, 'scheduler_read_deadline')
    process = subprocess.run(argv, env=dict(os.environ, TZ='UTC', LC_ALL='C',
                             SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S'),
                             capture_output=True, timeout=left, check=True)
    require(len(process.stdout) <= 2 << 20, 'scheduler_output_extent')
    return process.stdout.decode('utf-8').splitlines()


def accounting(identifier, deadline):
    return [line.split('|') for line in read_query(['sacct', '-nP', '-j', identifier,
                                                     '-o', ACCOUNTING], deadline)]


def placement_source():
    for name, digest in {'placement.py': 'a7b5c52e71ab4c953d50f0c72080190b227b40c67600f7b2f203370c68a3a386',
                         'PLACEMENT.json': '620a62c9801e5bd2505b87d457fccd537a67cd529ab2d35caaca8dbd0f1e1312'}.items():
        path = ORIGINAL_SOURCE / name
        require(path.resolve(strict=True) == path and path.stat().st_size <= 32 << 10
                and sha256(path.read_bytes()).hexdigest() == digest, 'exact_original_placement_source')
    sys.path.insert(0, str(ORIGINAL_SOURCE))
    import placement
    require(placement.PARENT == PARENT and placement.NODE == NODE, 'exact_original_node')
    return placement


def cpu_resources(text):
    values = text.split(',')
    return len(values) == 3 and set(values) in ({'cpu=2', 'node=1', 'mem=4G'},
                                                {'cpu=2', 'node=1', 'mem=4096M'})


def compute(run_dir, label):
    placement = placement_source()
    step = placement.numeric_cgroup()
    require(socket.gethostname().split('.')[0] == NODE and not os.environ.get('SLURM_STEP_GPUS'),
            'cpu_node_and_empty_gpu_allocation')
    listing = read_query(['scontrol', 'show', 'step', step, '--oneliner'], time.monotonic() + 7)
    rows = [row for line in listing if (row := placement.fields(line)).get('StepId') == step]
    require(len(rows) == 1, 'cpu_step_controller_row')
    row = rows[0]
    require(row.get('Name', row.get('JobName')) == label and row.get('State') == 'RUNNING'
            and row.get('NodeList') == NODE and cpu_resources(row.get('TRES', row.get('AllocTRES', ''))),
            'cpu_step_controller_resources')
    runtime = ROOT / 'results/training-t06-sdxl-runtime-v1'
    native = ROOT / 'results/training-hunyuanvideo-original-source-v1/source'
    alias = '/t09-native/hunyuanvideo-1.5'
    config = ROOT / 'results/training-hunyuanvideo-original-inputs-v1/config.json'
    executable = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
    env = ('set -euo pipefail\nsource /opt/dtk/env.sh >&2\nunset PYTHONPYCACHEPREFIX\n'
           'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\n'
           'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1\n'
           'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/training-runtime/site\n'
           'exec python3 -B -u "$@"')
    args = [executable, 'exec', '--rocm', '--no-mount', 'home,cwd', '--pwd', '/',
            '--bind', f'{ROOT}:{ROOT}:ro', '--bind', f'{runtime}:/training-runtime:ro',
            '--bind', f'{native}:{alias}:ro', '--bind', '/opt/hyhal:/opt/hyhal:ro',
            '--bind', f'{OLD}/overlay/flash_attn_no_pad.py:{alias}/hyvideo/utils/flash_attn_no_pad.py:ro',
            str(ROOT / 'containers/dtk26-sandbox-r1'), '/bin/bash', '--noprofile', '--norc', '-c', env, label,
            str(run_dir / 'topology.py'), '--native-root', alias, '--config', str(config),
            '--launch', str(ORIGINAL_SOURCE / 'ORIGINAL_LAUNCH.json'),
            '--observer', str(run_dir / 'backward_observer.py'), '--backend', str(ORIGINAL_SOURCE / 'backward_backend.py')]
    print('T09_META_STEP ' + step, flush=True)
    os.execv(executable, args)


def receive(payload):
    request = payload['request']
    publication = payload['publication']
    require(re.fullmatch('[a-f0-9]{32}', request) and re.fullmatch('[a-f0-9]{40}', publication), 'request_and_publication')
    require(os.getuid() == 154459 and set(payload['sources']) == GUEST_NAMES.keys(), 'owner_and_source_set')
    bodies = {name: base64.b64decode(payload['sources'][name], validate=True) for name in GUEST_NAMES}
    require(all(0 < len(body) < 32 << 10 and sha256(body).hexdigest() == GUEST_NAMES[name]
                for name, body in bodies.items()), 'guest_source_digests')
    carrier = base64.b64decode(payload['carrier'], validate=True)
    require(0 < len(carrier) < 32 << 10 and sha256(carrier).hexdigest() == payload['carrier_sha256'],
            'carrier_digest')
    label = 't09-meta-topology-' + request[:12]
    run_dir = ROOT / 'results' / label
    placement = placement_source()
    with placement.shared_lock(ROOT):
        deadline = time.monotonic() + 18
        placement.live_views(lambda argv: read_query(argv, deadline), os.getuid(), time.time())
        require(not any(len(row) == 9 and row[1] == label for row in accounting(PARENT, deadline)),
                'no_prior_request_label')
        run_dir.mkdir(mode=0o700)
        for name, content in bodies.items():
            write_new(run_dir / name, content)
        write_new(run_dir / 'target.py', carrier)
        write_new(run_dir / 'intent.json', encode({'request': request, 'publication': publication,
                   'label': label, 'cpu': 2, 'memory_MiB': 4096, 'gres': 0, 'worker_seconds': 120}))
    cmd = ['srun', '--jobid='+PARENT, '--exact', '--nodes=1', '--ntasks=1', '--nodelist='+NODE,
           '--immediate=5', '--cpus-per-task=2', '--mem=4096M', '--gres=none', '--time=00:02:00',
           '--job-name='+label, MANAGED_PYTHON, '-I', '-S', '-B', str(run_dir / 'target.py'), '--compute', str(run_dir), label]
    result = {'request': request, 'publication': publication, 'label': label, 'actual_GPU': 0,
              'passed': False, 'Slurm_attempted': True, 'replay_allowed': False}
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT)
        result['returncode'] = p.returncode
        require(len(p.stdout) <= 32 << 10 and len(p.stderr) <= 128 << 10, 'bounded_captured_process_output')
        write_new(run_dir / 'stdout.private', p.stdout)
        write_new(run_dir / 'stderr.private', p.stderr)
        result['stderr_SHA256'] = sha256(p.stderr).hexdigest()
        result['stdout_SHA256'] = sha256(p.stdout).hexdigest()
        lines = p.stdout.decode('utf-8').splitlines()
        handles = [line.removeprefix('T09_META_STEP ') for line in lines if line.startswith('T09_META_STEP ')]
        if handles:
            step, = handles
            require(re.fullmatch(PARENT + r'\.[0-9]+', step), 'exact_numbered_step')
            result['step'] = step
            stop = time.monotonic() + 12
            while time.monotonic() < stop - 2:
                rows = [row for row in accounting(step, stop) if row[0] == step]
                if len(rows) == 1 and rows[0][2] in placement.TERMINAL:
                    result['accounting'] = rows[0]
                    break
                time.sleep(0.35)
        require(p.returncode == 0 and len(lines) == 2 and len(handles) == 1, 'actual_guest_completion')
        guest = json.loads(next(line for line in lines if not line.startswith('T09_META_STEP ')))
        row = result.get('accounting', [])
        require(len(row) == 9 and row[0:4] == [step, label, 'COMPLETED', '0:0']
                and row[5] == NODE and cpu_resources(row[6]), 'exact_zero_gres_terminal_accounting')
        expect = {'double_blocks': 54, 'single_blocks': 0}
        require(guest['passed'] and guest['actual_family_counts'] == guest['actual_wrapped'] == expect
                and guest['GPU_calls'] == guest['forward_calls'] == guest['optimizer_updates']
                == guest['weight_payloads_opened'] == 0,
                'native_meta_topology_only')
        result.update(native=guest, passed=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result['error_type'] = type(error).__name__
        if isinstance(error, ValueError) and re.fullmatch('[a-z_]{3,80}', str(error)):
            result['error_code'] = str(error)
    write_new(run_dir / 'terminal.json', encode(result))
    return result


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--compute':
        compute(Path(sys.argv[2]), sys.argv[3])
    else:
        print(json.dumps(receive(json.load(sys.stdin)), sort_keys=True), flush=True)
