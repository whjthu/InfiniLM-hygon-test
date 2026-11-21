"""Run the frozen GLM I128 component link or single-device comparison once."""

import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

import link_glm_i128_component_providers as link
from read_training_slurm_membership import allocated_dcu, snapshot as membership


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
RESULTS = ROOT / 'results'
PARENT, NODE = link.PARENT, link.NODE
LABEL = 'GLM-I128-four-routed-variants-one-device-v1'
CALLER = Path('code/bw1000/GLM-C1-layer-observer-r474/routed-i128-one-gpu-caller-v1')
CALLER_MF = 'ce395475027bb715363d6b8bc03cd5656d7b2ff401c610592fbbedd091a5df3a'
INPUT_MF = '19a5fcd6e25ae80ecfbe538d9f83bac2edb9d009d128b4d1a7e15191e2cf645e'
SNAPSHOT = Path('code/bw1000/training-qwen-native25-hip-v3')
KFD = Path('code/bw1000/GLM-C1-layer-observer-r474/TP16-FFN-f32-transport-pair-v1')
STEP = re.compile(PARENT + r'\.\d+')
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
HOST_PY = '/usr/local/python3.12/bin/python3'
GLM_ENV = ('/refactor/candidates/latest-family-native-r269/GLM/python:'
           '/refactor/install/hygon-glm-5f91ad7-r3672/lib:'
           '/refactor/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib:'
           '/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch.libs:'
           '/opt/dtk/cuda/cuda/lib64:/opt/dtk/lib:/usr/local/lib')
LINK_REQUEST = sha256((link.LABEL + ':' + CALLER_MF).encode()).hexdigest()[:32]
GPU_REQUEST = sha256((LABEL + ':' + CALLER_MF).encode()).hexdigest()[:32]
PHASES = {'link': (0, 4, 8192, 300), 'pre': (0, 2, 2048, 60),
          'gpu': (1, 4, 16384, 300), 'post': (0, 2, 2048, 60)}
FILES = ('tools/control_glm_i128_component.py', 'tools/link_glm_i128_component_providers.py',
         'tools/read_training_slurm_membership.py', 'coordination/training/training80-active-node-task-map-v1.json',
         (SNAPSHOT / 'snapshot.py').as_posix(), (SNAPSHOT / 'slurm_accounting.py').as_posix(),
         (KFD / 'assigned_kfd.py').as_posix(), (CALLER / 'SOURCE_FILES.json').as_posix())
CONTROL_MF = 'coordination/tuning/glm-i128-component-controller-files-v1.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def verify_sources(wanted):
    pins = json.loads(link.read_pin(REPO / CONTROL_MF, wanted))
    need(set(pins) == set(FILES), 'unexpected controller source members')
    for name, pin in pins.items():
        link.read_pin(REPO / name, pin)
    members = json.loads(link.read_pin(REPO / CALLER / 'SOURCE_FILES.json', CALLER_MF))['files_SHA256']
    need(len(members) == 7 and all(Path(name).name == name for name in members), 'caller source members differ')
    for name, pin in members.items():
        link.read_pin(REPO / CALLER / name, pin)


def node_is_clear():
    raw = (REPO / 'coordination/training/training80-active-node-task-map-v1.json').read_bytes()
    ledger = json.loads(raw)
    groups = {parent: {zone: item['allocations'][parent] for zone, item in ledger['groups'].items()}
              for parent in ('1087656', '1103606')}
    report = membership(groups, owner_uid=os.getuid())
    node = report['parents'][PARENT]['coverage_by_group_and_node']['I02'][NODE]
    need(report['membership_qualified'] and node['state'] == 'NO_NUMBERED_STEP', 'original I02 node is not clear')


def accounting():
    raw = subprocess.check_output(['sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-19T00:00:00', '-o', COLUMNS], timeout=8)
    need(len(raw) <= 2*1024**2, 'unexpected accounting bytes')
    return raw


def label_for(phase):
    return link.LABEL if phase == 'link' else LABEL + '-' + phase


def row_for(raw, label):
    rows = [line.split('|') for line in raw.decode().splitlines() if line.strip()]
    need(all(len(row) == 7 for row in rows), 'malformed independent Slurm accounting')
    rows = [row for row in rows if row[1] == label]
    need(len(rows) <= 1, 'original phase label is duplicated')
    return rows[0] if rows else None


def srun(phase, command):
    cards, cpus, memory, seconds = PHASES[phase]
    return ['srun', '--jobid='+PARENT, '--exact', '--immediate=5', '--cpu-bind=none', '-N1', '-n1',
            f'-c{cpus}', f'--mem={memory}M', '--gres='+(f'dcu:{cards}' if cards else 'none'), '-w', NODE,
            '-t', str(seconds // 60), '--job-name='+label_for(phase), *command]


def run_step(run, phase, command, deadline):
    sys.path.insert(0, str(REPO / SNAPSHOT))
    from slurm_accounting import wait_terminal

    label = label_for(phase)
    need(row_for(accounting(), label) is None, 'original phase may have been submitted')
    left = min(PHASES[phase][3] + 5, deadline - time.monotonic() - 12)
    need(left > 5, 'original phase budget exhausted')
    report = dict(rc=None, terminal_visible=False, row=None, passed=False)
    try:
        with (run / (phase + '.log')).open('xb') as log:
            try:
                report['rc'] = subprocess.run(srun(phase, command), stdout=log, stderr=subprocess.STDOUT,
                                               timeout=left).returncode
            except subprocess.TimeoutExpired:
                report['client_timeout'] = True
        report.update(wait_terminal(lambda: row_for(accounting(), label), min(deadline-9, time.monotonic()+20),
                                    max_observations=7))
        row = report['row']
        report['passed'] = bool(report['rc'] == 0 and row and STEP.fullmatch(row[0]) and row[2:4] == ['COMPLETED', '0:0']
                                and row[5] == NODE and allocated_dcu(row[6]) == PHASES[phase][0])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report['observation_error'] = type(error).__name__
        raise
    finally:
        link.write(run / (phase + '-accounting.json'), report)
    return report


def mount(run, inner, for_link=False):
    cmd = ['/public/software/apps/apptainer/1.3.4/bin/apptainer', 'exec', '--rocm', '--no-home', '--pwd', '/']
    binds = [(ROOT, '/refactor', 'ro'), (REPO, '/source', 'ro'), (run, '/scope', 'rw'),
             (Path('/opt/hyhal'), '/opt/hyhal', 'ro')]
    if for_link:
        binds.append((ROOT / 'candidates', '/refactor/candidates', 'rw'))
    for host, target, mode in binds:
        cmd += ['--bind', f'{host}:{target}:{mode}']
    runtime = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
               'export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HSA_FORCE_FINE_GRAIN_PCIE=1; '
               'unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH; '
               'export LD_LIBRARY_PATH="' + GLM_ENV + ':${LD_LIBRARY_PATH:-}"; ')
    if for_link:
        runtime += 'export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
    return cmd + [str(ROOT / 'containers/dtk26-sandbox-r1'), '/bin/bash', '--noprofile', '--norc', '-c',
                  runtime + 'exec ' + shlex.join(inner)]


def small_json(path):
    with path.open('rb') as stream:
        raw = stream.read(4*1024**2+1)
    need(0 < len(raw) <= 4*1024**2, 'unexpected original metadata size')
    return json.loads(raw)


def snapshot_helper():
    sys.path[:0] = [str(REPO / SNAPSHOT), str(REPO / KFD)]
    import snapshot
    snapshot.NODE, snapshot.STEP = NODE, STEP
    return snapshot


def host_snapshot(run, phase):
    info = snapshot_helper().snapshot(False)
    need(info['full_fd_inventory_performed'] and info['kfd_holder_count'] == 0, 'original PCI device is in use')
    info['owned_registry_present'] = None
    if phase == 'post':
        owner_file = run / 'worker-owner.json'
        info['owned_registry_present'] = owner_file.is_file()
        if info['owned_registry_present']:
            owner = small_json(owner_file)
            need(owner['request'] == GPU_REQUEST and owner['boot_id'] == info['boot_id']
                 and type(owner['pid']) is int and owner['pid'] > 1,
                 'original worker process identity differs')
            try:
                fields = Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(') ', 1)[1].split()
            except FileNotFoundError:
                pass
            else:
                need(fields[0] == 'Z' or int(fields[19]) != owner['startticks'], 'original GPU worker is still live')
    print(json.dumps(info), flush=True)


def device_worker(run):
    scope = small_json(run / 'intent.json')
    before = small_json(run / 'before.json')
    inventory = snapshot_helper()
    step = os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', '')
    current_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    need(scope['request'] == GPU_REQUEST and scope['replay_allowed'] is False and scope['worker_deadline'] == scope['epoch']+300
         and STEP.fullmatch(step) and socket.gethostname().split('.')[0] == NODE and len(os.sched_getaffinity(0)) >= 4,
         'incorrect original component step identity')
    need(before['pci'] == inventory.drm_inventory() and not inventory.kernel_holders() and
         before['boot_id'] == current_boot and 0 <= time.monotonic_ns()-before['monotonic_ns'] <= 45_000_000_000,
         'fresh original whole-node PCI admission differs')
    info = json.loads(link.read_pin(link.OUTPUT / 'CPU-link-result.json', scope['link_receipt_SHA256']))
    raw = link.read_pin(run / 'original-link-accounting.psv', scope['link_accounting_SHA256'])
    dependencies = link.admitted_dependencies(info, raw, link.load_auditor())
    hip = ctypes.CDLL(dependencies['hip']['path'])
    count, identifier = ctypes.c_int(), ctypes.create_string_buffer(32)
    need(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
         and hip.hipDeviceGetPCIBusId(identifier, len(identifier), 0) == 0, 'expected exactly one physical DCU')
    pci = identifier.value.decode().lower()
    if len(pci) == 7:
        pci = '0000:' + pci
    need(pci in before['pci'], 'assigned PCI is outside original admitted node')
    from assigned_kfd import assigned_kfd
    link.write(run / 'worker-owner.json', dict(request=GPU_REQUEST, step=step, node=NODE, pci=pci, kfd=assigned_kfd(pci),
               pid=os.getpid(), boot_id=current_boot,
               startticks=int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])))
    binding = link.worker_binding(dependencies, pci, scope['epoch'], scope['worker_deadline'])
    link.write(run / 'binding.json', binding)
    remain = int(scope['worker_deadline'] - time.time())
    need(remain > 10, 'original component budget exhausted before caller')
    argv = [sys.executable, '-I', '-S', '-B', str(REPO / CALLER / 'caller.py'), '--binding', str(run / 'binding.json'),
            '--out', str(run / 'component-private.json')]
    os.execv('/usr/bin/timeout', ['/usr/bin/timeout', '--kill-after=5s', f'{remain}s', *argv])


def run_link(args):
    need(all(getattr(args, arm + '_audit') is not None and link.check_sha(getattr(args, arm + '_sha256'))
             for arm in link.ROLES), 'two exact original compiler audits are required')
    audits = {arm: link.read_pin(getattr(args, arm + '_audit').absolute(), getattr(args, arm + '_sha256'))
              for arm in link.ROLES}
    records = {arm: json.loads(raw) for arm, raw in audits.items()}
    auditor = link.load_auditor()
    link.adopted_objects(records, auditor)
    raw = accounting()
    for arm, record in records.items():
        need(auditor.accounting(raw, auditor.load(arm)[0]) == record['accounting'], 'historical compiler step is not confirmed')
    node_is_clear()
    need(row_for(raw, link.LABEL) is None and not (ROOT / 'candidates' / link.LABEL).exists(), 'original link already exists')
    run = RESULTS / (link.LABEL + '-controller')
    run.mkdir(mode=0o700)
    epoch, deadline = time.time(), time.monotonic()+300
    link.write(run/'intent.json', dict(request=LINK_REQUEST, publication=args.publication, epoch=epoch, deadline=epoch+300,
               replay_allowed=False, original_compiles={arm: row['original_request'] for arm, row in records.items()}))
    command = ['/usr/bin/python3', '-I', '-S', '-B', '/source/tools/link_glm_i128_component_providers.py', 'link',
               '--epoch', str(epoch), '--deadline', str(epoch+300)]
    for arm, value in audits.items():
        pin = link.write(run / (arm + '-audit.json'), value)
        command += ['--'+arm+'-audit', '/scope/'+arm+'-audit.json', '--'+arm+'-sha256', pin]
    phase = run_step(run, 'link', mount(run, command, for_link=True), deadline)
    return dict(scope='original_CPU_link', request=LINK_REQUEST, phase=phase, actual_GPU_steps=0, device_qualified=False)


def component_result_pass(run, row):
    measured = small_json(run/'component-private.json')
    owner = small_json(run/'worker-owner.json')
    with (run/'binding.json').open('rb') as stream:
        binding = stream.read(65537)
    need(0 < len(binding) <= 65536, 'original component binding exceeds its byte bound')
    expected = json.loads(binding)
    return (measured['passed'] is True and measured['completed'] is True
            and measured['fixture_binding_sha256'] == INPUT_MF and measured['normal_stream_and_event_destroy'] is True
            and measured['binding_sha256'] == sha256(binding).hexdigest() and measured['providers'] == expected['providers']
            and measured['hip'] == expected['hip'] and measured['device']['pci'] == owner['pci'] == expected['expected_pci']
            and measured['device']['visible_devices'] == 1 and measured['passed_timed_launches'] == 3840
            and measured['numeric_counts']['PASS'] == 24 and measured['pair_counts']['PASS'] == 48
            and owner['request'] == GPU_REQUEST and owner['step'] == row[0])


def run_component(args):
    need(link.check_sha(args.link_sha256), 'exact original CPU link receipt pin is required')
    raw = accounting()
    stored = ROOT / 'candidates' / link.LABEL / 'CPU-link-result.json'
    report = json.loads(link.read_pin(stored, args.link_sha256))
    row = link.load_auditor().accounting(raw, dict(label=link.LABEL, parent=PARENT, node=NODE))
    need(report['state'] == 'ISOLATED_HOST_LINK_PASS' and report['actual_CPU_step'] == row[0]
         and row[2:4] == ['COMPLETED', '0:0'], 'original CPU link is not independently terminal')
    node_is_clear()
    need(all(row_for(raw, label_for(phase)) is None for phase in ('pre','gpu','post')), 'original component already exists')
    run = RESULTS / LABEL
    run.mkdir(mode=0o700)
    epoch, end = time.time(), time.monotonic()+420
    link.write(run/'intent.json', dict(request=GPU_REQUEST, publication=args.publication, epoch=epoch, worker_deadline=epoch+300,
               link_receipt_SHA256=args.link_sha256, link_accounting_SHA256=link.write(run/'original-link-accounting.psv', raw),
               replay_allowed=False, one_DCU=True, fixture_only=True))
    snapshot = [HOST_PY, '-B', str(Path(__file__).resolve()), '--run-root', str(run), '--snapshot']
    phases, qualified, clean = {}, False, False
    try:
        phases['pre'] = run_step(run, 'pre', [*snapshot,'pre'], end)
        if phases['pre']['passed']:
            before = small_json(run/'pre.log')
            need(before['step'] == phases['pre']['row'][0], 'original preflight data differs')
            link.write(run/'before.json', before)
            cmd = ['/usr/bin/python3', '-B', '/source/tools/control_glm_i128_component.py', '--run-root', '/scope', '--device-worker']
            phases['gpu'] = run_step(run, 'gpu', mount(run, cmd), end)
            if phases['gpu']['terminal_visible']:
                phases['post'] = run_step(run, 'post', [*snapshot,'post'], end)
                if phases['post']['passed']:
                    after = small_json(run/'post.log')
                    clean = (after['step'] == phases['post']['row'][0] and after['pci'] == before['pci']
                             and after['boot_id'] == before['boot_id'] and after['owned_registry_present'] is True)
                    link.write(run/'after.json', after)
            if clean and phases['gpu']['passed']:
                qualified = component_result_pass(run, phases['gpu']['row'])
    finally:
        outcome = dict(scope='synthetic_one_device_component', request=GPU_REQUEST, passed=qualified, phases=phases,
                       original_post_clean=clean, independent_numerical_audit=False, full_model_qualified=False)
        link.write(run/'outcome.json', outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('link', 'component'))
    parser.add_argument('--snapshot', choices=('pre', 'post'))
    parser.add_argument('--device-worker', action='store_true')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--publication')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--link-sha256')
    for arm in link.ROLES:
        parser.add_argument('--'+arm+'-audit', type=Path)
        parser.add_argument('--'+arm+'-sha256')
    args = parser.parse_args()
    if args.snapshot:
        host_snapshot(args.run_root, args.snapshot)
        return 0
    if args.device_worker:
        device_worker(args.run_root)
        return 1
    need(args.mode and re.fullmatch('[0-9a-f]{40}', args.publication or '') and link.check_sha(args.manifest_sha256),
         'mode, actual publication and source manifest required')
    verify_sources(args.manifest_sha256)
    result = run_link(args) if args.mode == 'link' else run_component(args)
    print(json.dumps(result), flush=True)
    return int(not (result['phase']['passed'] if args.mode == 'link' else result['passed']))


if __name__ == '__main__':
    raise SystemExit(main())
