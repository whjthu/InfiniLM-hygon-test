"""Explicit USGS derivative recipe at the common entry; native Swift owns training."""
import argparse
import csv
import io
import socket
import time
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import hygon_vlm_recipe as vlm

CONFIG_ENV = ('USGS_T04_DOMAIN_OPTIN', 'USGS_T04_DOMAIN_CONFIG', 'USGS_T04_DOMAIN_CONFIG_SHA256')
KEY = 'native_usgs_domain_witness'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def options(argv, flag):
    need(argv.count(flag) == 1, 'Expected one original option: ' + flag)
    start = argv.index(flag) + 1
    end = start
    while end < len(argv) and not argv[end].startswith('--'):
        end += 1
    return start, end


def make_launch(c, original, *, swift, models, plugins, framework, master_port):
    """Pure planner. The separate published contract validates all actual inputs."""
    need(c['case'] == 'T04.sft' and c['mode'] in {'fresh25', 'resume20to25'}
         and c['arm'] in {'domain', 'generic'} and c['device'] == 'cuda'
         and c['formal17'] is False, 'Explicit independent T04 CUDA recipe required')
    output = vlm.absolute_path(c['output'])
    need(output.name == 'witness', 'Witness must have its own fixed scope child')
    scope = output.parent
    framework = vlm.absolute_path(str(framework))
    dataset = vlm.absolute_path(c['training800']['path'])
    plan = vlm.build_launch('T04.sft', original, vlm.absolute_path(str(swift)),
        vlm.absolute_path(str(models)), dataset, vlm.absolute_path(str(plugins)),
        scope / 'train', master_port)
    command = plan['command']
    start, end = options(command, '--callbacks')
    need(command[start:end] == ['nvidia_baseline'], 'Original baseline callback inventory differs')
    command[start:end] = [KEY]
    start, end = options(command, '--external_plugins')
    expected = [str(PurePosixPath(plugins) / 'llava_onevision.py'),
                str(PurePosixPath(plugins) / 'swift_callback_nvidia_baseline_v2.py')]
    need(command[start:end] == expected, 'Original compatibility/baseline plugin inventory differs')
    # The common compatibility provider is the already qualified original implementation.
    replacement = [str(framework / 'vlm/plugins/llava_onevision.py'),
                   str(framework / 'vlm/plugins/usgs_domain_training.py')]
    command[start:end] = replacement
    if c['mode'] == 'resume20to25':
        checkpoint = vlm.absolute_path(c['checkpoint'])
        need(checkpoint != scope / 'train' and checkpoint not in (scope / 'train').parents
             and scope / 'train' not in checkpoint.parents, 'Resume output overlaps checkpoint')
        need(not any(x in command for x in ['--resume_from_checkpoint', '--resume_only_model',
             '--ignore_data_skip', '--load_args']), 'Original resume options unexpectedly present')
        command += ['--resume_from_checkpoint', str(checkpoint), '--resume_only_model', 'false',
                    '--ignore_data_skip', 'false', '--load_args', 'false']
    plan.update(independent_domain_recipe=True, formal17=False, formal_performance=False,
        arm=c['arm'], mode=c['mode'], unique_rows=45, physical_rows=800,
        backend='cuda', common_entry='tools/hygon_training_recipe.py',
        diagnostic_observer={'plugin': replacement[1], 'callback': KEY},
        required_assets={'model_directory': plan['required_assets']['model_directory'],
                         'dataset': str(dataset), 'plugins': replacement})
    return plan


def pinned(path, pin, cap):
    need(path.is_file() and not path.is_symlink() and 0 < pin['bytes'] <= cap,
         'Bounded regular source required')
    before = path.stat()
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
         and len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256'],
         'Published source/input identity differs')
    return raw


def verify_framework(c, root):
    p = c['framework_manifest']
    manifest = json.loads(pinned(Path(p['path']), p, 256 * 1024))
    files = manifest['files']
    need(len(files) == c['framework_members'] and len(files) >= 400,
         'Complete merged common framework required')
    for name, pin in files.items():
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and '..' not in rel.parts, 'Framework path escape')
        pinned(root / name, pin, 4 * 1024**2)
    need({'tools/usgs_domain_launch.py', 'tools/hygon_training_recipe.py',
          'vlm/plugins/usgs_domain_training.py', 'vlm/runtime/usgs_domain_contract.py',
          'vlm/runtime/usgs_domain_witness.py'} <= set(files), 'Merged domain sources missing')
    return len(files)


def verify_owned_cleanup(c):
    """Admit only a held common-entry child of the exact new cleanup supervisor."""
    owned = c['owned_cleanup']
    need(owned['scope_request_id'] == c['request_id'], 'Cleanup belongs to another request')
    anchor = owned['anchor']
    fields = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    need(anchor['PID'] == os.getpid() == anchor['PGID'] == os.getpgrp()
         and anchor['startticks'] == int(fields[19]) and anchor['start_new_session'] is True
         and anchor['request_id'] == c['request_id'] and anchor['uid'] == os.getuid()
         and anchor['host'] == socket.gethostname()
         and anchor['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'Actual held child/session/identity differs')
    source = owned['supervisor_source']
    pinned(Path(source['path']), source, 256 * 1024)
    # Verify only this owned parent's source-path argument; never emit its argv.
    parent_argv = (Path('/proc') / str(c['supervisor']['PID']) / 'cmdline').read_bytes().split(b'\0')
    need(os.fsencode(source['path']) in parent_argv, 'Actual supervisor source is not this parent')
    expected = {
        'held_child': '06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351',
        'owned_processes': '5a3dd9b7530526f324b230ef0fe5be6ff2953f1331718d6520a1362bca229817'}
    for name, digest in expected.items():
        pin = owned[name]
        need(pin['SHA256'] == digest, 'Qualified held cleanup helper differs')
        pinned(Path(pin['path']), pin, 64 * 1024)
    return dict(held_session_identity_verified=True, cleanup_helpers_verified=True,
                resource_release_permitted=False)


def fresh_gpu_admission(config, master_port, remaining):
    """One current ownership snapshot, never a busy/utilization measurement."""
    expected = config['expected_GPU_inventory']
    need(len(expected) == 8 and len({r['UUID'] for r in expected}) == 8, 'CUDA original physical card denominator differs')
    def query(fields, kind):
        timeout = min(10.0, remaining(config) - 10.0)
        need(timeout > 0, 'CUDA admission deadline exhausted')
        completed = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=timeout, check=False)
        need(completed.returncode == 0 and len(completed.stdout) <= 65536, 'CUDA fresh inventory query failed')
        return list(csv.reader(io.StringIO(completed.stdout.strip())))
    def pci(value):
        fields = value.strip().lower().split(':')
        need(len(fields) == 3, 'CUDA PCI encoding differs')
        return f'{int(fields[0], 16):04x}:{int(fields[1], 16):02x}:' + fields[2]
    devices = []
    for row in query('index,uuid,pci.bus_id,memory.free,memory.total', 'gpu'):
        need(len(row) == 5, 'CUDA physical inventory columns differ')
        index, uuid, bus, free, total = [v.strip() for v in row]
        match = next((r for r in expected if int(r['index']) == int(index)), None)
        need(match is not None and uuid == match['UUID'] and pci(bus) == pci(match['PCI']),
                'CUDA physical identity differs from original reference')
        need(0 < int(free) <= int(total), 'CUDA physical memory inventory differs')
        devices.append(dict(index=int(index), UUID=uuid, PCI=pci(bus), free_MiB=int(free), total_MiB=int(total)))
    need(len(devices) == 8 and len({r['index'] for r in devices}) == 8, 'CUDA complete physical snapshot required')
    applications = query('gpu_uuid,pid', 'compute-apps')
    need(not applications, 'CUDA cards have existing compute processes; no process is terminated')
    need(type(master_port) is int and 1024 <= master_port <= 65535, 'CUDA master port differs')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', master_port))
    return dict(observed_epoch=time.time(), observed_monotonic=time.monotonic(), devices=devices,
                active_compute_processes=0, master_port_bindable=True, GPU_busy_measurement=None)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', choices=['T04.sft'])
    p.add_argument('--usgs-domain-config', type=Path, required=True)
    p.add_argument('--usgs-domain-config-sha256', required=True)
    p.add_argument('--a100-launch', type=Path, required=True)
    for n in ['swift', 'models', 'plugins']:
        p.add_argument('--' + n, type=vlm.absolute_path, required=True)
    p.add_argument('--master-port', type=int, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    need(a.usgs_domain_config.is_absolute() and len(a.usgs_domain_config_sha256) == 64
         and set(a.usgs_domain_config_sha256) <= set('0123456789abcdef'), 'Explicit config identity required')
    for key, value in zip(CONFIG_ENV, ['1', str(a.usgs_domain_config), a.usgs_domain_config_sha256]):
        need(key not in os.environ or os.environ[key] == value, 'Inherited domain opt-in differs')
        os.environ[key] = value
    framework = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(framework))
    from vlm.runtime.usgs_domain_contract import load_config, remaining, load_inputs
    from vlm.runtime.usgs_domain_contract import verify_supervisor, verify_provider_files
    c = load_config()
    # Exact merged source closure precedes runtime probing and model loading.
    members = verify_framework(c, framework)
    original = a.a100_launch.read_bytes()
    plan = make_launch(c, original, swift=a.swift, models=a.models, plugins=a.plugins,
                       framework=framework, master_port=a.master_port)
    need(a.receipt == Path(c['output']).parent / 'common-entry-receipt.jsonl', 'Receipt escaped scope')
    with a.receipt.open('x', encoding='utf-8') as receipt:
        def record(stage, **values):
            receipt.write(json.dumps(dict(stage=stage, **values), allow_nan=False) + '\n')
            receipt.flush()
        record('planned', launch=plan, framework_members=members)
        if not a.run:
            return 0
        need(c['root_GO'] is True and c['GPU_GO'] is True, 'Published actual GPU claim required')
        verify_supervisor(c, direct_parent=True)
        record('owned-cleanup-contract', actual=verify_owned_cleanup(c))
        verify_provider_files(c)
        load_inputs(c)
        admission = fresh_gpu_admission(c, a.master_port, remaining)
        record('fresh-admission', actual=admission)
        env = os.environ.copy()
        need(not set(vlm.RANK_ENV).intersection(env), 'Launch from unranked owned parent')
        need(env.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7') == '0,1,2,3,4,5,6,7',
             'Inherited eight-card mask differs')
        env.update(plan['environment_overrides'])
        env['PYTHONPATH'] = str(framework) + os.pathsep + env.get('PYTHONPATH', '')
        for path in [str(a.swift), plan['required_assets']['dataset'], *plan['required_assets']['plugins']]:
            need(Path(path).is_file(), 'Required native training input missing')
        need(Path(plan['required_assets']['model_directory']).is_dir(), 'Native model directory missing')
        source = framework / 'generation/src'
        probe = subprocess.run([sys.executable, '-c', vlm.RUNTIME_PROBE, str(source)],
            env=env, capture_output=True, text=True, check=True, timeout=min(45, remaining(c) - 15))
        need(len(probe.stdout) <= 65536, 'Runtime receipt exceeded cap')
        observed = json.loads(probe.stdout)
        sys.path.insert(0, str(source))
        from gencuda.runtime import RuntimeInfo, validate_accelerator
        validate_accelerator(RuntimeInfo(**observed), 'cuda', 8)
        record('runtime', observed=observed)
        output = Path(plan['environment_overrides']['OUTPUT_DIR'])
        output.mkdir(exist_ok=False)
        record('running')
        budget = remaining(c) - 15
        need(budget > 0, 'Original native deadline exhausted')
        need(0 <= time.monotonic() - admission['observed_monotonic'] <= 60,
             'Original admission is stale; do not refresh or submit')
        result = subprocess.run([sys.executable, *plan['command']], env=env, timeout=budget)
        record('exited', returncode=result.returncode)
        return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
