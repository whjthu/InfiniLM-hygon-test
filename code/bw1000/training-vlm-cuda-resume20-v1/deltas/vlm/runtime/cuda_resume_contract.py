"""Pure-stdlib admission for the common T03/T04 CUDA resume observer."""
import hashlib
import csv
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import time


CASES = {'T03.sft', 'T04.sft'}
MEMBERS = {'trainer_state.json', 'args.json', 'adapter_model.safetensors', 'adapter_config.json',
           'optimizer.pt', 'scheduler.pt', *(f'rng_state_{rank}.pth' for rank in range(8))}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def remaining(config):
    epoch, deadline = config['epoch'], config['deadline']
    require(type(epoch) in (int, float) and type(deadline) in (int, float)
            and 0 < deadline - epoch <= 900 and epoch <= time.time() < deadline, 'CUDA resume original epoch exhausted')
    return deadline - time.time()


def checked_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and not path.is_symlink(), 'CUDA resume invalid asset path')
    return path


def bounded_bytes(path, maximum, *, library=False):
    path = checked_path(str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum
                and before.st_uid in ((0, os.getuid()) if library else (os.getuid(),)),
                'bounded CUDA resume metadata type/size/owner differs')
        with os.fdopen(fd, 'rb', buffering=0, closefd=False) as stream:
            raw = stream.read(before.st_size + 1)
        after = os.fstat(fd); current = path.stat()
        key = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        require(len(raw) == before.st_size and key(before) == key(after) == key(current),
                'bounded CUDA resume metadata changed')
        return raw
    finally:
        os.close(fd)


def check_config(config):
    require(config['case'] in CASES and config['device'] == 'cuda'
            and config['root_GO'] is True and config['GPU_GO'] is True
            and config['checkpoint_step'] == 20 and config['max_steps'] == 25
            and config['world_size'] == 8 and config['microbatches_per_rank'] == 20,
            'CUDA resume case/device/scope opt-in differs')
    require(config['formal_numerics'] is False and config['formal_performance'] is False,
            'CUDA resume engineering scope cannot declare formal quality/performance')
    remaining(config)
    require(config['comparison_policy'] == 'EXACT_ENGINEERING_SAVED_STATE'
            and config['original_tail_micro_reference_available'] is False,
            'CUDA resume comparison/evidence boundary differs')
    original = checked_path(config['original_root'])
    require(checked_path(config['checkpoint']) == original / 'train/checkpoint-20'
            and checked_path(config['comparison_checkpoint']) == original / 'train/checkpoint-25',
            'CUDA resume original checkpoint paths differ')
    output = checked_path(config['output'])
    require(output != original and original not in output.parents and output not in original.parents,
            'CUDA resume output overlaps original namespace')
    for key in ('checkpoint_files', 'comparison_files'):
        pins = config[key]
        require(set(pins) == MEMBERS and sum(p['bytes'] for p in pins.values()) <= 512 * 1024**2,
                'CUDA resume requires bounded complete state')
        for pin in pins.values():
            require(type(pin['bytes']) is int and 0 < pin['bytes'] <= 512 * 1024**2
                    and isinstance(pin['SHA256'], str) and len(pin['SHA256']) == 64
                    and not set(pin['SHA256']) - set('0123456789abcdef'), 'CUDA resume file pin differs')
    return config


def load_config(environment=None):
    environment = os.environ if environment is None else environment
    require(environment.get('NATIVE_CUDA_RESUME_GO') == '1', 'CUDA resume requires explicit device opt-in')
    path = checked_path(environment.get('NATIVE_RESUME_CONFIG', ''))
    raw = bounded_bytes(path, 65536)
    require(0 < len(raw) <= 65536 and hashlib.sha256(raw).hexdigest() == environment.get('NATIVE_RESUME_CONFIG_SHA256'),
            'CUDA resume configuration identity differs')
    return check_config(json.loads(raw))


def verify_file(path, pin, check):
    check(); path = checked_path(str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                and before.st_size == pin['bytes'], 'CUDA resume reference file metadata differs')
        digest = hashlib.sha256(); consumed = 0
        with os.fdopen(fd, 'rb', buffering=0, closefd=False) as stream:
            while consumed < pin['bytes']:
                check(); chunk = stream.read(min(1024**2, pin['bytes'] - consumed))
                require(chunk, 'CUDA resume reference short read')
                digest.update(chunk); consumed += len(chunk)
        after = os.fstat(fd); current = path.stat()
        key = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        require(key(before) == key(after) == key(current) and digest.hexdigest() == pin['SHA256'],
                'CUDA resume reference file identity changed')
        return dict(bytes=consumed, SHA256=digest.hexdigest())
    finally:
        os.close(fd)


def verify_references(config):
    """One controller CPU check before model construction, not a new hash benchmark."""
    check_config(config); result = {}; total = 0
    raw = bounded_bytes(checked_path(config['binding_path']), 2 * 1024**2)
    require(hashlib.sha256(raw).hexdigest() == config['binding_SHA256']
            and json.loads(raw)['spec']['case'] == config['case'], 'CUDA resume native case binding differs')
    for root_key, pins_key in [('checkpoint', 'checkpoint_files'), ('comparison_checkpoint', 'comparison_files')]:
        root = checked_path(config[root_key]); rows = {}
        for name, pin in sorted(config[pins_key].items()):
            rows[name] = verify_file(root / name, pin, lambda: remaining(config)); total += rows[name]['bytes']
        result[root_key] = rows
    require(total <= 1024**3, 'CUDA resume reference total byte budget exceeded')
    reference = config['canonical64_reference']['document']
    require(config['canonical64_reference']['rows'] == 64 and reference['bytes'] <= 1024**2,
            'CUDA resume canonical64 reference identity differs')
    doc = verify_file(checked_path(config['canonical64_reference_staged_path']), reference, lambda: remaining(config))
    return dict(files=result, whole_reference_bytes=total, canonical64_reference=doc,
                original_tail_micro_match_qualified=False, GPU_compute=False)


def verify_supervisor(config, *, direct_parent=False):
    """Exact existing A100 controller identity; no process scans or foreign-host PID inference."""
    row = config['supervisor']
    require(row['host'] == socket.gethostname() and row['uid'] == os.getuid()
            and row['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'CUDA resume supervisor host/UID/boot differs')
    require(type(row['PID']) is int and row['PID'] > 0
            and (not direct_parent or row['PID'] == os.getppid()), 'CUDA resume owned supervisor differs')
    proc = Path('/proc') / str(row['PID'])
    require(proc.stat().st_uid == os.getuid(), 'CUDA resume supervisor UID differs')
    fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    require(int(fields[19]) == row['startticks'] and fields[0] not in ('Z', 'X'),
            'CUDA resume supervisor identity not live')
    require(row.get('scope_request_id') == config['request_id'], 'CUDA supervisor belongs to another request')
    age = float(Path('/proc/uptime').read_text().split()[0]) - row['startticks'] / os.sysconf('SC_CLK_TCK')
    require(0 <= age <= time.time() - config['epoch'] + 10,
            'CUDA supervisor must be a new instance for this original epoch')
    return row


def verify_provider_files(config):
    require(len(config['providers']) == 15, 'CUDA resume provider source denominator differs')
    roots = [Path('/home/whj/nvidia-baseline/env/lib/python3.12/site-packages'),
             Path('/usr/local/lib/python3.12/dist-packages')]
    result = []
    for row in config['providers']:
        path = Path(row['canonical_path'])
        allowed = any(path.is_relative_to(root) for root in roots)
        allowed |= path == Path('/home/whj/nvidia-baseline/swift_callback_nvidia_baseline_v2.py')
        require(path.is_absolute() and allowed and path.resolve(strict=True) == path
                and 0 < row['bytes'] <= 1024**2, 'CUDA resume provider canonical path differs')
        remaining(config); raw = bounded_bytes(path, 1024**2, library=True)
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['SHA256'],
                'CUDA resume provider source changed')
        result.append(dict(path=str(path), bytes=len(raw), SHA256=row['SHA256']))
    return result


def fresh_gpu_admission(config, master_port):
    """One current ownership snapshot, never a busy/utilization measurement."""
    expected = config['expected_GPU_inventory']
    require(len(expected) == 8 and len({r['UUID'] for r in expected}) == 8, 'CUDA original physical card denominator differs')
    def query(fields, kind):
        timeout = min(10.0, remaining(config) - 10.0)
        require(timeout > 0, 'CUDA admission deadline exhausted')
        completed = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=timeout, check=False)
        require(completed.returncode == 0 and len(completed.stdout) <= 65536, 'CUDA fresh inventory query failed')
        return list(csv.reader(io.StringIO(completed.stdout.strip())))
    def pci(value):
        fields = value.strip().lower().split(':')
        require(len(fields) == 3, 'CUDA PCI encoding differs')
        return f'{int(fields[0], 16):04x}:{int(fields[1], 16):02x}:' + fields[2]
    devices = []
    for row in query('index,uuid,pci.bus_id,memory.free,memory.total', 'gpu'):
        require(len(row) == 5, 'CUDA physical inventory columns differ')
        index, uuid, bus, free, total = [v.strip() for v in row]
        match = next((r for r in expected if int(r['index']) == int(index)), None)
        require(match is not None and uuid == match['UUID'] and pci(bus) == pci(match['PCI']),
                'CUDA physical identity differs from original reference')
        require(0 < int(free) <= int(total), 'CUDA physical memory inventory differs')
        devices.append(dict(index=int(index), UUID=uuid, PCI=pci(bus), free_MiB=int(free), total_MiB=int(total)))
    require(len(devices) == 8 and len({r['index'] for r in devices}) == 8, 'CUDA complete physical snapshot required')
    applications = query('gpu_uuid,pid', 'compute-apps')
    require(not applications, 'CUDA cards have existing compute processes; no process is terminated')
    require(type(master_port) is int and 1024 <= master_port <= 65535, 'CUDA master port differs')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', master_port))
    return dict(observed_epoch=time.time(), observed_monotonic=time.monotonic(), devices=devices,
                active_compute_processes=0, master_port_bindable=True, GPU_busy_measurement=None)
