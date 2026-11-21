"""Link one isolated GLM I128 component provider per verified original object."""

import argparse
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
PACKAGES = REPO / 'code/bw1000/GLM-C1-layer-observer-r474'
PARENT, NODE = '1103606', 'f13r1n08'
LABEL = 'GLM-I128-five-isolated-providers-CPU-v1'
OUTPUT = Path('/refactor/candidates') / LABEL
ROLES = {'gate': {1: 'G1', 2: 'G2'}, 'down': {2: 'D2', 4: 'D4'}}
AUDIT_SHA = '5db40da457f7b1449b30764bb9c224a5cd4f50824555b2fa866294aedca5df46'
PROVIDER_PLAN_SHA = '7bb0a4ebc72612b00e4498ac0542dd258dacb853c1d151aad311637bd2365fca'
RUNTIME = '/opt/dtk/cuda/cuda/lib64/libcudart.so.12'
HIP = '/opt/dtk/lib/libgalaxyhip.so.5'
EXPORTS = {'glm_r470_i128_validate', 'glm_r470_i128_gate', 'glm_r470_i128_down'}
HIP_EXPORTS = {'hip' + name for name in (
    'GetDeviceCount', 'GetDevice', 'SetDevice', 'DeviceGetPCIBusId', 'RuntimeGetVersion', 'DriverGetVersion',
    'MemGetInfo', 'StreamCreateWithFlags', 'StreamSynchronize', 'StreamDestroy', 'EventCreateWithFlags',
    'EventDestroy', 'EventRecord', 'EventSynchronize', 'EventElapsedTime', 'Malloc', 'Free', 'Memcpy', 'MemsetAsync',
)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def check_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def read_pin(path, digest, limit=4 * 1024**2):
    need(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path.absolute(),
         'expected a canonical original file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest, 'original file pin or byte bound differs')
    return raw


def stream_pin(path, expected=None):
    start = path.stat()
    current = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024**2), b''):
            current.update(block)
    end = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    need(all(getattr(start, key) == getattr(end, key) for key in fields), 'linked or runtime object changed')
    identity = current.hexdigest()
    need(expected is None or expected == identity, 'linked or runtime object SHA differs')
    return dict(path=str(path), sha256=identity, bytes=start.st_size)


def write(path, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def adopted_objects(audits, auditor):
    needed = {role for role in ('ORIG', 'G1', 'G2', 'D2', 'D4')}
    objects = {}
    steps, requests = set(), set()
    for arm, expected in ROLES.items():
        plan, variants, manifest, _ = auditor.load(arm)
        record = audits[arm]
        row = record['accounting']
        request = record['original_request']
        need(record['arm'] == arm and record['passed'] is True
             and record['outcome'] == 'TWO_TARGET_COMPILER_OBJECTS_VERIFIED_WITHOUT_GPU_EXECUTION'
             and record['device_numerical_qualified'] is False
             and len(row) == 7 and row[1:4] == [plan['label'], 'COMPLETED', '0:0'] and row[5] == plan['node']
             and auditor.accounting('|'.join(row).encode(), plan) == row
             and re.fullmatch('[0-9a-f]{32}', request) and re.fullmatch('[0-9a-f]{40}', record['publication'])
             and all(check_sha(record[key]) for key in ('launch_SHA256', 'build_SHA256')),
             'original CPU compile audit did not pass')
        need(row[0] not in steps and request not in requests, 'same original job used for different compile branches')
        steps.add(row[0])
        requests.add(request)
        entries = record['verified_objects']
        need(len(entries) == len(expected) and {item['rows'] for item in entries} == set(variants) == expected.keys(),
             'original compile audit is incomplete')
        baseline = dict(object=plan['original_object'], object_SHA256=plan['original_object_SHA256'])
        need(objects.setdefault('ORIG', baseline) == baseline, 'original baseline objects differ')
        for item in entries:
            size = item['rows']
            pinned = plan['variants'][str(size)]
            object_path = f'/refactor/candidates/{plan["label"]}/rows{size}/routed_i128.o'
            need(item['object'] == object_path and item['candidate_id'] == pinned['id']
                 and item['source_SHA256'] == pinned['source_SHA256'] and check_sha(item['object_SHA256'])
                 and type(item['compiled_bytes']) is int and 0 < item['compiled_bytes'] <= 128*1024**2,
                 'proved CPU object is outside its frozen variant')
            objects[expected[size]] = dict(object=object_path, object_SHA256=item['object_SHA256'],
                                         compiled_bytes=item['compiled_bytes'], CPU_step=row[0], CPU_request=request,
                                         CPU_plan_manifest_SHA256=manifest, candidate_id=pinned['id'], rows=size)
    need(set(objects) == needed and len({item['object_SHA256'] for item in objects.values()}) == len(needed),
         'expected five distinct verified objects')
    return objects


def shared_command(object_path, library, script):
    return ['/usr/bin/g++', '-shared', '-Wl,-z,defs,-Bsymbolic', '-Wl,--version-script=' + str(script),
            '-Wl,-soname,' + library.name, '-o', str(library), object_path, RUNTIME]


def dynamic_names(text):
    return {parts[0].split('@')[0] for row in text.splitlines() if len(parts := row.split()) >= 2}


def bound_runtime_graph(ldd, plan):
    graph = {}
    for raw in ldd.splitlines():
        line = raw.strip()
        if not line or line.startswith('linux-vdso.so.'):
            continue
        match = re.fullmatch(r'(?:\S+ => )?(/\S+) \(0x[0-9a-f]+\)', line)
        need(match and match[1] in plan, 'unrecognized or unresolved dependency in passive loader graph')
        graph[match[1]] = plan[match[1]]
    need(RUNTIME in graph, 'linked provider does not bind the DTK CUDA runtime')
    return graph


def load_auditor():
    auditor_path = PACKAGES / 'routed-i128-cpu-artifact-audit-v1/audit_once.py'
    code = read_pin(auditor_path, AUDIT_SHA)
    spec = importlib.util.spec_from_file_location('i128_original_artifacts', auditor_path)
    audit = importlib.util.module_from_spec(spec)
    exec(compile(code, str(auditor_path), 'exec'), audit.__dict__)
    return audit


def original_tools():
    audit = load_auditor()
    plan = audit.load('gate')[0]
    native_plan = PACKAGES / 'attention-detail-CPU-v1/BUILD_PLAN.json'
    libraries = json.loads(read_pin(native_plan, PROVIDER_PLAN_SHA))['providers']
    need(libraries[RUNTIME] == 'ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e'
         and libraries[HIP] == 'b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b',
         'original GLM runtime SHA differs')
    for tool in ('/usr/bin/g++', '/usr/bin/ld', '/usr/bin/nm', '/usr/bin/readelf', '/usr/lib/gcc/x86_64-linux-gnu/11/collect2'):
        expected = plan['tools'][tool]
        reported = Path(tool).resolve(strict=True)
        need(str(reported) == expected['canonical'], 'original link tool path differs')
        stream_pin(reported, expected['sha256'])
    return audit, libraries


def caller_binding(report, accounting, pci, epoch, deadline, auditor):
    scope = dict(label=LABEL, parent=PARENT, node=NODE)
    row = auditor.accounting(accounting, scope)
    need(report['state'] == 'ISOLATED_HOST_LINK_PASS' and report['request_scope'] == LABEL
         and report['actual_CPU_step'] == row[0] and row[2:4] == ['COMPLETED', '0:0']
         and report['deadline'] == report['epoch'] + 300 and report['actual_target_GPU_steps'] == 0
         and report['GPU_GO'] is False, 'missing independent original CPU link terminal')
    need(re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]', pci)
         and math.isfinite(epoch) and math.isfinite(deadline) and 0 < deadline - epoch <= 300,
         'invalid original GPU binding scope or deadline')
    providers = report['verified_roles']
    need(set(providers) == {'ORIG', 'G1', 'G2', 'D2', 'D4'} and len({r['sha256'] for r in providers.values()}) == 5,
         'unverified or duplicate component providers')
    expected = {}
    for role, member in providers.items():
        path = OUTPUT / (role + '.so')
        need(member['path'] == str(path) and check_sha(member['sha256']), 'unexpected isolated provider path')
        need(stream_pin(path, member['sha256'])['bytes'] == member['bytes'], 'isolated provider differs after original link')
        expected[role] = {key: member[key] for key in ('path', 'sha256')}
    hip = report['hip']
    path = Path(HIP).resolve(strict=True)
    need(str(path) == hip['path'] and check_sha(hip['sha256'])
         and stream_pin(path, hip['sha256'])['bytes'] == hip['bytes'], 'HIP provider pin differs after link')
    return dict(schema='glm_i128_one_gpu_v1', hip={key: hip[key] for key in ('path', 'sha256')},
                providers=expected, expected_pci=pci, epoch=epoch, deadline=deadline)


def produce(audits, epoch, deadline):
    audit, libraries = original_tools()
    objects = adopted_objects(audits, audit)
    for item in objects.values():
        path = Path(item['object'])
        raw = read_pin(path, item['object_SHA256'], 128 * 1024**2)
        audit.host_elf(raw)
        need('compiled_bytes' not in item or len(raw) == item['compiled_bytes'], 'GPU object byte identity differs')
    stream_pin(Path(RUNTIME), libraries[RUNTIME])
    hip = stream_pin(Path(HIP).resolve(strict=True), libraries[HIP])
    step = os.environ['SLURM_STEP_ID']
    need(os.environ.get('SLURM_JOB_ID') == PARENT and step.isdecimal() and socket.gethostname().split('.')[0] == NODE
         and os.environ.get('SLURM_CPUS_PER_TASK') == '4' and len(os.sched_getaffinity(0)) >= 4
         and os.environ.get('SLURM_STEP_GPUS') in (None, '', '(null)')
         and all(os.environ.get(key) == '-1' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')),
         'original isolated CPU step identity differs')
    OUTPUT.mkdir(mode=0o700)
    script = OUTPUT / 'only-i128.map'
    write(script, ('{ global: ' + '; '.join(sorted(EXPORTS)) + '; local: *; };\n').encode())
    commands, providers = [], {}
    def run(args):
        left = min(24, deadline - time.time() - 8)
        need(left > 1, 'original component link deadline exhausted')
        result = subprocess.run(args, capture_output=True, timeout=left, check=False, text=True)
        commands.append(dict(argv=args, rc=result.returncode))
        need(result.returncode == 0, 'component link or passive verification failed')
        return result.stdout
    try:
        hip_names = dynamic_names(run(['/usr/bin/nm', '-D', '--defined-only', '--format=posix', hip['path']]))
        need(HIP_EXPORTS <= hip_names, 'original pinned HIP provider misses its API')
        verified_libraries = {RUNTIME: libraries[RUNTIME], hip['path']: hip['sha256']}
        for role in ('ORIG', 'G1', 'G2', 'D2', 'D4'):
            item = objects[role]
            library = OUTPUT / (role + '.so')
            run(shared_command(item['object'], library, script))
            names = dynamic_names(run(['/usr/bin/nm', '-D', '--defined-only', '--format=posix', str(library)]))
            need(names == EXPORTS, 'DSO does not expose exactly the three original wrapper names')
            header = run(['/usr/bin/readelf', '-h', str(library)])
            need(re.search(r'Type:\s+DYN\b', header) and 'Advanced Micro Devices X86-64' in header, 'expected a target host DSO')
            graph = bound_runtime_graph(run(['/usr/bin/ldd', '-r', str(library)]), libraries)
            for name, digest in graph.items():
                if name not in verified_libraries:
                    stream_pin(Path(name), digest)
                    verified_libraries[name] = digest
            providers[role] = stream_pin(library)
        need(len({item['sha256'] for item in providers.values()}) == 5, 'provider files are not distinct')
    finally:
        complete = len(providers) == 5 and len({item['sha256'] for item in providers.values()}) == 5
        report = dict(state='ISOLATED_HOST_LINK_PASS' if complete else 'ORIGINAL_LINK_INCOMPLETE', request_scope=LABEL,
                      actual_CPU_step=PARENT + '.' + step, epoch=epoch, deadline=deadline, verified_roles=providers,
                      input_objects=objects, hip=hip, commands=commands, actual_target_GPU_steps=0,
                      actual_explicit_SDK_calls=0, GPU_GO=False, device_numerical_qualified=False)
        write(OUTPUT / 'CPU-link-result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    linking = commands.add_parser('link', help='link from two original passing compiler audits on the fixed CPU step')
    for arm in ROLES:
        linking.add_argument('--' + arm + '-audit', type=Path, required=True)
        linking.add_argument('--' + arm + '-sha256', required=True)
    binding = commands.add_parser('bind', help='bind passing original CPU-link terminal and a fresh allocated PCI')
    binding.add_argument('--cpu-link-report-sha256', required=True)
    binding.add_argument('--cpu-accounting', type=Path, required=True)
    binding.add_argument('--cpu-accounting-sha256', required=True)
    binding.add_argument('--expected-pci', required=True)
    binding.add_argument('--output', type=Path, required=True)
    for command in (linking, binding):
        command.add_argument('--epoch', type=float, required=True)
        command.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    need(math.isfinite(args.epoch) and args.epoch <= time.time() < args.deadline
         and 0 < args.deadline - args.epoch <= 300, 'original epoch or deadline differs')
    def timeout(*_):
        raise TimeoutError('original CPU link deadline expired')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.deadline - time.time())
    try:
        if args.mode == 'bind':
            report = json.loads(read_pin(OUTPUT / 'CPU-link-result.json', args.cpu_link_report_sha256))
            accounting = read_pin(args.cpu_accounting.absolute(), args.cpu_accounting_sha256)
            result = caller_binding(report, accounting, args.expected_pci, args.epoch, args.deadline, load_auditor())
            output = args.output.absolute()
            need(output.name == 'binding.json' and output.parent.resolve(strict=True) == output.parent
                 and output.parent.is_relative_to(Path('/refactor/results')), 'unexpected GPU binding namespace')
            print(json.dumps(dict(binding_SHA256=write(output, result), actual_target_GPU_steps=0)), flush=True)
            return
        need(args.deadline == args.epoch + 300, 'original CPU link epoch differs')
        audits = {arm: json.loads(read_pin(getattr(args, arm + '_audit').absolute(), getattr(args, arm + '_sha256')))
                  for arm in ROLES}
        result = produce(audits, args.epoch, args.deadline)
        print(json.dumps(dict(state=result['state'], actual_CPU_step=result['actual_CPU_step'],
                              verified_roles=sorted(result['verified_roles']), GPU_GO=False)), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
