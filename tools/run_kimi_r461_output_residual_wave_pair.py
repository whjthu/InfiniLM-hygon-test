"""Run the fixed Kimi residual experiment with both isolated libraries in mode 2."""

import argparse
import ctypes as C
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
from statistics import median
import subprocess
import sys
import time

from link_kimi_r461_output_residual_wave import LABEL as LINK_LABEL, MANIFEST, VISIBLE, digest
from read_training_slurm_membership import allocated_dcu, completed_numbered_step


REPO = Path(__file__).resolve().parents[1]
BASE = Path('code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-component-GPU-v2')
RESULTS = Path('/refactor/results')
INPUT = RESULTS/'Kimi461-output-residual-component-GPU-v1'
LIBS = Path('/refactor/candidates')/LINK_LABEL
ORDER = ('full-tree', 'wave32', 'wave32', 'full-tree', 'full-tree', 'wave32')
MODE = 2
PINS = {
    'run_arm.py': '14f0320fb481fa60b87bb06f535a8b12063875f4c6bd6374bd8a5bcede3f7e50',
    'compare_pair.py': 'aa22d04b06a62e614233948f9303b00c1f79c49701aba5b3bf67e0aaaba29e7d',
    'arm_watchdog.py': '97848eb1672cfdff42ccd0f2a67ab1258f6b5a1e3c672b569b059750bde5d2c8',
    'CHECKPOINT_WEIGHTS.json': '69d600435bcf35906c2d2f72452340467613e652868d1eb80bf8708a371ff23d',
    'CLAIM.json': '5bff0fc3ea0fd463b47b824ac20ba43a042c22446096b343d78c5b1f1768a23e',
}


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def pinned(path, pin, limit=256*1024):
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == pin, 'pinned file differs: '+path.name)
    return raw


def helper(name):
    scope = {'__name__': 'kimi_residual_published_helper', '__file__': str(REPO/BASE/name)}
    exec(compile(pinned(REPO/BASE/name, PINS[name]), scope['__file__'], 'exec'), scope)
    return scope


def trusted_inputs(claim, checkpoint, binding):
    sizes = dict(residual=66060288, norm=14336, projection=14336)
    need(set(binding['BF16_files']) == set(sizes), 'exactly three original BF16 operands required')
    need(all(binding[key] == checkpoint[key] for key in ('config_SHA256', 'index_SHA256', 'rms_norm_eps'))
         and binding['pre_transpose'] is claim['pre_transpose'] is False,
         'original checkpoint settings differ')
    need(all(binding['BF16_files'][name]['bytes'] == size for name, size in sizes.items())
         and all(all(binding['BF16_files'][name][field] == expected[field] for field in ('bytes', 'SHA256'))
                 for name, expected in checkpoint['bindings'].items())
         and binding['BF16_files']['residual']['SHA256'] == claim['residual_fixture']['SHA256'],
         'original residual fixture or checkpoint operand differs')
    return binding['BF16_files']


def load_binding():
    claim = json.loads(pinned(REPO/BASE/'CLAIM.json', PINS['CLAIM.json']))
    checkpoint = json.loads(pinned(REPO/BASE/'CHECKPOINT_WEIGHTS.json', PINS['CHECKPOINT_WEIGHTS.json']))
    binding = json.loads(pinned(INPUT/'inputs.json', claim['original_input_receipt_SHA256']))
    return binding, trusted_inputs(claim, checkpoint, binding), claim


def linked_libraries(receipt, accounting, folder=LIBS):
    need(receipt['completed'] is True and receipt['passed'] is True and receipt['generated_manifest_SHA256'] == MANIFEST
         and receipt['device_calls'] == receipt['weights_read'] == 0 and set(receipt['arms']) == set(ORDER),
         'both original linked arms must be qualified')
    row = completed_numbered_step(accounting['columns'], accounting['row'], receipt['step'], scope='Kimi paired link')
    res = dict(part.split('=', 1) for part in row['AllocTRES'].split(','))
    need(receipt['step'].split('.')[0] in {'1087656', '1103606'} and row['NodeList'] == receipt['node']
         and allocated_dcu(row['AllocTRES']) == 0 and res.get('node') == '1' and res.get('cpu') == '4'
         and res.get('mem') in {'8G', '8192M'} and row['ElapsedRaw'].isdecimal() and int(row['ElapsedRaw']) <= 300,
         'original component target link resources differ')
    a, b = (receipt['arms'][name] for name in ('full-tree', 'wave32'))
    need(a['SHA256'] != b['SHA256'] and a['providers'] == b['providers'] and a['exports'] == b['exports'] and
         set(VISIBLE) <= set(a['exports']), 'paired DSO identities, providers or caller exports differ')
    for name, record in receipt['arms'].items():
        path = folder/(name+'.so')
        need(record['path'] == str(path) and path.resolve(strict=True) == path and digest(path) == record['SHA256'],
             'linked arm artifact differs: '+name)
    return receipt['arms']


def write(path, document):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def arm(name, library_sha, output, *, inputs=load_binding, load=C.CDLL):
    need(sys.byteorder == 'little', 'little-endian BF16 ABI required')
    binding, files, _ = inputs()
    started = time.monotonic_ns()
    native = helper('run_arm.py')
    operands = [native['load_exact'](INPUT/'inputs', files[key], key) for key in native['WORDS']]
    library = (LIBS/(name+'.so')).resolve(strict=True)
    need(digest(library) == library_sha, 'isolated arm library bytes differ')
    loading = time.monotonic_ns()
    handle = load(str(library), mode=C.RTLD_LOCAL)
    ready = time.monotonic_ns()
    startup = dict(PID=os.getpid(), ready_monotonic_ns=ready, inputs_and_hash_seconds=(loading-started)*1e-9,
                   dynamic_load_seconds=(ready-loading)*1e-9)
    write(output/'startup.tmp', startup)
    (output/'startup.tmp').rename(output/'startup.json')
    values, metrics, memory = native['native_arm'](handle, MODE, int(binding['pre_transpose']),
                                                    binding['rms_norm_eps'], operands)
    raw = bytes(values)
    with (output/'output.bf16').open('xb') as stream:
        stream.write(raw)
    return dict(complete=True, arm=name, mode=MODE, original_input_receipt_SHA256=load_original_pin(),
                library_SHA256=library_sha, output_SHA256=sha256(raw).hexdigest(), output_bytes=len(raw),
                startup=startup, finished_monotonic_ns=time.monotonic_ns(), driver_samples=memory,
                seconds=dict(dispatch_through_sync=metrics[0], post_dispatch_audit=metrics[1],
                             full_CABI_before_destroy=metrics[2]))


def load_original_pin():
    return json.loads(pinned(REPO/BASE/'CLAIM.json', PINS['CLAIM.json']))['original_input_receipt_SHA256']


def children(output, libraries, monotonic_deadline, *, started_arm=None, compare=None):
    launch = start_arm if started_arm is None else started_arm
    compare = helper('compare_pair.py')['compare'] if compare is None else compare
    result = dict(complete=False, planned_arms=list(ORDER), planned_pairs=3, mode=MODE, records=[], pairs=[],
                  model_qualified=False, performance_qualified=False)
    try:
        for index, name in enumerate(ORDER):
            need(time.monotonic() < monotonic_deadline-10, 'original GPU budget ended before next arm')
            directory = output/f'arm-{index}'
            directory.mkdir(mode=0o700)
            library_sha = libraries[name]['SHA256']
            rec = dict(ordinal=index, arm=name, complete=False, library_SHA256=library_sha)
            result['records'].append(rec)
            report = launch(directory, name, library_sha, monotonic_deadline)
            need(report['complete'] is True and report['arm'] == name and report['mode'] == MODE
                 and report['library_SHA256'] == library_sha and report['original_input_receipt_SHA256'] == load_original_pin(),
                 'arm result does not bind its original library and input')
            rec.update(report)
            if index % 2:
                pair = result['records'][-2:]
                operands = {}
                for item in pair:
                    path = output/f"arm-{item['ordinal']}"/'output.bf16'
                    operands[item['arm']] = pinned(path, item['output_SHA256'], 2*512*7168)
                    need(len(operands[item['arm']]) == item['output_bytes'], 'saved residual size differs')
                trial = compare(operands['full-tree'], operands['wave32'])
                timing = {item['arm']: item['seconds']['dispatch_through_sync'] for item in pair}
                need(all(type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0
                         for seconds in timing.values()), 'invalid paired dispatch timing')
                trial.update(full_tree_ordinal=next(item['ordinal'] for item in pair if item['arm'] == 'full-tree'),
                             wave32_ordinal=next(item['ordinal'] for item in pair if item['arm'] == 'wave32'),
                             dispatch_through_sync_seconds=timing,
                             dispatch_wave32_over_full_tree=timing['wave32']/timing['full-tree'] if timing['full-tree'] else None)
                result['pairs'].append(trial)
        result['complete'] = True
        result['fixed_input_BF16_output_exact'] = all(item['changed_words'] == 0 for item in result['pairs'])
        ratios = [item['dispatch_wave32_over_full_tree'] for item in result['pairs']]
        result['median_dispatch_wave32_over_full_tree'] = median(ratios) if None not in ratios else None
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result['failure'] = dict(kind=type(error).__name__, detail=str(error)[:160])
    finally:
        result['missing_arms'] = len(ORDER)-sum(item['complete'] for item in result['records'])
        result['missing_pairs'] = 3-len(result['pairs'])
        write(output/'worker-result.json', result)
    return result


def start_arm(output, name, library_sha, deadline):
    started = time.monotonic()
    with (output/'worker.log').open('xb') as stream:
        child = subprocess.Popen([sys.executable, '-B', __file__, 'arm', '--name', name, '--library-sha', library_sha,
                                  '--output', str(output)], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        try:
            rc, startup = helper('arm_watchdog.py')['wait'](child, output/'startup.json', started, deadline-3, 65, 40)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)
    need(rc == 0 and startup is not None, 'isolated GPU arm did not complete')
    report = json.loads((output/'result.json').read_bytes())
    need(report['startup'] == startup and startup['ready_monotonic_ns'] <= report['finished_monotonic_ns']
         <= startup['ready_monotonic_ns']+40_000_000_000 and report['finished_monotonic_ns']*1e-9 < deadline,
         'isolated GPU arm exceeded its startup/native deadline')
    return report


def physical_gpu(preflight):
    job, step = (os.environ[key] for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID'))
    device = os.environ['SLURM_STEP_GPUS']
    node = socket.gethostname().split('.')[0]
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    need(job in {'1087656', '1103606'} and step.isdecimal() and re.fullmatch(job+r'\.\d+', preflight['actual_CPU'])
         and node == preflight['node'] and preflight['KFD_holders'] == preflight['matching_owned_PIDs'] == []
         and preflight['boot_id'] == boot and len(preflight['PCI']) == len(set(preflight['PCI'])) == 8
         and all(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', card) for card in preflight['PCI'])
         and re.fullmatch('[0-7]', device) and os.environ['HIP_VISIBLE_DEVICES'] == device,
         'GPU worker does not match the caller placement and assigned mask')
    cuda = C.CDLL('libcudart.so.12')
    count, pci = C.c_int(), C.create_string_buffer(32)
    need(cuda.cudaGetDeviceCount(C.byref(count)) == 0 and count.value == 1
         and cuda.cudaDeviceGetPCIBusId(pci, len(pci), 0) == 0, 'physical assignment could not be resolved')
    address = pci.value.decode('ascii').lower()
    need(address in preflight['PCI'], 'allocated physical card did not appear in the preflight')
    return dict(step=f'{job}.{step}', node=node, boot_id=boot, PCI=address, physical_id=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='kind', required=True)
    one = modes.add_parser('arm')
    one.add_argument('--name', required=True, choices=sorted(set(ORDER)))
    one.add_argument('--library-sha', required=True)
    one.add_argument('--output', required=True, type=Path)
    multi = modes.add_parser('pair')
    for name in ('link-receipt', 'link-accounting', 'preflight', 'output'):
        multi.add_argument('--'+name, required=True, type=Path)
    for name in ('link-receipt-sha', 'link-accounting-sha', 'preflight-sha'):
        multi.add_argument('--'+name, required=True)
    multi.add_argument('--epoch', required=True, type=float)
    args = parser.parse_args()
    if args.kind == 'arm':
        original = json.loads((args.output.parent/'worker-identity.json').read_bytes())
        step = os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        need(step == original['step'] and os.environ['SLURM_STEP_GPUS'] == original['physical_id']
             and args.library_sha == original['library_SHA256'][args.name]
             and re.fullmatch(r'arm-[0-5]', args.output.name) and ORDER[int(args.output.name[-1])] == args.name,
             'isolated arm differs from the admitted pair and ordinals')
        write(args.output/'result.json', arm(args.name, args.library_sha, args.output))
        return 0
    wall, mono = time.time(), time.monotonic()
    need(math.isfinite(args.epoch) and 0 < args.epoch <= wall < args.epoch+690, 'invalid original GPU epoch')
    deadline = mono+args.epoch+700-wall
    linked = json.loads(pinned(args.link_receipt, args.link_receipt_sha))
    libraries = linked_libraries(linked, json.loads(pinned(args.link_accounting, args.link_accounting_sha)))
    claim = json.loads(pinned(REPO/BASE/'CLAIM.json', PINS['CLAIM.json']))
    need(all(digest(Path(path)) == pin for path, pin in claim['provider_SHA256'].items()), 'original active runtime provider differs')
    identity = physical_gpu(json.loads(pinned(args.preflight, args.preflight_sha)))
    args.output.mkdir(mode=0o700)
    write(args.output/'worker-identity.json', dict(identity, link_receipt_SHA256=args.link_receipt_sha,
               link_accounting_SHA256=args.link_accounting_sha, preflight_SHA256=args.preflight_sha, epoch=args.epoch,
               library_SHA256={name: item['SHA256'] for name, item in libraries.items()}))
    result = children(args.output, libraries, deadline)
    print(json.dumps({key: result[key] for key in ('complete', 'missing_arms', 'missing_pairs')}, allow_nan=False))
    return int(not result['complete'])


if __name__ == '__main__':
    raise SystemExit(main())
