"""Link both qualified Kimi score objects into the shared residual component."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

from read_training_slurm_membership import allocated_dcu, completed_numbered_step


ARMS = ('full-tree', 'wave32')
VISIBLE = ('kimi_output_residual_component_r461', 'kimi_output_residual_error_r461')
HIDDEN = 'kimi_output_residual_scores_v1_launch'
LABEL = 'Kimi461-output-residual-wave-component-link-CPU-v1'
MANIFEST = 'd2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6'
CAPTURE_MANIFEST = '3755a70a0603e32c38b36b909c3b28672f34ea3b0f3cb55edde601c024fb2967'
CAPTURE_LABEL = 'Kimi461-output-residual-wave-score-capture-link-CPU-v1'
CAPTURE_EXPORT = 'kimi_output_residual_scores_probe_r461'
CANDIDATES = Path('/refactor/candidates')
SCORE_OUTPUT = CANDIDATES/'Kimi461-output-residual-wave-score-gfx936-CPU-v1'
PARENT330 = CANDIDATES/'Kimi461-output-residual-tile64-CPU-v1'
PARENT335 = CANDIDATES/'Kimi461-output-residual-stage-diagnostic-CPU-v1'
PARENT987 = CANDIDATES/'Kimi-vocab-parallel-embedding-r461-probe-link-CPU-v2'
CPU330_SHA = '6d9421acefdb6091694ffe8429c3c87b90239e00eea9c95afb28a23844cb65e1'
CPU335_SHA = '8b40413bbf8a9ef6cef6fe3b79c1efd5ba0977d994e052cdc0866303b3b94f43'
CPU987_SHA = '982f780949434bbee15a336ea1e67e78d8482b77913a55b2f9987b959fe08217'
CUDA = '/opt/dtk/cuda/cuda/lib64/libcudart.so.12'
CUDA_SHA = 'ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e'
HOST_FILES = {'component.cpp', 'residual_method.inc', 'output_residual_tiles.hpp'}
REQUIRED_PROVIDERS = {'_infinilm.cpython-310-x86_64-linux-gnu.so', 'libinfinilm_runtime.so'}


def need(condition, why):
    if not condition:
        raise ValueError(why)


def digest(path):
    content = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            content.update(chunk)
    return content.hexdigest()


def checked_json(path, pin=None, size=4*1024**2):
    with path.open('rb') as stream:
        raw = stream.read(size+1)
    need(0 < len(raw) <= size and (pin is None or sha256(raw).hexdigest() == pin), 'metadata identity differs: '+path.name)
    return json.loads(raw)


def checked_sources(where, *, manifest=MANIFEST):
    pins = checked_json(where/'SOURCE_FILES.json', manifest)['files_SHA256']
    need(len(pins) == 13, 'expected the exact rendered score pair')
    for name, pin in pins.items():
        path = where/name
        need(path.is_relative_to(where) and path.resolve(strict=True) == path and digest(path) == pin,
             'rendered source bytes differ: '+name)
    need(all(pins[f'{ARMS[0]}/{name}'] == pins[f'{ARMS[1]}/{name}'] for name in HOST_FILES),
         'the two score arms must share the same HOST caller')
    return pins


def qualified_scores(receipt, accounting, *, output=SCORE_OUTPUT):
    need(receipt['completed'] is True and receipt['passed'] is True and receipt['manifest_SHA256'] == MANIFEST
         and receipt['all_fresh_M_before_compile'] is True and receipt['actual_device_TUs'] == 2
         and receipt['actual_GPU_steps'] == receipt['runtime_calls'] == 0 and set(receipt['arms']) == set(ARMS),
         'the actual score compile receipt is incomplete')
    step = receipt['original_step']
    row = completed_numbered_step(accounting['columns'], accounting['row'], step, scope='Kimi score build')
    resources = dict(pair.split('=', 1) for pair in row['AllocTRES'].split(','))
    epoch = receipt['epoch']
    elapsed = row['ElapsedRaw']
    need(type(epoch) in (int, float) and math.isfinite(epoch) and epoch > 0 and receipt['deadline'] == epoch+300
         and type(elapsed) is str and elapsed.isdecimal() and 0 <= int(elapsed) <= 300
         and step.split('.')[0] in {'1087656', '1103606'} and row['NodeList'] == receipt['node']
         and allocated_dcu(row['AllocTRES']) == 0 and resources.get('cpu') == '4'
         and resources.get('node') == '1' and resources.get('mem') in {'8G', '8192M'},
         'actual score build scope differs')
    objects = {}
    for arm in ARMS:
        record = receipt['arms'][arm]
        obj = output/arm/'fused_scores.o'
        need(record['object'] == str(obj) and obj.resolve(strict=True) == obj
             and digest(obj) == record['object_SHA256'] and record['exports'] == [HIDDEN]
             and digest(obj.parent/'fused_scores.before.d') == record['fresh_M_SHA256']
             and digest(obj.parent/'fused_scores.d') == record['actual_MD_SHA256'],
             'qualified score object differs: '+arm)
        objects[arm] = obj
    need(receipt['arms'][ARMS[0]]['object_SHA256'] != receipt['arms'][ARMS[1]]['object_SHA256'],
         'score object identity reused across arms')
    return objects


def dep_paths(path, target):
    left, colon, remainder = path.read_text().replace('\\\n', ' ').partition(':')
    deps = set(shlex.split(remainder))
    need(colon and shlex.split(left) == [str(target)] and deps and all(name.startswith('/') for name in deps),
         'fresh HOST dependency target differs')
    return deps


def compiler_pin(cpu330, cpu987):
    need(cpu987['actual_CPU'] == '910407.987' and cpu987['passed'] is True and cpu987['completed'] is True,
         'original compiler qualification differs')
    origin, = [cmd for cmd in cpu330['commands'] if '-c' in cmd]
    compiler = Path(origin[0]).resolve(strict=True)
    known = cpu987['compiler_current_container']
    need(str(compiler) == known['canonical'] and digest(compiler) == known['SHA256'],
         'the current HOST compiler differs from the original provider')


def component_build(source, output, pins, scores, cpu330, cpu335, run, *, capture=False):
    need(cpu330['actual_CPU'] == '1103606.330' and cpu335['actual_CPU'] == '1103606.335'
         and all(row['completed'] is True and row['passed'] is True for row in (cpu330, cpu335)),
         'original HOST compiler or closure producer differs')
    origin, = [cmd for cmd in cpu330['commands'] if '-c' in cmd]
    original_link, = [cmd for cmd in cpu330['commands'] if '-shared' in cmd]
    prefix = origin[:origin.index('-MD')]
    known = cpu335['dependency_SHA256']
    producer_files = {'component.cpp', 'residual_method.inc', 'output_residual_tiles.hpp', 'stage.hpp'}
    previous = {key for key in known if '/output-residual-stage-diagnostic-CPU-v1/' in key}
    need({Path(key).name for key in previous} == producer_files and len(previous) == len(producer_files),
         'the original stage-diagnostic HOST dependency inventory differs')
    host = source/ARMS[0]
    owned = {str(host/name): pins[f'{ARMS[0]}/{name}'] for name in HOST_FILES}
    expected = {name: pin for name, pin in known.items() if name not in previous}
    expected.update(owned)
    obj, before, after = (output/name for name in ('component.o', 'component.before.d', 'component.d'))
    path = str(host/'component.cpp')
    run('HOST-fresh-M', [*prefix, '-M', '-MF', str(before), '-MT', str(obj), path], 45)
    need(dep_paths(before, obj) == set(expected) and all(digest(name) == pin for name, pin in expected.items()),
         'fresh HOST dependencies differ from the original closure plus generated source')
    run('HOST-compile-once', [*prefix, '-MD', '-MF', str(after), '-MT', str(obj), '-c', path, '-o', str(obj)], 100)
    need(dep_paths(after, obj) == set(expected) and all(digest(name) == pin for name, pin in expected.items()),
         'compiled HOST dependencies differ from the fresh closure')
    host_sha = digest(obj)
    flags = [flag for flag in original_link if flag.startswith(('-L', '-l'))]
    known_providers = cpu330['actual_passive_providers']
    cuda = str(Path(CUDA).resolve(strict=True))
    need(digest(cuda) == CUDA_SHA, 'CUDA runtime provider differs')
    qualified = {**known_providers, cuda: CUDA_SHA}
    prior_providers = set(cpu335['passive_providers'])
    results = {}
    for arm, score in scores.items():
        library = output/(arm+'.so')
        run(arm+'-link', [prefix[0], '-shared', '-pthread', '-Wl,-Bsymbolic', '-Wl,--no-undefined', str(obj),
                         str(score), CUDA, *flags, '-o', str(library)], 45)
        symbols = run(arm+'-exports', ['/usr/bin/nm', '-D', '--defined-only', str(library)], 12)
        exported = {part[-1] for line in symbols.splitlines() if len(part := line.split()) >= 3 and part[-2] == 'T'}
        need(set(VISIBLE) <= exported and HIDDEN not in exported, 'paired component export or hidden score symbol differs')
        need((CAPTURE_EXPORT in exported) == capture, 'score capture export disagrees with the selected source')
        table = run(arm+'-symtab', ['/usr/bin/readelf', '-sW', str(library)], 12)
        need(any(len(part := line.split()) >= 8 and part[-1] == HIDDEN and part[6] != 'UND'
                 and (part[4] == 'LOCAL' or part[5] == 'HIDDEN') for line in table.splitlines()),
             'linked library did not bind its own score definition')
        loader = run(arm+'-passive', ['/usr/bin/ldd', '-r', str(library)], 25)
        need('not found' not in loader and 'undefined symbol' not in loader, 'unresolved component relocations')
        loaded = set()
        for line in loader.splitlines():
            parts = line.split()
            if parts:
                item = parts[2] if len(parts) > 2 and parts[1] == '=>' else parts[0]
                if item.startswith('/'):
                    loaded.add(str(Path(item).resolve(strict=True)))
        need(cuda in loaded and REQUIRED_PROVIDERS <= {Path(name).name for name in loaded}
             and prior_providers <= loaded <= qualified.keys()
             and all(digest(name) == qualified[name] for name in loaded), 'component provider identity differs')
        results[arm] = dict(path=str(library), SHA256=digest(library), providers=sorted(loaded), exports=sorted(exported))
    first, second = [results[name] for name in ARMS]
    need(first['providers'] == second['providers'] and first['exports'] == second['exports']
         and first['SHA256'] != second['SHA256'] and digest(obj) == host_sha, 'paired DSO provider or export scope differs')
    return dict(host_object_SHA256=host_sha, host_dependency_count=len(expected), arms=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-source', required=True, type=Path)
    parser.add_argument('--score-receipt-sha256', required=True)
    parser.add_argument('--score-accounting', required=True, type=Path)
    parser.add_argument('--score-capture', action='store_true', help='Link the separately rendered diagnostic HOST caller')
    args = parser.parse_args()
    job, step = (os.environ[key] for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID'))
    node = socket.gethostname().split('.')[0]
    need(job in {'1087656', '1103606'} and job == os.environ['TUNING_CPU_PARENT'] and step.isdecimal()
         and node == os.environ['TUNING_CPU_NODE'] and len(os.sched_getaffinity(0)) >= 4
         and all(os.environ.get(key) == '-1' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
         and not any(os.environ.get(key) for key in ('SLURM_STEP_GPUS', 'LD_PRELOAD', 'LD_AUDIT')),
         'expected the admitted zero-DCU CPU step')
    epoch, deadline = (float(os.environ[key]) for key in ('TUNING_CPU_EPOCH', 'TUNING_CPU_DEADLINE'))
    wall, mono = time.time(), time.monotonic()
    need(math.isfinite(epoch) and deadline == epoch+300 and epoch <= wall < deadline-12,
         'invalid original CPU link deadline')
    where = args.generated_source.resolve(strict=True)
    manifest = CAPTURE_MANIFEST if args.score_capture else MANIFEST
    pins = checked_sources(where, manifest=manifest)
    score = checked_json(SCORE_OUTPUT/'CPU-result.json', args.score_receipt_sha256)
    accounting = checked_json(args.score_accounting, size=65536)
    scores = qualified_scores(score, accounting)
    cpu330 = checked_json(PARENT330/'CPU-result.json', CPU330_SHA)
    cpu335 = checked_json(PARENT335/'CPU-result.json', CPU335_SHA)
    compiler_pin(cpu330, checked_json(PARENT987/'CPU-result.json', CPU987_SHA))
    output = CANDIDATES/(CAPTURE_LABEL if args.score_capture else LABEL)
    output.mkdir(mode=0o700)
    report = dict(step=f'{job}.{step}', node=node, epoch=epoch, deadline=deadline, passed=False,
                  score_compile_step=score['original_step'], score_receipt_SHA256=args.score_receipt_sha256,
                  score_compile_accounting={key: accounting[key] for key in ('columns', 'row')},
                  generated_manifest_SHA256=manifest, device_calls=0, weights_read=0, commands=[])

    def run(phase, command, limit):
        left = min(deadline-time.time(), (deadline-wall)-(time.monotonic()-mono))-8
        need(left > 1, 'original CPU linker budget expired')
        report['commands'].append(dict(phase=phase, argv=command))
        log = output/(phase+'.log')
        with log.open('xb') as stream:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=min(limit,left), check=True)
        if phase.endswith(('-exports', '-symtab', '-passive')):
            return log.read_text(encoding='utf-8', errors='replace')
        return ''

    try:
        report.update(component_build(where, output, pins, scores, cpu330, cpu335, run,
                                      capture=args.score_capture), passed=True)
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        report.update(error_type=type(error).__name__, reason=str(error)[:240])
    finally:
        with (output/'CPU-result.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(report, completed=True), stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('step', 'score_compile_step', 'passed')}, allow_nan=False))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
