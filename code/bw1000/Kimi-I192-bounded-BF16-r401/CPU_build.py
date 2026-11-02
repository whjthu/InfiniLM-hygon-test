"""Compile one private helper HOST plus I192 static decoder; no dlopen/device."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import traceback

D = Path(__file__).resolve().parent
B = Path('/refactor/candidates/Kimi-I192-bounded-BF16-r401')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
start = time.monotonic()
q = dict(completed=False, passed=False, actual_CPU=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
         CPU_only=True, GPU_initialized=False, Native_dlopen=False, weights_opened=False,
         existing_Native_or_SDK_rebuilt=False, commands=[], compiled=[])


def run(argv, label, limit=150):
    left = 250 - (time.monotonic() - start)
    assert left > 0, 'Original CPU budget'
    z = subprocess.run([str(x) for x in argv], text=True, capture_output=True, timeout=min(left, limit))
    log = B / (label + '.log'); log.write_text(z.stdout + z.stderr)
    q['commands'].append(dict(label=label, argv=[str(v) for v in argv], rc=z.returncode, log=str(log), SHA256=sha(log)))
    if z.returncode: raise RuntimeError(label + ': ' + z.stderr[-5000:])
    return z.stdout


def deps(p):
    return set(shlex.split(p.read_text().replace('\\\n', ' ').split(':', 1)[1]))


if B.exists(): raise FileExistsError('Immutable CPU output exists; do not replay')
try:
    B.mkdir()
    assert all(os.environ.get(n) == '-1' for n in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'])
    mf = D / 'SOURCE_MANIFEST.json'
    assert sha(mf) == os.environ['KIMI401_SOURCE_SHA256']
    m = json.loads(mf.read_bytes())
    assert all(sha(D / p) == h for p, h in m['files_SHA256'].items())
    c = json.loads((D / 'BUILD_CONTRACT.json').read_bytes())
    for p, h in {**c['dependencies_SHA256'], **c['reused_GPU_objects_SHA256']}.items(): assert sha(p) == h, p
    prior = c['prior_CPU391']; assert sha(prior['path']) == prior['SHA256']
    old = json.loads(Path(prior['path']).read_bytes()); assert old['completed'] and old['passed']
    q['recording_consumer'] = json.loads(run(['python3', '-B', str(D / 'check_CPU.py')], 'actual-consumer-recording', 20))
    assert q['recording_consumer']['passed']
    # Both original and new private units include exactly the pinned existing MXFP4 decoder header.
    assert sha(c['math_header']['path']) == c['math_header']['SHA256']
    for name, prefix in [('decoder.cu', c['GPU_prefix']), ('helper.cpp', c['HOST_prefix'])]:
        obj = B / (name + '.o'); md = B / (name + '.d'); before = B / (name + '.M')
        prefix = prefix + ['-I' + str(D)]
        run(prefix + ['-M', '-MF', str(before), '-MT', str(obj), str(D / name)], name + '-M', 40)
        pin = {p: sha(p) for p in deps(before)}
        run(prefix + ['-MD', '-MF', str(md), '-c', str(D / name), '-o', str(obj)], name + '-compile', 150)
        assert deps(md) == set(pin) and all(sha(p) == h for p, h in pin.items())
        if name.endswith('.cu'):
            assert pin[c['math_header']['path']] == c['math_header']['SHA256']
        else:
            checked = 0
            for p, h in pin.items():
                if p in old['dependency_SHA256']:
                    assert old['dependency_SHA256'][p] == h, p
                    checked += 1
            assert checked > 100
            q['successful_current391_shared_header_bytes_checked'] = checked
        q['compiled'].append(dict(source=str(D / name), source_SHA256=sha(D / name), object=str(obj),
                                  object_SHA256=sha(obj), MD=str(md), MD_SHA256=sha(md),
                                  dependencies_SHA256=pin, M_equals_MD=True))
    library = B / 'libKimi_I192_bounded_BF16_r401.so'
    run(c['link'] + [v['object'] for v in q['compiled']] + list(c['reused_GPU_objects_SHA256']) + c['link_libraries'] + ['-o', str(library)], 'link', 40)
    dynamic = run(['readelf', '--wide', '-d', str(library)], 'dynamic', 15)
    assert 'libKimi_append' not in dynamic and 'libkimi_grouped_prefill' not in dynamic
    symbols = run(['nm', '-D', '--defined-only', str(library)], 'exports', 15)
    required = ['kimi401_grouped_mxfp4', 'kimi401_grouped_error', 'kimi401_set_operand_mode', 'kimi401_grouped_shutdown',
                'kimi401_decoder_counts', 'kimi401_grouped_metadata_counts_r5858', 'kimi401_bounded_decode_bf16_r6054']
    exports = {line.split()[-1] for line in symbols.splitlines() if line.split()}
    assert set(required) <= exports
    ldd = run(['ldd', '-r', str(library)], 'ldd-r', 30)
    assert 'not found' not in ldd and 'undefined symbol' not in ldd
    paths = {}
    for line in ldd.splitlines():
        match = re.match(r'^\s*(?:\S+\s+=>\s+)?(/\S+)\s+\(', line)
        if match: paths[match.group(1)] = sha(match.group(1))
    for p, h in c['dependencies_SHA256'].items():
        if p in paths: assert paths[p] == h
    sdk = '/refactor/install/hygon-kimi-5f91ad7-r3672/lib/libinfiniops.so'
    assert paths[sdk] == c['dependencies_SHA256'][sdk]
    q.update(completed=True, passed=True, source_manifest_SHA256=sha(mf), library=str(library), library_SHA256=sha(library),
             HOST_TUs_compiled=1, GPU_TUs_compiled=1, reused_GPU_objects_SHA256=c['reused_GPU_objects_SHA256'],
             actual_dynamic_dependencies_SHA256=paths, required_CAPI=required,
             qualification='Private2TU compile/link/fullMD/currentRuntimeheaders+SDK+reusedobjects/exports/ldd-r and actual consumer recording only. No device Runtime, checkpoint, numerical or performance qualification.', GPU_GO=False)
except BaseException as error:
    q.update(error=repr(error), traceback=traceback.format_exc())
finally:
    q['elapsed_seconds'] = time.monotonic() - start
    q['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if B.exists(): (B / 'CPU-build.json').write_text(json.dumps(q, indent=2) + '\n')
    print(json.dumps({k: q.get(k) for k in ['completed', 'passed', 'actual_CPU', 'library_SHA256', 'elapsed_seconds', 'error']}))
if not q['passed']: raise SystemExit(1)
