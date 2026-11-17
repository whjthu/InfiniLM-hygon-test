"""One CPU qualification; imported/executed only inside the separately approved step."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import time

D = Path(__file__).resolve().parent
W = D.parent
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
J = read(W / 'intent.json')
P = read(D / 'BUILD_PLAN.json')
anchor = time.monotonic()
# This is the unspent original epoch, never a fresh work allowance.
budget = min(J['work_deadline'], J['deadline']) - time.time()
step = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
receipt = dict(step=step, node=socket.gethostname().split('.')[0], passed=False,
               new_HOST_TUs=0, new_GPU_TUs=0, GPU_cards=0, commands=[])


def remaining():
    value = min(budget - (time.monotonic() - anchor), J['work_deadline'] - time.time(),
                J['deadline'] - time.time())
    if not math.isfinite(value) or value <= 2:
        raise TimeoutError('original CPU work epoch exhausted')
    return value


def command(label, argv, limit=50):
    run = subprocess.run(argv, capture_output=True, text=True, timeout=min(limit, remaining() - 1))
    for stream in ('stdout', 'stderr'):
        (W / (label + '.' + stream)).write_text(getattr(run, stream), encoding='utf-8')
    receipt['commands'].append(dict(name=label, argv=argv, rc=run.returncode,
        stdout_SHA256=H(W / (label + '.stdout')), stderr_SHA256=H(W / (label + '.stderr'))))
    if run.returncode:
        raise RuntimeError(label + ' returned ' + str(run.returncode))
    return run.stdout, run.stderr


def dependencies(path):
    _, separator, body = path.read_text().replace('\\\n', ' ').partition(':')
    names = set(shlex.split(body))
    assert separator and names and all(n.startswith('/') for n in names)
    return names


def check_pins(pins):
    for path, pin in pins.items():
        remaining()
        assert H(path) == pin, path


def symbols(path, label):
    out, _ = command(label, ['/usr/bin/nm', '-D', '--defined-only', str(path)])
    return {x[-1]: x[-2] for line in out.splitlines()
            if len(x := line.split()) >= 3 and x[-2] in 'TDBR'}


def elf(path, label):
    out, _ = command(label, ['/usr/bin/readelf', '-d', str(path)])
    assert 'SYMBOLIC' in out
    return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]', out)


def providers(path, label):
    out, err = command(label, ['/usr/bin/ldd', '-r', str(path)], 30)
    text = out + '\n' + err
    assert not any(x in text for x in ('not found', 'undefined symbol'))
    paths = set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)', text, re.M))
    return {p: H(p) for p in sorted(paths)}


def verify_tls(label, path, names):
    table, _ = command(label + '-tls', ['/usr/bin/readelf', '-Ws', '--demangle', str(path)])
    for name in names:
        rows = [row.split() for row in table.splitlines() if row.rstrip().endswith(' ' + name)]
        locations = {(row[1], row[6]) for row in rows if row[3] == 'TLS' and row[6] != 'UND'}
        assert len(locations) == 1, (label, name)


def main():
    assert str(D) == '/refactor/results/GLM474-attention-detail-CPU-v1/src'
    assert receipt['node'] == J['node'] == 'f13r1n05' and re.fullmatch(r'1103606\.[0-9]+', step)
    assert J['epoch'] < J['work_deadline'] <= J['epoch'] + 430 <= J['deadline'] == J['epoch'] + 600
    assert all(os.environ.get(k) == '-1' for k in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'))
    assert not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT')
    from verify_package import verify
    verify(D)
    assert H(D / 'CLAIM.json') == J['claim_SHA256']
    tools = read(D / 'TOOLCHAIN.json')
    for path, pin in tools['tools'].items():
        assert str(Path(path).resolve()) == pin['canonical'] and H(path) == pin['sha256']
    for key, value in tools['required_environment'].items():
        assert os.environ.get(key) == value
    check_pins({P['parent_library']: P['parent_library_SHA256'],
                P['parent_build']: P['parent_build_SHA256'],
                P['parent_MD']: P['parent_MD_SHA256'],
                P['parent_outcome']: P['parent_outcome_SHA256']})
    parent = read(P['parent_build'])
    outcome = read(P['parent_outcome'])
    assert parent['passed'] and parent['step'] == '1103606.143' and outcome['passed']
    assert parent['library_SHA256'] == P['parent_library_SHA256'] and parent['linked_objects'] == 32
    assert parent['providers'] == 71 and parent['strong_exports_preserved'] == 280
    check_pins(P['parent_objects'])
    assert dependencies(Path(P['parent_MD'])) == set(read(D / 'parents/nov-dependencies.json'))
    selected = {}
    # All three fresh-M complete before the first compiler invocation.
    for unit in P['units']:
        label = unit['name']
        depfile = W / (label + '.before.d')
        command(label + '-M', unit['compiler_prefix'] + ['-M', '-MF', str(depfile), '-MT', unit['object'], unit['source']])
        paths = dependencies(depfile)
        required = unit['required_dependencies']
        allowed = dict(unit['optional_pinned_system_dependencies'], **required)
        if not set(required) <= paths <= set(allowed):
            receipt['dependency_mismatch'] = dict(unit=label, missing=sorted(set(required)-paths), extra=sorted(paths-set(allowed)))
            raise RuntimeError('fresh M outside immutable dependency contract')
        pins = {p: allowed[p] for p in sorted(paths)}
        check_pins(pins)
        selected[label] = pins
        receipt.setdefault('fresh_M', {})[label] = dict(count=len(paths), SHA256=H(depfile), dependencies_SHA256=pins)
    receipt['all3_fresh_M_before_compile'] = True
    for unit in P['units']:
        label = unit['name']
        check_pins(selected[label])
        md = W / (label + '.actual.d')
        command(label + '-compile', unit['compiler_prefix'] + ['-MD', '-MF', str(md), '-MT', unit['object'], '-c', unit['source'], '-o', unit['object']], 100)
        receipt['new_HOST_TUs'] += 1
        assert dependencies(md) == set(selected[label])
        check_pins(selected[label])
        receipt.setdefault('MD', {})[label] = dict(count=len(selected[label]), SHA256=H(md), equal_fresh_M=True)
    check_pins(P['reused_objects'])
    link = P['link_argv']
    objects = [p for p in link if p.endswith('.o')]
    replacements = {u['parent_object']: u['object'] for u in P['units']}
    assert objects == [replacements.get(p, p) for p in P['parent_object_order']]
    assert len(objects) == len(set(objects)) == 32 and len(P['reused_objects']) == 29
    assert max(link.index(p) for p in objects) < min(i for i, arg in enumerate(link) if arg.startswith('-l'))
    command('link', link, 60)
    candidate = Path(link[link.index('-o') + 1])
    baseline = symbols(P['parent_library'], 'parent-nm')
    assert len(baseline) == 280 and symbols(candidate, 'candidate-nm') == baseline
    assert elf(candidate, 'candidate-elf') == elf(P['parent_library'], 'parent-elf') == P['NEEDED']
    assert providers(P['parent_library'], 'parent-ldd') == P['providers']
    assert providers(candidate, 'candidate-ldd') == P['providers'] and len(P['providers']) == 71
    owner, observer = 'glm_selected_value_layer0::detail::decode_owner', 'glm_sparse_r131::current'
    verify_tls('nov', W / 'nov.o', (owner, observer))
    verify_tls('prepare', W / 'prepare.o', (observer,))
    verify_tls('output', W / 'output.o', (observer,))
    verify_tls('bridge', P['bridge_object'], (owner,))
    verify_tls('linked', candidate, (owner, observer))
    receipt.update(passed=True, parent_step='1103606.143', reused_objects=29, linked_objects=32,
        all3_MD_equal_fresh_M=True, object_SHA256={u['name']: H(u['object']) for u in P['units']},
        library_SHA256=H(candidate), library_bytes=candidate.stat().st_size,
        providers=71, strong_exports_preserved=280, tls_storage_verified=True,
        GPU_GO=False, Model_GO=False, Native_dlopen=False)


try:
    main()
except Exception as exc:
    receipt.update(error_type=type(exc).__name__, detail=str(exc)[:350])
finally:
    receipt['elapsed_seconds'] = time.monotonic() - anchor
    with (W / 'build-result.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
raise SystemExit(0 if receipt['passed'] else 1)
