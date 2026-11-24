"""Render the trace-only TU07 stage from the frozen one-object producer."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PARENT = BASE.parent / 'Deep-C4-constant-slot-metadata-r482/Native-CPU-v1'
STAGE = '/refactor/incoming/Deep-CSA-identity-SWA-callsite-trace-CPU-v1'
OUTPUT = '/refactor/candidates/Deep-CSA-identity-SWA-callsite-trace-CPU-v1'
RENDERER = 'csa-identity-swa-callsite-trace-v1/render_trace.py'
RENDERER_PIN = '5e8c31c40bcce96166bcf48876f31fb92c7933f7b8c5ce8568720373414dff0b'
TRACE_TU_PIN = '255dd2844e3ad7f9e6066aa95f18853de4b21f1b43cd665dcede571f4d752155'
TRACE_HEADER_PIN = 'bbbeb9f301385acfe9cb3fb91475782234647fd8c307b43b1233031ec03bfa2b'
BASE_PLAN = 'csa-identity-swa-commit-build-plan-v1/tu07_plan.py'
BASE_PLAN_PIN = '9c581edf061ab3536450fd1c13840e91c3d35f87a95297db6fd3572c8446207e'
BASE_INSIDE = 'csa-identity-swa-commit-CPU-v1/compile_once.py'
BASE_INSIDE_PIN = '07dc6dc738789451fe6459f180e4a718ba188fd888ee05375ab669697501e51d'
ENV_PIN = 'd2b7f8d3370d5f5427b9e8d04324642d560305bad971fe429ab0372ad2cc250a'
TOOLS_PIN = '4bbb8cd130125f47cec62c943f4c4ade5980e288efab5d5c9116f1de452dd8a0'
EXPORTS = ('modern_deep_csa_commit_trace_arm_v1', 'modern_deep_csa_commit_trace_clear_v1',
           'modern_deep_csa_commit_trace_read_v1')
TU = 'deepseek_v4_csa_attention_batch_r784.cpp'
TRACE = 'commit_trace.hpp'


def digest(raw):
    return sha256(raw).hexdigest()


def json_bytes(doc):
    return (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def pinned(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f'expected frozen bytes: {path.name}')
    return raw


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one frozen anchor: {old}')
    return text.replace(old, new, 1)


def source_bundle():
    path = BASE / RENDERER
    renderer = ModuleType('frozen_trace_renderer')
    renderer.__file__ = str(path)
    exec(compile(pinned(path, RENDERER_PIN), str(path), 'exec'), vars(renderer))
    files, proof = renderer.build()
    if digest(files[TU]) != TRACE_TU_PIN or digest(files[TRACE]) != TRACE_HEADER_PIN:
        raise ValueError('published trace source differs')
    header = once(files[TRACE].decode(), 'namespace deep_csa_commit_trace {',
                  'namespace {\nnamespace deep_csa_commit_trace {')
    header = once(header, '} // namespace deep_csa_commit_trace',
                  '} // namespace deep_csa_commit_trace\n} // namespace')
    files[TRACE] = header.encode()
    proof.update(trace_header_prior_SHA256=TRACE_HEADER_PIN, trace_helpers_have_internal_linkage=True,
                 only_added_strong_exports=list(EXPORTS),
                 files_SHA256={name: digest(raw) for name, raw in files.items() if name != 'SOURCE_PROOF.json'})
    files['SOURCE_PROOF.json'] = json_bytes(proof)
    return files, json_bytes({'files_SHA256': {name: digest(raw) for name, raw in files.items()}})


def plan_source(manifest_sha):
    text = pinned(BASE / BASE_PLAN, BASE_PLAN_PIN).decode()
    changes = (
        ('58fe99c729dc44509d5e32f12ce03760875815087982732a932d176f25406ea6', manifest_sha),
        ('/refactor/candidates/Deep-CSA-identity-SWA-commit-CPU-v1', OUTPUT),
        ('/refactor/incoming/Deep-CSA-identity-SWA-commit-CPU-v1/native-src', STAGE + '/native-src'),
        ("{SOURCE_NAME, HEADER_NAME, 'SOURCE_PROOF.json'}", "{SOURCE_NAME, HEADER_NAME, 'commit_trace.hpp', 'SOURCE_PROOF.json'}"),
        ("    deps.update({source: pins[SOURCE_NAME], header: pins[HEADER_NAME]})",
         "    trace = str(STAGED / 'commit_trace.hpp')\n"
         "    require(trace not in deps, 'trace path collides with an original dependency')\n"
         "    deps.update({source: pins[SOURCE_NAME], header: pins[HEADER_NAME], trace: pins['commit_trace.hpp']})"),
        ('libDeep_csa_identity_swa_commit.so', 'libDeep_csa_identity_swa_callsite_trace.so'),
    )
    for old, new in changes:
        text = once(text, old, new)
    return text.encode()


def worker_source(plan_sha):
    text = pinned(BASE / BASE_INSIDE, BASE_INSIDE_PIN).decode()
    changes = (
        (BASE_PLAN_PIN, plan_sha),
        ("'native-src/identity_swa_commit.hpp'}", "'native-src/identity_swa_commit.hpp', 'native-src/commit_trace.hpp'}"),
        ("parent in {'1087656', '1103606'}", "parent in {'1103606', '1194900'}"),
        ("    syms = strong_symbols(run(prefix + '-nm', ['/usr/bin/nm', '-D', '--defined-only', library]))",
         "    raw_symbols = run(prefix + '-nm', ['/usr/bin/nm', '-D', '--defined-only', library])\n"
         "    require('deep_csa_commit_trace' not in raw_symbols.replace('modern_deep_csa_commit_trace_', ''),\n"
         "            'recorder implementation escaped the shared-library ABI')\n"
         "    syms = strong_symbols(raw_symbols)"),
        ("    require(candidate_symbols == parent_symbols, 'candidate strong exports differ')",
         f"    additions = {EXPORTS!r}\n"
         "    require(not parent_symbols.keys() & set(additions), 'original unexpectedly contains trace exports')\n"
         "    expected = dict(parent_symbols, **dict.fromkeys(additions, 'T'))\n"
         "    require(candidate_symbols == expected, 'candidate strong exports differ from the three trace controls')\n"
         "    result['trace_CABI'] = list(additions)"),
        ('strong_exports=len(parent_symbols), NEEDED=',
         'strong_exports=len(candidate_symbols), parent_strong_exports=len(parent_symbols), NEEDED='),
    )
    for old, new in changes:
        text = once(text, old, new)
    return text.encode()


def prepare(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    native, source_manifest = source_bundle()
    plan = plan_source(digest(source_manifest))
    worker = worker_source(digest(plan))
    old_env = pinned(PARENT / 'CPU_inside.sh', ENV_PIN)
    prefix, invocation = old_env.rstrip(b'\n').rsplit(b'\n', 1)
    if not invocation.endswith(b'/Deep-C4-constant-slot-metadata-r482/Native-CPU-v1/build_native.py'):
        raise ValueError('original environment invocation differs')
    files = {'native-src/' + name: raw for name, raw in native.items()}
    files.update({'native-src/SOURCE_FILES.json': source_manifest, 'tu07_plan.py': plan, 'compile_once.py': worker,
                  'ORIGINAL_ENV.sh': old_env, 'TOOLCHAIN.json': pinned(PARENT / 'TOOLCHAIN.json', TOOLS_PIN),
                  'inside.sh': prefix + f'\nexport LC_ALL=C\nexec /usr/bin/python3 -I -S -B -u {STAGE}/compile_once.py\n'.encode()})
    manifest = json_bytes({'files_SHA256': {path: digest(raw) for path, raw in files.items()}})
    with TemporaryDirectory(dir=output.parent, prefix='deep-csa-trace-cpu-') as temporary:
        target = Path(temporary) / 'stage'
        (target / 'native-src').mkdir(parents=True)
        for name, raw in files.items():
            (target / name).write_bytes(raw)
        (target / 'STAGE_FILES.json').write_bytes(manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        target.rename(output)
    return dict(stage_manifest_SHA256=digest(manifest), source_manifest_SHA256=digest(source_manifest), members=len(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_output', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.new_output.expanduser().absolute()), sort_keys=True))
