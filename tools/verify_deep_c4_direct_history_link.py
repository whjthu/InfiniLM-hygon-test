"""Prepare and passively audit a guarded probe link against the DeepSeek C4 candidate.

Run against the original /refactor artifacts with the audited container tools.
The default mode emits argv for the coordinator's bounded link step. The
--verify mode checks the resulting ELF. It leaves mapped provider identity for
the subsequent device test, where the original probe also checks dladdr.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


REPO = Path(__file__).resolve().parents[1]
PLAN = 'code/bw1000/Deep-C4-causal-consumer-r483/CPU-v2/BUILD_PLAN.json'
PLAN_SHA = '587a51c72b7bfb3342676a90b73ffa7cab84bdbfd9c8271ff46d23048fea5082'
PROBE_REPORT = 'results/bw1000/2026-11-17/Deep483-guarded-CPU-v2-original-1103606.156-terminal.json'
PROBE_REPORT_SHA = '08eb7a9d738cd0618683c98705ba54a60eb3bca749a70a4d8164e042bcec8a1b'
CANDIDATE_RECEIPT_SHA = 'f3e2687bb10d01e5340def99ca15b7cfabca17b936c4ed50eaf1191339d75d5f'
CANDIDATE_DIR = '/refactor/candidates/DeepC4-T1-history-host-compile-v1'
CANDIDATE_NATIVE = CANDIDATE_DIR + '/libDeep_c4_direct_history_t1.so'
OUTPUT = '/refactor/candidates/DeepC4-T1-history-guarded-probe-link-v1'
RELINKED_PROBE = OUTPUT + '/libDeep_c4_causal_probe_r483.so'
COMPILER = '/usr/bin/g++'
READELF = '/usr/bin/readelf'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path, expected=None):
    value = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    actual = value.hexdigest()
    require(expected is None or actual == expected, f'bytes differ for {path.name}')
    return actual


def read_pinned_json(path, expected):
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024 and sha256(raw).hexdigest() == expected,
            f'receipt bytes differ for {path.name}')
    return json.loads(raw)


def parse_candidate_receipt(receipt):
    expected = dict(step='1103606.487', completed=True, passed=True, explicit_GPU_calls=0,
                    new_HOST=1, new_GPU_TU=0, model_weights_read=0, reused_objects=49,
                    linked_objects=50, fresh_M_before_compile=True, library=CANDIDATE_NATIVE)
    require(all(type(receipt.get(key)) is type(value) and receipt[key] == value
                for key, value in expected.items()), 'candidate receipt scope differs')
    candidate_sha = receipt.get('library_SHA256')
    require(isinstance(candidate_sha, str) and re.fullmatch(r'[0-9a-f]{64}', candidate_sha),
            'candidate receipt lacks an exact Native SHA-256')
    return candidate_sha


@dataclass(frozen=True)
class Elf:
    needed: tuple
    soname: str | None
    symbolic: bool
    search_paths: tuple
    exports: frozenset
    imports: frozenset


def parse_readelf(dynamic, symbols):
    require('Dynamic section at offset' in dynamic and "Symbol table '.dynsym'" in symbols,
            'expected dynamic tags and dynamic symbol table from a shared object')
    tags = re.findall(r'^[ \t]*0x[\da-fA-F]+[ \t]+\((\w+)\)[ \t]*(.*?)[ \t]*$', dynamic, re.M)
    needed, soname, paths = [], [], []
    symbolic = False
    for kind, value in tags:
        if kind in ('NEEDED', 'SONAME', 'RPATH', 'RUNPATH'):
            match = re.search(r'\[([^]]+)\]$', value)
            require(match is not None, f'invalid ELF {kind} tag')
            if kind == 'NEEDED':
                needed.append(match[1])
            elif kind == 'SONAME':
                soname.append(match[1])
            else:
                paths.append((kind, match[1]))
        elif kind == 'SYMBOLIC' or (kind == 'FLAGS' and 'SYMBOLIC' in value.split()):
            symbolic = True
        elif kind in ('FILTER', 'AUXILIARY', 'AUDIT', 'DEPAUDIT'):
            raise ValueError(f'unexpected ELF loader redirection {kind}')
    require(len(soname) <= 1 and len(needed) == len(set(needed)), 'ambiguous ELF dependency tags')
    exports, imports = set(), set()
    for line in symbols.splitlines():
        cells = line.split()
        if len(cells) < 8 or not re.fullmatch(r'\d+:', cells[0]) or cells[4] not in ('GLOBAL', 'WEAK', 'UNIQUE'):
            continue
        _, _, _, kind, binding, visibility, index, name, *_ = cells
        if index == 'UND':
            imports.add((name, kind, binding, visibility))
        elif visibility in ('DEFAULT', 'PROTECTED'):
            exports.add((name, kind, binding, visibility))
    require(bool(exports), 'shared object has no visible dynamic exports')
    return Elf(tuple(needed), soname[0] if soname else None, symbolic,
               tuple(paths), frozenset(exports), frozenset(imports))


def passive_elf(path, readelf):
    env = {key: value for key, value in os.environ.items() if not key.startswith(('LD_', 'DYLD_'))}
    env['LC_ALL'] = 'C'

    def report(option):
        return subprocess.check_output([str(readelf), '--wide', option, '--', str(path)],
                                       env=env, timeout=20, encoding='utf-8')
    return parse_readelf(report('--dynamic'), report('--dyn-syms'))


def expected_probe(baseline_probe, baseline_native, candidate_native, link, audited_needed, compiled_needed):
    original = link['required_NEEDED']
    require(baseline_probe.needed == tuple(audited_needed) and baseline_probe.needed.count(original) == 1
            and baseline_probe.symbolic and baseline_probe.soname is None,
            'guarded baseline probe does not match the audited link')
    expected_exports = set(link['required_exports'])
    exports = {name for name, kind, bind, _ in baseline_probe.exports
               if name.startswith('deep_c4_probe_') and kind == 'FUNC' and bind == 'GLOBAL'}
    require(len(expected_exports) == 13 and exports == expected_exports, 'guarded probe C ABI differs')
    require(candidate_native.soname in (None, CANDIDATE_NATIVE),
            'candidate SONAME would override the exact candidate path in DT_NEEDED')
    strong_candidate = {symbol for symbol in candidate_native.exports if symbol[2] == 'GLOBAL'}
    strong_baseline = {symbol for symbol in baseline_native.exports if symbol[2] == 'GLOBAL'}
    require(candidate_native.needed == baseline_native.needed == tuple(compiled_needed)
            and candidate_native.symbolic and baseline_native.symbolic
            and candidate_native.search_paths == baseline_native.search_paths
            and strong_candidate and strong_candidate == strong_baseline
            and original not in candidate_native.needed,
            'Native dependency, symbol or search-path closure differs')
    require(CANDIDATE_NATIVE not in baseline_probe.needed, 'baseline probe already references the candidate')
    return tuple(CANDIDATE_NATIVE if name == original else name for name in baseline_probe.needed)


def link_argv(plan, link):
    old = plan['candidate']
    original = link['required_NEEDED']
    argv = link['argv']
    substitutions = {
        'g++': COMPILER,
        '$UNIT:probe': old + '/probe.o',
        original: CANDIDATE_NATIVE,
        old + '/' + link['output']: RELINKED_PROBE,
        '-Wl,-Map,' + old + '/probe-link.map': '-Wl,-Map,' + OUTPUT + '/probe-link.map',
    }
    require(all(argv.count(key) == 1 for key in substitutions) and argv[0] == 'g++'
            and argv[argv.index('-o') + 1] == old + '/' + link['output']
            and argv.index('$UNIT:probe') + 1 == argv.index(original)
            and '-Wl,--no-undefined' in argv and '-Wl,-Bsymbolic' in argv,
            'original guarded-probe linker recipe differs')
    return [substitutions.get(arg, arg) for arg in argv]


def verify_relinked(probe, baseline, needed):
    require(probe.needed == needed and probe.soname == baseline.soname and probe.symbolic == baseline.symbolic
            and probe.search_paths == baseline.search_paths and probe.exports == baseline.exports
            and probe.imports == baseline.imports,
            'relinked probe must change only the pinned Native dependency')


def audit(refactor_root, verify=False):
    def artifact(container_path):
        return refactor_root.joinpath(*PurePosixPath(container_path).relative_to('/refactor').parts)

    plan = read_pinned_json(REPO / PLAN, PLAN_SHA)
    link, = plan['links']
    original = read_pinned_json(REPO / PROBE_REPORT, PROBE_REPORT_SHA)['independent_original_build']['result']
    receipt = read_pinned_json(artifact(CANDIDATE_DIR + '/CPU-result.json'), CANDIDATE_RECEIPT_SHA)
    candidate_sha = parse_candidate_receipt(receipt)
    reference = link['required_NEEDED']
    baseline_sha = original['project_provider_SHA256'][reference]
    require(candidate_sha != baseline_sha, 'candidate Native bytes equal the baseline')
    pins = {
        plan['candidate'] + '/probe.o': original['object_SHA256'],
        plan['candidate'] + '/' + link['output']: original['probe_SHA256'],
        reference: baseline_sha,
        CANDIDATE_NATIVE: candidate_sha,
    }
    for name, value in pins.items():
        digest(artifact(name), value)
    digest(Path(READELF), plan['tools'][READELF]['sha256'])
    digest(Path(COMPILER), plan['tools'][COMPILER]['sha256'])
    baseline = passive_elf(artifact(plan['candidate'] + '/' + link['output']), READELF)
    proposed = expected_probe(baseline, passive_elf(artifact(reference), READELF),
                              passive_elf(artifact(CANDIDATE_NATIVE), READELF), link,
                              original['needed'], receipt['NEEDED'])
    result = dict(candidate_receipt_SHA256=CANDIDATE_RECEIPT_SHA, input_SHA256=pins,
                  linker_argv=link_argv(plan, link), expected_DT_NEEDED=proposed,
                  expected_transitive_SHA256={
                      **{path: pin for path, pin in plan['dynamic_SHA256'].items() if path != reference},
                      CANDIDATE_NATIVE: candidate_sha,
                  }, relinked_probe_passive_verified=False, actual_link_invocations=0,
                  actual_GPU_calls=0, runtime_provider_verified=False, GPU_GO=False, Model_GO=False)
    if verify:
        output = artifact(RELINKED_PROBE)
        output_sha = digest(output)
        verify_relinked(passive_elf(output, READELF), baseline, proposed)
        result.update(relinked_probe_SHA256=digest(output, output_sha), relinked_probe_passive_verified=True)
    else:
        require(not artifact(OUTPUT).exists(), 'output namespace exists; reconcile the original link')
    for name, value in pins.items():
        digest(artifact(name), value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refactor-root', type=Path, default=Path('/refactor'))
    parser.add_argument('--verify', action='store_true', help='audit an already-linked probe without running it')
    args = parser.parse_args()
    print(json.dumps(audit(args.refactor_root, args.verify), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
