"""Compile just the direct-history TU and relink against original CPU1113 objects."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import time


HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'PLAN.json').read_bytes())
LABEL = PLAN['label']


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else digest_file(handle)


def digest_file(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024**2), b''):
        digest.update(block)
    return digest.hexdigest()


def dependency_paths(path):
    lhs, colon, rhs = Path(path).read_text().replace('\\\n', ' ').partition(':')
    names = set(shlex.split(rhs))
    need(bool(lhs and colon and names) and all(p.startswith('/') for p in names), 'UNQUALIFIED_DEPENDENCY_LIST')
    return names


def strong_symbols(text):
    return {row[-1]: row[-2] for line in text.splitlines()
            if len(row := line.split()) >= 3 and row[-2] in ('T', 'D', 'B', 'R')}


def dynamic_paths(text):
    need('not found' not in text and 'undefined symbol' not in text, 'LINK_GRAPH_UNRESOLVED')
    return set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)', text, re.M))


def inspect_reuse(prior):
    original = PLAN['unit']['original_object']
    objects = prior['all50_linked_objects_SHA256']
    compiled = prior['compiled']
    replacements = {row['original_object']: row['object'] for row in compiled.values()}
    need(len(compiled) == len(replacements) == 19 and len(objects) == 50, 'ORIGINAL_19_31_GRAPH')
    need(compiled['TU31']['original_object'] == original and compiled['TU31']['source'] == PLAN['unit']['source']
         and compiled['TU31']['source_SHA256'] == PLAN['unit']['source_SHA256'], 'ORIGINAL_TU31_IDENTITY')
    original_link_order = [replacements.get(v, v) for v in PLAN['link_argv'] if v.endswith('.o')]
    need(len(original_link_order) == len(set(original_link_order)) == 50
         and set(original_link_order) == set(objects), 'ORIGINAL_LINK_ORDER_IDENTITY')
    return objects, replacements


def main():
    started = time.monotonic()
    epoch = float(os.environ['TUNING_CPU_EPOCH'])
    deadline = float(os.environ['TUNING_CPU_DEADLINE'])
    need(0 <= time.time() - epoch < 270 and deadline == epoch + 300, 'ORIGINAL_EPOCH_BUDGET')
    need(os.environ['SLURM_JOB_ID'] == PLAN['parent'] and os.environ['SLURM_STEP_ID'].isdecimal()
         and socket.gethostname().split('.')[0] == PLAN['node'], 'WRONG_SLURM_IDENTITY')
    need(all(os.environ.get(k) == '-1' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
         and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'CPU_ONLY_ENVIRONMENT')
    mf = HERE / 'SOURCE_FILES.json'
    need(sha(mf) == os.environ['TUNING_CPU_MANIFEST_SHA'], 'SOURCE_MANIFEST_PIN')
    for name, pin in json.loads(mf.read_bytes())['files_SHA256'].items():
        need(Path(name).name == name and sha(HERE / name) == pin, 'STAGED_FILE_PIN')
    work = Path('/refactor/candidates') / LABEL
    work.mkdir(mode=0o700)
    report = dict(step=PLAN['parent']+'.'+os.environ['SLURM_STEP_ID'], node=PLAN['node'], epoch=epoch,
                  deadline=deadline, commands=[], completed=False, passed=False, explicit_GPU_calls=0,
                  new_HOST=0, new_GPU_TU=0, reused_objects=0, model_weights_read=0)
    def save():
        temporary = work / 'CPU-result.tmp'
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2)+'\n')
        os.replace(temporary, work / 'CPU-result.json')

    def run(label, command, limit):
        left = min(deadline - time.time() - 12, 260 - (time.monotonic() - started))
        need(left > 1, 'COMPILE_BUDGET')
        item = dict(label=label, argv=command)
        report['commands'].append(item)
        save()
        process = subprocess.run(command, capture_output=True, text=True, timeout=min(left, limit))
        for which in ('stdout', 'stderr'):
            p = work / (label+'.'+which)
            p.write_text(getattr(process, which))
            item[which+'_SHA256'] = sha(p)
        item['rc'] = process.returncode
        save()
        need(process.returncode == 0, label+'_FAILED')
        return process.stdout + process.stderr

    try:
        for path, tool in PLAN['tools'].items():
            need(str(Path(path).resolve(strict=True)) == tool['canonical'] and sha(path) == tool['sha256'], 'TOOLCHAIN_DRIFT')
        for key, value in PLAN['required_environment'].items():
            need(os.environ.get(key) == value, 'TOOLCHAIN_ENVIRONMENT_DRIFT')
        need(shutil.which('g++') == '/usr/bin/g++', 'HOST_COMPILER_SELECTION')
        prior_path = Path(PLAN['reference_receipt'])
        need(prior_path.stat().st_size == PLAN['reference_receipt_bytes'] and sha(prior_path) == PLAN['reference_receipt_SHA256'],
             'ORIGINAL_BUILD_RECEIPT_DRIFT')
        prior = json.loads(prior_path.read_bytes())
        original = PLAN['reference_library']
        need(prior['actual_CPU'] == '910407.1113' and prior['completed'] and prior['passed'] and prior['all19_MD_equal_fresh_M']
             and prior['library'] == original and prior['library_SHA256'] == sha(original) == PLAN['reference_library_SHA256'],
             'ORIGINAL_BUILD_QUALIFICATION')
        objects, replacements = inspect_reuse(prior)
        need(all(sha(path) == value for path, value in objects.items()), 'REUSED_OBJECT_DRIFT')
        changed = PLAN['unit']
        old = changed['source']
        need(sha(old) == changed['source_SHA256'], 'ORIGINAL_TU31_SOURCE_DRIFT')
        patch = HERE / 'deep_c4_smallt_r9252.cpp'
        need(sha(patch) == PLAN['candidate_source_SHA256'], 'CHANGED_TU31_SOURCE_DRIFT')
        # The new source directory contains just this copied TU. Quoted local includes
        # are looked up in the original mirrored directory before all original -I flags.
        source_dir = work / 'source'
        source_dir.mkdir()
        candidate = str(source_dir / patch.name)
        shutil.copyfile(patch, candidate)
        recipe = [*changed['prefix'], '-iquote', str(Path(old).parent)]
        obj = str(work / 'TU31.o')
        before = work / 'TU31.before.d'
        after = work / 'TU31.actual.d'
        original_deps = prior['fresh_dependencies_SHA256']['TU31']
        need(original_deps.get(old) == changed['source_SHA256']
             and PLAN['class_header_SHA256'] in original_deps.values(), 'ORIGINAL_TU31_DEPENDENCY_GRAPH')
        expected_deps = dict(original_deps)
        del expected_deps[old]
        expected_deps[candidate] = PLAN['candidate_source_SHA256']
        run('host-M', recipe+['-M','-MF',str(before),'-MT',obj,candidate], 75)
        need(dependency_paths(before) == set(expected_deps)
             and all(sha(p) == pin for p, pin in expected_deps.items()), 'FRESH_M_DRIFT')
        report['fresh_M_before_compile'] = True
        report['fresh_M_path_count'] = len(expected_deps)
        report['fresh_M_SHA256'] = sha(before)
        run('host-build', recipe+['-MD','-MF',str(after),'-c',candidate,'-o',obj], 100)
        need(dependency_paths(after) == set(expected_deps)
             and all(sha(p) == pin for p, pin in expected_deps.items()), 'COMPILED_MD_DRIFT')
        report.update(new_HOST=1, MD_SHA256=sha(after), object_SHA256=sha(obj))
        replacements[changed['original_object']] = obj
        new_library = str(work / 'libDeep_c4_direct_history_t1.so')
        args = [replacements.get(x, x) for x in PLAN['link_argv']]
        args[args.index('-o') + 1] = new_library
        linked = [p for p in args if p.endswith('.o')]
        reused = set(objects) - {prior['compiled']['TU31']['object']}
        need(len(linked) == len(set(linked)) == 50 and set(linked) == reused | {obj}
             and max(args.index(p) for p in linked) < min(i for i, value in enumerate(args) if value.startswith('-l')),
             'RELINK_49_PLUS_1_SCOPE')
        run('link', args, 90)
        old_exports = strong_symbols(run('old-symbols', ['/usr/bin/nm','-D','--defined-only',original], 20))
        new_exports = strong_symbols(run('new-symbols', ['/usr/bin/nm','-D','--defined-only',new_library], 20))
        need(bool(old_exports) and new_exports == old_exports, 'STRONG_SYMBOL_DRIFT')
        elf = run('new-elf', ['/usr/bin/readelf','-d',new_library], 20)
        needed = re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]', elf)
        need(needed == prior['NEEDED'] and 'SYMBOLIC' in elf, 'ELF_LINKAGE_DRIFT')
        resolved = dynamic_paths(run('loader-metadata', ['/usr/bin/ldd','-r',new_library], 25))
        pins = prior['new_transitive_library_SHA256']
        need(resolved == set(pins) and all(sha(p) == pin for p, pin in pins.items()), 'TRANSITIVE_PROVIDER_DRIFT')
        need(all(sha(p) == objects[p] for p in reused), 'REUSED_OBJECTS_CHANGED_DURING_LINK')
        report.update(passed=True, reused_objects=49, linked_objects=50,
                      retained_strong_symbols=len(old_exports), transitive_library_count=len(pins),
                      library=new_library, library_SHA256=sha(new_library), NEEDED=needed,
                      loader_child_implicit_initialization='UNKNOWN', numerical_or_GPU_qualified=False)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report.update(error_kind=type(error).__name__, failed_check=str(error)[:240])
    finally:
        report.update(completed=True, elapsed_seconds=time.monotonic()-started)
        save()
    print(json.dumps({k: report.get(k) for k in ('step','passed','new_HOST','new_GPU_TU','reused_objects','failed_check')}),
          flush=True)
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
