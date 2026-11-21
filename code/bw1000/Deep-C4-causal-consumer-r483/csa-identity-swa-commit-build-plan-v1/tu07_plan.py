"""Derive an exact incremental TU07 build from the original CPU1113 producer."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shlex


ORIGINAL_ROOT = PurePosixPath('/refactor/candidates/Deep-C4-constant-slots-r482-Native-CPU-v1')
PARENT_RECEIPT = ORIGINAL_ROOT / 'CPU-result.json'
PARENT_RECEIPT_SHA = '1d3938d699393a85ccddc65a6aeb34a687db14447c4b5ba8861386174d029a6b'
PARENT_LIBRARY_SHA = '90ba080a7666c35e4160003412be2f0c127b176eba1fc6c5fa0f1eb21ea244b1'
SOURCE_NAME = 'deepseek_v4_csa_attention_batch_r784.cpp'
HEADER_NAME = 'identity_swa_commit.hpp'
OLD_SOURCE_SHA = '2611335a7d57ee12411f3d7d5f076bf138f6db17d8607fec184e8d209bff8c8c'
CLASS_HEADER_SHA = '38b53b223be516315c8e6479e20a9cfdcdf089f0a1e59fe328faf2492d4991b1'
TARGET_SOURCE_MANIFEST_SHA = '58fe99c729dc44509d5e32f12ce03760875815087982732a932d176f25406ea6'
OUTPUT = PurePosixPath('/refactor/candidates/Deep-CSA-identity-SWA-commit-CPU-v1')
STAGED = PurePosixPath('/refactor/incoming/Deep-CSA-identity-SWA-commit-CPU-v1/native-src')
COMPILER = '/usr/bin/g++'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        result = sha256()
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def unique_command(receipt, suffix):
    [command] = [command for command in receipt['commands'] if command['argv'][-len(suffix):] == suffix]
    require(command['rc'] == 0 and command['argv'][0] == 'g++', 'original compiler command did not pass')
    return command['argv']


def derive(raw_receipt, source_manifest):
    require(digest(raw_receipt) == PARENT_RECEIPT_SHA, 'original CPU1113 receipt identity differs')
    require(digest(source_manifest) == TARGET_SOURCE_MANIFEST_SHA, 'rendered candidate manifest differs')
    pins = json.loads(source_manifest)['files_SHA256']
    require(set(pins) == {SOURCE_NAME, HEADER_NAME, 'SOURCE_PROOF.json'}, 'unexpected candidate source inventory')
    receipt = json.loads(raw_receipt)
    require(receipt['actual_CPU'] == '910407.1113' and receipt['completed'] is True and receipt['passed'] is True
            and receipt['all19_MD_equal_fresh_M'] is True and not receipt['unmatched_dependencies']
            and receipt['library_SHA256'] == PARENT_LIBRARY_SHA, 'CPU1113 did not qualify the parent build')
    unit = receipt['compiled']['TU07']
    previous = PurePosixPath(unit['source'])
    require(unit['kind'] == 'HOST' and previous.is_relative_to(ORIGINAL_ROOT / 'project-mirror')
            and previous.name == SOURCE_NAME and unit['source_SHA256'] == OLD_SOURCE_SHA
            and unit['object'] == str(ORIGINAL_ROOT / 'objects/TU07.o')
            and unit['MD'] == str(ORIGINAL_ROOT / 'TU07.actual.d'), 'original TU07 binding differs')
    deps = dict(receipt['fresh_dependencies_SHA256']['TU07'])
    require(len(deps) == 996 and deps.pop(str(previous)) == OLD_SOURCE_SHA
            and sum(path.endswith('/deepseek_v4_c4_indexer.hpp') and name == CLASS_HEADER_SHA
                    for path, name in deps.items()) == 1, 'original TU07 include closure differs')
    objects = dict(receipt['all50_linked_objects_SHA256'])
    require(len(objects) == 50 and objects.pop(unit['object']) == unit['object_SHA256'],
            'original TU07 object is not in the 50-object link')
    compile_tail = ['-MD', '-MF', unit['MD'], '-c', unit['source'], '-o', unit['object']]
    original_compile = unique_command(receipt, compile_tail)
    prefix = [COMPILER, '-iquote', str(previous.parent), *original_compile[1:-len(compile_tail)]]
    source = str(STAGED / SOURCE_NAME)
    header = str(STAGED / HEADER_NAME)
    require(source not in deps and header not in deps, 'candidate path collides with an original dependency')
    deps.update({source: pins[SOURCE_NAME], header: pins[HEADER_NAME]})
    obj = str(OUTPUT / 'TU07.o')
    fresh_depfile = str(OUTPUT / 'TU07.before.d')
    md = str(OUTPUT / 'TU07.actual.d')
    library = str(OUTPUT / 'libDeep_csa_identity_swa_commit.so')
    original_library = receipt['library']
    require(original_library == str(ORIGINAL_ROOT / 'libDeep_constant_slot_metadata_r482.so'),
            'original link output differs')
    original_link = unique_command(receipt, ['-o', original_library])
    linked_objects = [item for item in original_link if item.endswith('.o')]
    require(len(linked_objects) == len(set(linked_objects)) == 50
            and set(linked_objects) == set(objects) | {unit['object']}
            and all(flag in original_link for flag in ('-Wl,-Bsymbolic', '-Wl,--no-undefined'))
            and max(original_link.index(obj) for obj in linked_objects) <
            min(index for index, item in enumerate(original_link) if item.startswith('-l')),
            'original linker options or object order differ')
    link = [COMPILER, *(obj if item == unit['object'] else item for item in original_link[1:-1]), library]
    return dict(
        parent_receipt_SHA256=PARENT_RECEIPT_SHA, rendered_manifest_SHA256=TARGET_SOURCE_MANIFEST_SHA,
        source=source, staged_files_SHA256={str(STAGED/name): pin for name, pin in pins.items()},
        expected_dependencies_SHA256=deps, parent_MD=unit['MD'], parent_MD_SHA256=unit['MD_SHA256'],
        parent_source=str(previous), parent_source_SHA256=OLD_SOURCE_SHA,
        parent_object=unit['object'], parent_object_SHA256=unit['object_SHA256'],
        reused_objects_SHA256=objects, parent_library=original_library, parent_library_SHA256=PARENT_LIBRARY_SHA,
        parent_NEEDED=receipt['NEEDED'], parent_loader_paths_SHA256=receipt['new_transitive_library_SHA256'],
        parent_required_CABI=receipt['new_CABI'], object=obj, library=library, fresh_depfile=fresh_depfile,
        actual_depfile=md, preprocess=prefix+['-M', '-MF', fresh_depfile, '-MT', obj, source],
        compile=prefix+['-MD', '-MF', md, '-c', source, '-o', obj], link=link,
    )


def read_dependencies(path, expected_target):
    lines = Path(path).read_text(encoding='utf8').replace('\\\n', ' ').splitlines()
    require(len(lines) == 1, 'expected one dependency rule')
    target, sep, dependencies = lines[0].partition(':')
    values = shlex.split(dependencies)
    require(sep and shlex.split(target) == [expected_target] and values
            and all(value.startswith('/') for value in values), 'unexpected generated dependency rule')
    return set(values)


def verify_recorded_dependencies(plan):
    md = plan['parent_MD']
    require(file_sha(md) == plan['parent_MD_SHA256'], 'original TU07 depfile changed')
    expected = set(plan['expected_dependencies_SHA256'])
    expected -= set(plan['staged_files_SHA256'])
    expected.add(plan['parent_source'])
    require(read_dependencies(md, plan['parent_object']) == expected,
            'original TU07 depfile does not match the receipt')


def verify_inputs(plan):
    """Read-only target check before a separate zero-GRES build claim can execute."""
    verify_recorded_dependencies(plan)
    checks = dict(plan['expected_dependencies_SHA256'], **plan['staged_files_SHA256'])
    checks.update(plan['reused_objects_SHA256'])
    checks.update(plan['parent_loader_paths_SHA256'])
    for role in ('parent_source', 'parent_object', 'parent_library'):
        checks[plan[role]] = plan[role+'_SHA256']
    for path, pin in checks.items():
        require(file_sha(path) == pin, f'dependency bytes differ: {path}')
    return len(checks)


def verify_new_depfile(plan, path):
    """Require the proposed include set on both GCC -M and the later -MD."""
    expected = plan['expected_dependencies_SHA256']
    require(read_dependencies(path, plan['object']) == set(expected),
            'fresh TU07 dependency set differs from the frozen substitution')
    for source, pin in expected.items():
        require(file_sha(source) == pin, f'preprocessor dependency changed: {source}')
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent_receipt', type=Path)
    parser.add_argument('rendered_manifest', type=Path)
    parser.add_argument('--verify-target-inputs', action='store_true')
    parser.add_argument('--verify-depfile', choices=('fresh', 'actual'))
    parser.add_argument('--write-local-plan', type=Path)
    args = parser.parse_args()
    require(not args.verify_depfile or args.verify_target_inputs, 'dependency checks require the target scope')
    require(not args.verify_target_inputs or args.write_local_plan is None,
            'target preflight is read only')
    if args.verify_target_inputs:
        require(args.parent_receipt.resolve(strict=True).as_posix() == str(PARENT_RECEIPT)
                and args.rendered_manifest.resolve(strict=True).as_posix() == str(STAGED / 'SOURCE_FILES.json'),
                'verify only the original producer receipt and the frozen staged manifest')
    plan = derive(args.parent_receipt.read_bytes(), args.rendered_manifest.read_bytes())
    plan_raw = (json.dumps(plan, indent=2, sort_keys=True) + '\n').encode()
    if args.write_local_plan is not None:
        with args.write_local_plan.open('xb') as stream:
            stream.write(plan_raw)
    checked = verify_inputs(plan) if args.verify_target_inputs else None
    depfile_checked = None
    if args.verify_depfile:
        path_key = 'fresh_depfile' if args.verify_depfile == 'fresh' else 'actual_depfile'
        depfile_checked = verify_new_depfile(plan, plan[path_key])
    print(json.dumps(dict(plan_SHA256=digest(plan_raw), target_files_verified=checked, target_compiles=0,
                          expected_dependencies=len(plan['expected_dependencies_SHA256']),
                          reused_objects=len(plan['reused_objects_SHA256']),
                          parent_loader_paths=len(plan['parent_loader_paths_SHA256']),
                          actual_depfile_entries_verified=depfile_checked), sort_keys=True))


if __name__ == '__main__':
    main()
