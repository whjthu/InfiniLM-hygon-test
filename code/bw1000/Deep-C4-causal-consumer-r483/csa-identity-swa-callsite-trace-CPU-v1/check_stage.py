"""Check the staged dependency and export deltas against the real parent receipt."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType

import prepare_stage as prep


def load(path):
    raw = path.read_bytes()
    ast.parse(raw, filename=str(path), feature_version=(3, 10))
    scope = ModuleType(path.stem)
    scope.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), vars(scope))
    return scope


def check_exports(worker, plan):
    original = {'original_entry': 'T'}
    candidate = dict(original, **dict.fromkeys(prep.EXPORTS, 'T'))
    real_check_elf = worker.check_elf
    paths = plan['expected_dependencies_SHA256'] | plan['reused_objects_SHA256']

    class Fixture:
        def verify_inputs(self, _):
            return len(paths)

        def verify_new_depfile(self, _, __):
            return len(plan['expected_dependencies_SHA256'])

        def file_sha(self, path):
            return paths.get(path, 'fixture')

    calls = []

    def commands(label, _):
        calls.append(label)

    def invoke(exports):
        result = {}
        try:
            worker.check_elf = lambda _run, _recipe, _plan, phase, _library: original if phase == 'parent' else exports
            worker.build(Fixture(), plan, result, commands)
        finally:
            worker.check_elf = real_check_elf
        return result

    good = invoke(candidate)
    assert list(calls) == ['fresh-M', 'compile', 'link'] and good['trace_CABI'] == list(prep.EXPORTS)
    assert good['parent_strong_exports'] == 1 and good['strong_exports'] == 4 and good['passed'] is True
    cases = [dict(candidate, unexpected='T'), dict(candidate, **{prep.EXPORTS[0]: 'D'}),
             {key: value for key, value in candidate.items() if key != prep.EXPORTS[1]}]
    for exports in cases:
        try:
            invoke(exports)
        except ValueError as error:
            assert 'candidate strong exports differ' in str(error)
        else:
            raise AssertionError('unexpected dynamic symbol set passed')
    forbidden = '_ZN21deep_csa_commit_trace5stateE'
    try:
        worker.check_elf(lambda label, _: 'SYMBOLIC' if label.endswith('readelf') else '00000000 u ' + forbidden,
                         Fixture(), dict(plan, parent_NEEDED=[]), 'candidate', plan['library'])
    except ValueError as error:
        assert 'implementation escaped' in str(error)
    else:
        raise AssertionError('implementation symbol escaped')
    return len(cases) + 1


def check(stage, original_receipt):
    manifest = (stage / 'STAGE_FILES.json').read_bytes()
    stage_pins = json.loads(manifest)['files_SHA256']
    assert len(stage_pins) == 10 and all(prep.digest((stage / name).read_bytes()) == pin for name, pin in stage_pins.items())
    worker = load(stage / 'compile_once.py')
    planner = worker.stage(prep.digest(manifest))
    frozen = load(prep.BASE / prep.BASE_PLAN)
    raw_receipt = original_receipt.read_bytes()
    source_manifest = (stage / 'native-src/SOURCE_FILES.json').read_bytes()
    plan = planner.derive(raw_receipt, source_manifest)
    dependencies = plan['expected_dependencies_SHA256']
    source = str(planner.STAGED / planner.SOURCE_NAME)
    assert len(dependencies) == 998 and dependencies[source] == prep.TRACE_TU_PIN
    source_pins = json.loads(source_manifest)['files_SHA256']
    assert dependencies[str(planner.STAGED / prep.TRACE)] == source_pins[prep.TRACE]
    original = json.loads(raw_receipt)
    parent = dict(original['fresh_dependencies_SHA256']['TU07'])
    del parent[plan['parent_source']]
    assert len(parent) == 995 and parent == {name: dependencies[name] for name in parent}
    assert all(sum(name.endswith('/' + header) for name in parent) == 1 for header in ('array', 'cstddef', 'cstdint'))
    assert len(plan['reused_objects_SHA256']) == 49 and len(plan['parent_loader_paths_SHA256']) == 68
    original_link = frozen.unique_command(original, ['-o', plan['parent_library']])
    assert plan['link'] == ['/usr/bin/g++', *(plan['object'] if a == plan['parent_object'] else a for a in original_link[1:-1]), plan['library']]
    assert plan['compile'][-7:] == ['-MD', '-MF', plan['actual_depfile'], '-c', source, '-o', plan['object']]
    for raw, source_mf in ((raw_receipt + b' ', source_manifest), (raw_receipt, source_manifest + b' ')):
        try:
            planner.derive(raw, source_mf)
        except ValueError:
            continue
        raise AssertionError('modified input bytes passed')
    return {'stage_manifest_SHA256': sha256(manifest).hexdigest(),
            'source_manifest_SHA256': sha256(source_manifest).hexdigest(), 'stage_members': len(stage_pins),
            'parent_receipt_SHA256': sha256(raw_receipt).hexdigest(), 'retained_parent_include_paths': len(parent),
            'expected_fresh_M_MD_paths': len(dependencies), 'reused_objects': len(plan['reused_objects_SHA256']),
            'parent_loader_paths': len(plan['parent_loader_paths_SHA256']), 'new_allowed_strong_exports': list(prep.EXPORTS),
            'negative_cases': check_exports(worker, plan) + 2, 'actual_Host_TU_compiles': 0, 'actual_GPU_steps': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    parser.add_argument('original_receipt', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.stage.resolve(), args.original_receipt), sort_keys=True))
