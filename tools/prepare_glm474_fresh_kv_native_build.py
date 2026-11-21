"""Bind a fresh K/V Native build to the actual CPU308 link and qualified pack object."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import shlex
import subprocess

from read_training_slurm_membership import allocated_dcu


REPO = Path(__file__).resolve().parents[1]
CALLER = REPO / 'code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-caller-source-v1'
CALLER_MANIFEST = '66f2657b00a1d9ed3d487d6182bbb70858c0c4764afba91fa5c6e9a8fdc946a3'
CALLER_PUBLICATION = 'de7d18d6ce837171ebd01726adc0077602bada6f'
ORIGIN = PurePosixPath('/refactor/results/GLM474-depth-stratified-observer-CPU-v1')
ORIGIN_BUILD_SHA = 'aa669c0ebc18072b0085c1f98fe34bee898e50af28f6e0542bb8c485ac200b6f'
ORIGIN_LIBRARY_SHA = 'ddfeae81526758fb06d68e4f688e9eeaa48e3b173018d0040a3de0c99ba8dfb5'
KERNEL = PurePosixPath('/refactor/candidates/GLM474-fresh-kv-pack-gfx936-CPU-v1')
KERNEL_LABEL = KERNEL.name
KERNEL_MANIFEST = '333551bc79cbd2e56e18274737b00c57c0a918cc00700b167a61b9a316ac7850'
KERNEL_REQUEST = 'b9a7f0b78c566911ed40004fc906ce44'
EXPORTS = ['glm_r474_fresh_kv_pack_t128', 'glm_r474_fresh_kv_pack_t64']
NATIVE = PurePosixPath('/refactor/candidates/GLM474-fresh-kv-pack-Native-CPU-v1')
OBJECT_NAMES = tuple(f'u{i:02}' for i in range(32))
CHANGED = {'u15': 'NoV_tp16.cpp', 'u17': 'prepare_tp16.cpp'}
SOURCE_DELTA = {'u15': ('fresh_kv_route.hpp',),
                'u17': ('fresh_kv_route.hpp', 'fresh_kv_state.inc', 'fresh_kv_materialize.inc')}
SYSTEM_PLAN_SHA = '48188514c7137d61d553232eb08f050f6f65d1c41a0be020aeceefb2275fe6c6'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected, limit=16*1024**2):
    path = Path(path)
    with path.open('rb') as handle:
        data = handle.read(limit+1)
    need(0 < len(data) <= limit and sha256(data).hexdigest() == expected,
         'frozen input bytes differ: ' + path.name)
    return data


def caller_sources():
    manifest = json.loads(checked(CALLER/'SOURCE_FILES.json', CALLER_MANIFEST, 8192))['files_SHA256']
    for name, pin in manifest.items():
        need(Path(name).name == name, 'generated source pin escaped its package')
        checked(CALLER/name, pin)
    result = runpy.run_path(str(CALLER/'render.py'))['sources']()
    expected = json.loads((CALLER/'PLAN.json').read_bytes())['generated_sources_SHA256']
    need({name: sha256(raw).hexdigest() for name,raw in result.items()} == expected,
         'new source bytes differ from the published caller')
    return result, expected


def compiled_kernel(proof, worker):
    row = proof['row']
    need(proof['request'] == KERNEL_REQUEST and proof['state'] == 'COMPILE_VERIFIED'
         and proof['independent_observation'] is True and proof['actual_GPU_steps'] == 0
         and len(row) == 7 and re.fullmatch(r'1103606\.\d+', row[0])
         and row[1:4] == [KERNEL_LABEL, 'COMPLETED', '0:0'] and row[5] == 'f12r2n13',
         'fresh K/V kernel lacks its independently verified target terminal')
    resources = dict(item.split('=', 1) for item in row[6].split(','))
    need(row[4].isdecimal() and int(row[4]) <= 300 and allocated_dcu(row[6]) == 0
         and resources.get('node') == '1' and resources.get('cpu') == '4'
         and resources.get('mem') in {'8G', '8192M', '8388608K'}, 'original kernel accounting scope differs')
    need(worker['passed'] is True and worker['completed'] is True and worker['original_step'] == row[0]
         and worker['manifest_SHA256'] == KERNEL_MANIFEST and worker['exports'] == EXPORTS
         and worker['object'] == (KERNEL/'fresh_kv_pack.o').as_posix()
         and worker['object_SHA256'] == proof['object_SHA256']
         and worker['actual_device_TUs'] == 1 and worker['actual_GPU_steps'] == 0,
         'kernel object differs from independently observed receipt')
    return dict(path=worker['object'], SHA256=worker['object_SHA256'], slurm_step=row[0])


def command(build, name):
    [record] = [record for record in build['commands'] if record['label'] == name]
    need(record['rc'] == 0 and record['owned_child_reaped'] is True, f'unqualified original command: {name}')
    return record['argv']


def make_recipe(build, source_pins, kernel):
    need(build['passed'] is True and build['step'] == '1103606.308'
         and build['library']['path'] == (ORIGIN/'B/libGLM_depth_observer_B.so').as_posix()
         and build['library']['SHA256'] == ORIGIN_LIBRARY_SHA and build['actual']['objects'] == 32
         and set(build['objects']) == set(build['fresh_M']) == set(OBJECT_NAMES), 'unexpected CPU308 build scope')
    old_paths = {build['objects'][name]['path']: name for name in OBJECT_NAMES}
    need(len(old_paths) == 32 and all(PurePosixPath(path).is_absolute() and path.endswith('.o') for path in old_paths)
         and all(build['objects'][name]['path'] == (ORIGIN/'B/objects'/(name+'.o')).as_posix() for name in CHANGED),
         'ambiguous CPU308 object identity')
    original = command(build, 'B-link')
    object_order = [path for path in original if path in old_paths]
    need(len(object_order) == 32 and set(object_order) == set(old_paths)
         and object_order == [arg for arg in original if arg.endswith('.o')]
         and original[original.index('-o')+1] == build['library']['path'], 'CPU308 linker object order differs')
    updated = []
    for arg in original:
        name = old_paths.get(arg)
        updated.append((NATIVE/(name+'.o')).as_posix() if name in CHANGED else arg)
    updated[updated.index('-o')+1] = (NATIVE/'libGLM_fresh_kv_pack.so').as_posix()
    # The new object follows all 32 callers and precedes their provider libraries.
    index = original.index(object_order[-1]) + 1
    need(all(not arg.startswith(('-l', '-Wl,-l')) and not re.search(r'\.so(?:\.\d+)*$', arg)
             for i, arg in enumerate(original[:index]) if i != original.index('-o')+1),
         'original providers precede the last object')
    updated.insert(index, kernel['path'])
    host = {}
    for name, leaf in CHANGED.items():
        prior = command(build, f'B-{name}-compile')
        md = prior.index('-MD')
        src = prior[prior.index('-c')+1]
        need(prior[0] == '/usr/bin/g++' and prior[prior.index('-o')+1] == build['objects'][name]['path']
             and src.startswith((ORIGIN/'B/project-mirror/').as_posix()+'/'), 'original changed unit escapes CPU308 mirror')
        mounted = (NATIVE/'src'/leaf).as_posix()
        prefix = [*prior[:md], '-iquote', str(PurePosixPath(src).parent)]
        object_path = (NATIVE/(name+'.o')).as_posix()
        deps = build['fresh_M'][name]
        need(deps['pins_file'] == (ORIGIN/'B'/(name+'.dependency-pins.json')).as_posix(),
             'CPU308 dependency receipt moved')
        host[name] = dict(original_source=src, generated_source=mounted, original_dep_receipt=deps['pins_file'],
                          original_dep_SHA256=deps['pins_SHA256'], generated_source_SHA256=source_pins[leaf],
                          required_generated_includes={n: source_pins[n] for n in SOURCE_DELTA[name]},
                          preprocess=[*prefix, '-M', '-MF', str(NATIVE/(name+'.before.d')), '-MT', object_path, mounted],
                          compile=[*prefix, '-MD', '-MF', str(NATIVE/(name+'.actual.d')), '-MT', object_path,
                                   '-c', mounted, '-o', object_path])
    return dict(caller_publication=CALLER_PUBLICATION, original_step='1103606.308', kernel=kernel,
                host=host, reuse={name: obj for name,obj in build['objects'].items() if name not in CHANGED},
                link=updated, linked_library=updated[updated.index('-o')+1],
                original_library=build['library'],
                planned=dict(fresh_M=2, HOST_compile=2, GPU_compile=0, reused_objects=30, link_objects=33, links=1),
                actual_compiles=0, actual_links=0, GPU_GO=False, Model_GO=False)


def dependencies(raw, object_path):
    need(0 < len(raw) <= 2*1024**2, 'fresh M/MD exceeds its read bound')
    head, sep, files = raw.decode('utf8').replace('\\\n', ' ').partition(':')
    paths = set(shlex.split(files))
    need(sep and head.strip() == object_path and paths and all(path.startswith('/') for path in paths),
         'unexpected fresh M/MD target or path')
    return paths


def check_host_graph(prior, actual, unit, allowed_system):
    expected = dict(prior)
    need(expected.pop(unit['original_source'], None) is not None, 'old source missing from original graph')
    expected[unit['generated_source']] = unit['generated_source_SHA256']
    expected.update({(NATIVE/'src'/name).as_posix(): pin for name,pin in unit['required_generated_includes'].items()})
    need(expected.items() <= actual.items(), 'original or generated dependency differs')
    need(all(allowed_system.get(path) == actual[path] for path in actual.keys()-expected.keys()),
         'fresh M added unregistered dependency')


def prepare(args):
    proof = json.loads(checked(args.kernel_reconciliation, args.kernel_reconciliation_sha256, 16384))
    worker = json.loads(checked(KERNEL/'CPU-result.json', proof['receipt_SHA256'], 1024**2))
    kernel = compiled_kernel(proof, worker)
    checked(kernel['path'], kernel['SHA256'])
    build = json.loads(checked(ORIGIN/'build-result.json', ORIGIN_BUILD_SHA, 1024**2))
    checked(build['library']['path'], ORIGIN_LIBRARY_SHA)
    files, pins = caller_sources()
    recipe = make_recipe(build, pins, kernel)
    for entry in recipe['reuse'].values():
        need(len(checked(entry['path'], entry['SHA256'])) == entry['bytes'], 'original object size differs')
    system = CALLER.parent/'depth-stratified-observer-CPU-v1/PLAN.json'
    plan = json.loads(checked(system, SYSTEM_PLAN_SHA, 16*1024**2))
    recipe.update(pinned_system_header_map=plan['system_pins'], expected_providers=plan['original169']['providers'])
    target = Path(NATIVE)
    target.mkdir(mode=0o700)
    (target/'src').mkdir(mode=0o700)
    for name, raw in files.items():
        # Keep the declaration on the original CPU308 include path.
        if name != 'glm_moe_dsa_attention_prepare.hpp':
            with (target/'src'/name).open('xb') as handle:
                handle.write(raw)
    raw = (json.dumps(recipe, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with (target/'BUILD_RECIPE.json').open('xb') as handle:
        handle.write(raw)
    return dict(recipe_SHA256=sha256(raw).hexdigest(), **recipe['planned'])


def verify_dependencies(recipe_sha, phase):
    recipe = json.loads(checked(NATIVE/'BUILD_RECIPE.json', recipe_sha, 2*1024**2))
    result = {}
    for name, unit in recipe['host'].items():
        original = json.loads(checked(unit['original_dep_receipt'], unit['original_dep_SHA256'], 2*1024**2))
        build_unit = NATIVE/(name+'.o')
        with Path(NATIVE/(name+'.before.d')).open('rb') as source:
            fresh = dependencies(source.read(2*1024**2+1), str(build_unit))
        if phase == 'actual':
            with Path(NATIVE/(name+'.actual.d')).open('rb') as source:
                actual = dependencies(source.read(2*1024**2+1), str(build_unit))
            need(actual == fresh, 'actual MD differs from the fresh M')
        expected = {**recipe['pinned_system_header_map'], **original,
                    unit['generated_source']: unit['generated_source_SHA256'],
                    **{str(NATIVE/'src'/leaf): pin for leaf,pin in unit['required_generated_includes'].items()}}
        need(fresh <= expected.keys(), 'fresh M registered an unexpected physical path')
        seen = {path: sha256(checked(path, expected[path])).hexdigest() for path in fresh}
        check_host_graph(original, seen, unit, recipe['pinned_system_header_map'])
        result[name] = dict(paths=len(seen), SHA256=sha256(json.dumps(seen, sort_keys=True).encode()).hexdigest())
    return dict(phase=phase, recipe_SHA256=recipe_sha, hosts=result)


def symbol_map(raw):
    return {parts[-1]: parts[-2] for line in raw.splitlines()
            if len(parts := line.split()) >= 3 and parts[-2] in {'T', 'D', 'B', 'R'}}


def needed_libraries(raw):
    need('SYMBOLIC' in raw, 'original SYMBOLIC link contract is missing')
    return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]', raw)


def provider_paths(raw):
    need('not found' not in raw and 'undefined symbol' not in raw, 'candidate Native provider is unresolved')
    return set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)', raw, re.M))


def linked_scope(reference_symbols, actual_symbols, reference_needed, actual_needed, expected_providers, actual_providers):
    need(len(reference_symbols) == 281 and not (set(EXPORTS) & reference_symbols.keys())
         and actual_symbols == {**reference_symbols, **dict.fromkeys(EXPORTS, 'T')}, 'Native strong export scope differs')
    need(reference_needed == actual_needed, 'Native NEEDED entries differ from CPU308')
    need(len(expected_providers) == 71 and expected_providers == actual_providers, 'Native provider identity differs')


def verify_link(recipe_sha):
    verified = verify_dependencies(recipe_sha, 'actual')
    recipe = json.loads(checked(NATIVE/'BUILD_RECIPE.json', recipe_sha, 2*1024**2))
    reference, candidate = recipe['original_library']['path'], recipe['linked_library']
    need(reference == str(ORIGIN/'B/libGLM_depth_observer_B.so')
         and candidate == str(NATIVE/'libGLM_fresh_kv_pack.so'), 'fixed Native paths differ')
    checked(reference, ORIGIN_LIBRARY_SHA)
    with Path(candidate).open('rb') as binary:
        native = binary.read(16*1024**2+1)
    need(native.startswith(b'\x7fELF') and 0 < len(native) <= 16*1024**2, 'expected a bounded native ELF')

    def query(argv):
        raw = subprocess.check_output(argv, timeout=30)
        need(len(raw) <= 2*1024**2, 'Native metadata exceeded its output bound')
        return raw.decode('utf8')

    reference_symbols, candidate_symbols = [symbol_map(query(['/usr/bin/nm', '-D', '--defined-only', path]))
                                             for path in (reference, candidate)]
    reference_needed, candidate_needed = [needed_libraries(query(['/usr/bin/readelf', '-d', path]))
                                           for path in (reference, candidate)]
    expected = recipe['expected_providers']
    paths = provider_paths(query(['/usr/bin/ldd', '-r', candidate]))
    need(paths == expected.keys(), 'Native passive provider path closure differs')
    actual = {}
    for path in sorted(paths):
        digest = sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
        actual[path] = digest.hexdigest()
    linked_scope(reference_symbols, candidate_symbols, reference_needed, candidate_needed, expected, actual)
    return dict(verified, library_SHA256=sha256(native).hexdigest(), library_bytes=len(native),
                strong_exports=len(candidate_symbols), providers=len(actual), explicit_Runtime_calls=0,
                implicit_ldd_initialization='UNKNOWN', GPU_GO=False, Model_GO=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    render = modes.add_parser('prepare')
    render.add_argument('--kernel-reconciliation', type=Path, required=True)
    render.add_argument('--kernel-reconciliation-sha256', required=True)
    deps = modes.add_parser('verify-dependencies')
    deps.add_argument('--recipe-sha256', required=True)
    deps.add_argument('--phase', choices=('fresh', 'actual'), required=True)
    linked = modes.add_parser('verify-link')
    linked.add_argument('--recipe-sha256', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare(args)
    elif args.mode == 'verify-dependencies':
        result = verify_dependencies(args.recipe_sha256, args.phase)
    else:
        result = verify_link(args.recipe_sha256)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
