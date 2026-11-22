"""Scope and checked source closure for one three-arm, zero-update Slurm step."""

import importlib.util
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re


LABEL = 'training-minicpm-checkpoint20-rng-bracket-target-v1'
REQUEST = '725c35a7012044e1a68c9bed22edcc75'
PARENT, NODE, PORT = '1103606', 'f13r1n19', 29949
RESULTS = PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
RUN = RESULTS / LABEL
ORIGINAL = RESULTS / 'training-minicpm-usgs-exposure25-resume20-target-v2'
AB = {'a1': 460, 'b': 840, 'a2': 1220}
ARM_CLEANUP, ARM_FLOOR = 30, 180
NATIVE, CLEANUP, GPU_CLIENT, WHOLE = 1260, 1360, 1440, 1800
PHASES = {'pre': (0, 2, 2048, 60), 'gpu': (8, 64, 224000, GPU_CLIENT), 'post': (0, 2, 2048, 90)}
PLANNED = 24
FIRST = 'code/bw1000/training-minicpm-checkpoint20-first-forward-target-v1'
FIRST_SHA = 'f8063e2475514d6463d9d5afd48dd6252457334d2a33ccf0490c25c0865c9fb6'
SOURCE = 'code/bw1000/training-minicpm-checkpoint20-rng-bracket-source-v1'
SOURCE_SHA = '90966b2ec210ac68eaceedbeb7cc7cdf59a5396070eec22936097f9ddf37fe64'
TERMINAL = 'results/bw1000/2026-11-22/training-minicpm-original1189-checkpoint20-first-forward-terminal.json'
TERMINAL_SHA = 'a367dcb55e35aed72975f5910e09d2a2d20bbd60f005efdd988bcf48f1d9a105'
ROUTE = 'results/bw1000/2026-11-22/training-minicpm-original1189-first-forward-SDPA-route.json'
ROUTE_SHA = 'db7bcb1609050b843b8260b018e61849f683a91af5991fdb52149141e5156b4a'
POST_ARTIFACT_SHA = 'd4820523482571dec8c46ba61ef067f36172f7befb247823c693a7dd6919b84e'
STAGE_SHA = 'ac4ca58990c955ec99db48b2e4cc8be4dac6e0fdefa59fbee3f5f3bf6784cd6d'
INVENTORY = dict(path=str(ORIGINAL / 'continuous-checkpoint-20-files.json'), bytes=2131,
                 SHA256='c596dd8368f9798c0555e721bd63512f0087dc2546a22588adf128245c96a752')
TARGET_FILES = {'CLAIM.json', 'README.zh-CN.md', 'bracket_contract.py', 'host_once.py', 'worker.py', 'check_local.py'}
STAGED = {'bracket_contract.py', 'worker.py', 'contract.py', 'first_worker.py',
          'compare_bracket.py', 'rng_treatment.py', 'native_rng_observer.py', 'original1189-route.json',
          'original_worker.py', 'snapshot.py', 'owned_processes.py', 'original1171.json'}
OLD_OBSERVER = {'native_observer.py', 'probe.py', 'compare_first.py'}


def need(valid, reason):
    if not valid:
        raise ValueError(reason)


def read(path, digest=None, size=None, cap=2*1024**2):
    need(path.resolve(strict=True) == path.absolute() and path.is_file(), 'unexpected bounded file path')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    need(0 < len(raw) <= cap and (size is None or len(raw) == size)
         and (digest is None or sha256(raw).hexdigest() == digest), 'wrong file identity: ' + path.name)
    return raw


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen(folder, digest, names=None):
    manifest = json.loads(read(folder / 'SOURCE_FILES.json', digest))
    pins = manifest['files']
    if names is not None:
        need(set(pins) == names, 'unexpected target manifest paths')
    need(all(Path(name).name == name for name in pins), 'invalid source member path')
    return {name: read(folder / name, pin['SHA256'], pin['bytes']) for name, pin in pins.items()}


def require_original(terminal_raw, route_raw):
    terminal = json.loads(terminal_raw)
    route = json.loads(route_raw)
    post = terminal['independent_post']
    need(terminal['original']['phases']['gpu'][:4] == ['1103606.1189',
         'training-minicpm-checkpoint20-first-forward-target-v1-gpu', 'COMPLETED', '0:0']
         and terminal['original']['phases']['gpu'][5] == NODE and terminal['checkpoint20'] == '1103606.1171'
         and terminal['checkpoint_inventory_SHA256'] == INVENTORY['SHA256']
         and post['step'] == '1103606.1191' and post['raw_SHA256'] == POST_ARTIFACT_SHA
         and all(post[key] is True for key in ('KFD_clean', 'PID_clean', 'ports_clean'))
         and route['gpu_step'] == '1103606.1189' and route['original_terminal_SHA256'] == TERMINAL_SHA,
         'original same-node checkpoint diagnostic or post gate differs')


def admission(doc):
    need(re.fullmatch('[0-9a-f]{32}', REQUEST) and doc['schema'] == 'MINICPM_RNG_BRACKET_ADMISSION_V1'
         and (doc['request'], doc['parent'], doc['node'], doc['port']) == (REQUEST, PARENT, NODE, PORT)
         and doc['original_gpu_step'] == '1103606.1189' and doc['original_post_step'] == '1103606.1191'
         and doc['original_terminal_SHA256'] == TERMINAL_SHA and doc['stage_SHA256'] == STAGE_SHA
         and doc['checkpoint20_inventory'] == INVENTORY and doc['planned_forwards'] == PLANNED
         and doc['planned_optimizer_updates'] == 0 and re.fullmatch('[0-9a-f]{40}', doc['publication'])
         and doc['root_GO'] is True and doc['GPU_GO'] is True, 'invalid adoption document')


def render(old, plan, arm):
    need(arm in AB, 'unexpected native arm')
    result = old.render(plan, original=ORIGINAL, run=RUN)
    argv = result['command']
    plugin = str(RUN / 'target-worker/native_observer.py')
    train = str(RUN / 'resumed/train')
    need(argv.count(plugin) == argv.count(train) == 1, 'prior exact planned overrides differ')
    argv[argv.index(plugin)] = str(RUN / 'target-worker/native_rng_observer.py')
    argv[argv.index(train)] = str(RUN / arm / 'train')
    result['environment_overrides']['MASTER_PORT'] = str(PORT)
    result['environment_overrides']['MINICPM_RNG_TREATMENT'] = arm
    return result


def observed_ranks(directory):
    return [rank for rank in range(8)
            if all((directory / 'train/minicpm-usgs-witness' / f'rank-{rank:02d}' / member).is_file()
                   for member in ('first-forward-state.json', 'first-forward-ready.json', 'first-forward-stop.json',
                                  'rng-treatment.json', 'rng-state-audit.json'))]
