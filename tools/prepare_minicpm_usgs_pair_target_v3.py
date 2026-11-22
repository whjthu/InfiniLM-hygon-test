"""Verify and stage the MiniCPM domain pair on the authenticated Hygon host."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
RESULTS = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
LABEL = 'training-minicpm-usgs-exposure25-resume20-target-v2'
REQUEST = 'fa68d03ea04f4588ac9c21459097d50e'
CORE_NAME = 'training-minicpm-usgs-exposure25-resume20-hip-v1'
CORE = REPO / 'code/bw1000' / CORE_NAME
CORE_SHA = 'dcf5356c2879bb2c61b1de1f0f037d8fd312542775e46e6556c6f6d89dc0ee82'
CORE_ASSETS_SHA = '16bd66c3df682fcaa3ea6cc0f66739c968b819a37d6511e97a50cd7ab03f83e7'
FRAMEWORK_SHA = 'e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41'
MODEL_SHA = 'ddcb8756dc66474f348bd81bb1730874ba78af8b46567e6a99596cedf5594ae6'
VIEW_SHA = '20cfbdb6f51211692ee68c70b925e7323c0fbe7304f509a588740300deeb120b'
CPU_LABEL = 'training-minicpm-usgs-native88-CPU-v1'
CPU_REQUEST = '7f9a0e2809bf4f53a7294fdc3dc74225'
PARENT, NODE, PORT = '1103606', 'f13r1n19', 29929
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def stable_file(path, *, size=None, digest=None, ceiling=None, collect=False):
    need(not path.is_symlink() and path.resolve(strict=True) == path.absolute(), 'asset has an aliased path')
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    before_path = path.lstat()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before_path.st_mode) and stat.S_ISREG(before.st_mode) and (size is None or before.st_size == size)
             and (ceiling is None or 0 < before.st_size <= ceiling), 'asset size or file type differs: ' + path.name)
        hasher = sha256()
        blocks = []
        remaining = before.st_size
        while remaining:
            block = stream.read(min(remaining, 2 * 1024**2))
            need(block, 'original file became shorter: ' + path.name)
            hasher.update(block)
            if collect:
                blocks.append(block)
            remaining -= len(block)
        need(not stream.read(1), 'original file became longer: ' + path.name)
        after = os.fstat(stream.fileno())
    identity = tuple(getattr(before, key) for key in fields)
    path_identity = tuple(getattr(before_path, key) for key in fields)
    # Windows fstat does not expose the same inode/device fields as lstat.
    need(identity == tuple(getattr(after, key) for key in fields)
         and path_identity == tuple(getattr(path.lstat(), key) for key in fields)
         and (os.name == 'nt' or identity == path_identity),
         'asset changed during read: ' + path.name)
    actual = hasher.hexdigest()
    need(digest is None or actual == digest, 'asset digest differs: ' + path.name)
    return b''.join(blocks) if collect else dict(bytes=before.st_size, SHA256=actual, identity=list(path_identity))


def small(path, digest=None):
    return stable_file(path, digest=digest, ceiling=2 * 1024**2, collect=True)


def write(path, body):
    if isinstance(body, dict):
        body = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())
    return dict(path=str(path), bytes=len(body), SHA256=sha256(body).hexdigest())


def verify_native_cpu(raw, wanted, accounting, worker_raw):
    from minicpm_domain_recipe import require_native88

    receipt = require_native88(raw, wanted)
    handle = receipt['actual_CPU_step']
    need(receipt['request'] == CPU_REQUEST and re.fullmatch(PARENT + r'\.\d+', handle)
         and re.fullmatch('[0-9a-f]{40}', receipt['publication']) and receipt['actual_GPU_allocation'] == 0
         and receipt['actual_optimizer_updates'] == 0 and receipt['model_loaded'] is False
         and receipt['EOS_supervised'] == receipt['full_untruncated'] == 88,
         'original CPU88 receipt identity differs')
    lines = [line.split('|') for line in accounting.decode().splitlines() if line.strip()]
    need(all(len(row) == 7 for row in lines), 'unexpected Slurm accounting columns')
    matches = [row for row in lines if row[0] == handle or row[1] == CPU_LABEL]
    need(len(matches) == 1, 'expected exact original CPU88 accounting row')
    row = matches[0]
    tres = dict(item.split('=', 1) for item in row[6].split(','))
    need(row[:4] == [handle, CPU_LABEL, 'COMPLETED', '0:0'] and row[4].isdecimal()
         and int(row[4]) == receipt['actual_step_elapsed_seconds'] and row[5] == NODE
         and len(tres) == len(row[6].split(',')) and tres.get('cpu') == '8' and tres.get('node') == '1'
         and tres.get('mem') in ('12G', '12288M') and not any(key.startswith('gres/') for key in tres),
         'original CPU88 step did not meet its zero-GPU resource and terminal contract')
    worker = json.loads(worker_raw)
    need(receipt['metadata_pins']['worker-result.json'] == dict(bytes=len(worker_raw), SHA256=sha256(worker_raw).hexdigest())
         and worker['source_dataset_SHA256'] == '0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62'
         and worker['passed'] is True and worker['completed_rows'] == 88,
         'original CPU88 worker did not validate the intended train-only inputs')
    return dict(step=handle, accounting=row, independent_SHA256=wanted,
                worker_SHA256=sha256(worker_raw).hexdigest(), publication=receipt['publication'])


def stage_framework(base, run):
    from mid_epoch_recovery import candidate_sources

    files = json.loads(small(base / 'source/FRAMEWORK_FILES.json', FRAMEWORK_SHA))['files']
    need(len(files) == 393, 'expected the full common framework source tree')
    original = base / 'framework'
    target = run / 'framework'
    patches = candidate_sources(original)
    need(set(patches) == {'vlm/runtime/resume_cursor.py', 'vlm/runtime/resume_witness.py'}, 'unexpected source patch')
    pinned = {}
    for name, row in sorted(files.items()):
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and '\\' not in name, 'unsafe framework member')
        path = original / name
        raw = stable_file(path, size=row['bytes'], digest=row['SHA256'], ceiling=2*1024**2, collect=True)
        need(row['mode'] in (0o644, 0o755) and stat.S_IMODE(path.stat().st_mode) == row['mode'],
             'frozen framework source mode differs')
        member = target / name
        member.parent.mkdir(parents=True, exist_ok=True)
        pin = write(member, patches.get(name, raw))
        member.chmod(row['mode'])
        pinned[name] = {k: pin[k] for k in ('bytes', 'SHA256')}
    return dict(files=393, base_manifest_SHA256=FRAMEWORK_SHA,
                patched={name: pinned[name] for name in patches},
                manifest=write(run / 'framework-installed-files.json', pinned))


def prepare(native_path, native_sha, accounting):
    run = RESULTS / LABEL
    need(not run.exists() and not run.is_symlink(), 'original staging namespace already exists')
    core_raw = small(CORE / 'SOURCE_FILES.json', CORE_SHA)
    files = json.loads(core_raw)['files']
    need(len(files) == 12 and all(Path(name).name == name and '\\' not in name for name in files), 'core membership differs')
    core = {name: stable_file(CORE / name, size=row['bytes'], digest=row['SHA256'], ceiling=2*1024**2, collect=True)
            for name, row in files.items()}
    need(sha256(core['ASSETS.json']).hexdigest() == CORE_ASSETS_SHA, 'core asset binding differs')
    sys.path.insert(0, str(CORE))
    from data_view import build
    from minicpm_domain_recipe import command_plan

    native_raw = small(native_path, native_sha)
    native = verify_native_cpu(native_raw, native_sha, accounting, small(RESULTS / CPU_LABEL / 'worker-result.json'))
    assets = json.loads(core['ASSETS.json'])
    data = RESULTS / 'training-domain-usgs-table-reading-v1/data/train'
    source = data / 'swift.jsonl'
    source_raw = stable_file(source, size=29658, digest=assets['dataset_SHA256'], ceiling=2*1024**2, collect=True)
    rows = [json.loads(line) for line in source_raw.splitlines()]
    media = {row['images'][0] for row in rows if len(row['images']) == 1}
    expected = {str(data / 'media' / name) for name in assets['media'] if re.fullmatch(r'table-[0-9]{3}\.png', name)}
    need(len(rows) == len({row['id'] for row in rows}) == 88 and len(media) == len(expected) == 22
         and media == expected and all(len(row['images']) == 1 for row in rows), 'target train-only media scope differs')
    for name, row in assets['media'].items():
        stable_file(data / 'media' / name, size=row['bytes'], digest=row['SHA256'])
    initial = assets['initial']
    init_file = stable_file(Path(initial['path']), size=initial['bytes'], digest=initial['SHA256'])
    model = RESULTS / 'training-minicpm-original-model-v1/MiniCPM-V-4.6'
    manifest = REPO / 'code/bw1000/training-minicpm-final-tail-r2/MODEL_FILES.json'
    weights = json.loads(small(manifest, MODEL_SHA))['files']
    need(len(weights) == 10 and 'model.safetensors' in weights and all(Path(name).name == name for name in weights),
         'original ten-file model manifest differs')
    model_pins = {name: stable_file(model / name, size=row['bytes'], digest=row['SHA256']) for name, row in weights.items()}
    need(shutil.disk_usage(RESULTS).free >= 16 * 1024**3, 'insufficient workspace for original checkpoints')
    run.mkdir(mode=0o700)
    src = run / 'src'
    src.mkdir(mode=0o700)
    for name, content in {**core, 'SOURCE_FILES.json': core_raw}.items():
        write(src / name, content)
    view = build(source, run / 'view')
    need(view['view']['SHA256'] == VIEW_SHA and view['physical_rows'] == 800, 'target view differs from original local iteration')
    base = RESULTS / 'training-qwen-a68952c-swift-initial-CPU-v1'
    framework = stage_framework(base, run)
    plans = {}
    for arm in ('continuous', 'resumed'):
        plan = command_plan(run / 'framework/tools/hygon_vlm_recipe.py', core['original-launch.json'], run, VIEW_SHA, arm, PORT)
        need(plan['command'][0] == str(base / 'framework/vlm/runtime/swift_entry.py'), 'shared entrypoint identity differs')
        plan['command'][0] = str(run / 'framework/vlm/runtime/swift_entry.py')
        need(all(Path(file).is_file() for file in plan['required_assets']['plugins']), 'native plugin missing')
        plans[arm] = write(run / (arm + '-plan.json'), plan)
    independent = write(run / 'native88-independent.json', native_raw)
    write(run / 'native88-accounting.psv', accounting)
    staged = dict(state='MINICPM_PAIR_TARGET_ASSETS_STAGED', request=REQUEST, parent=PARENT, node=NODE, port=PORT,
                  native88=native, native88_receipt=independent, plans=plans, framework=framework, view=view,
                  source_rows=88, train_media=22, model_files=model_pins, initial=init_file,
                  source_core_manifest_SHA256=CORE_SHA, asset_binding_SHA256=CORE_ASSETS_SHA,
                  actual_optimizer_updates=0, actual_GPU_steps=0, device_admitted=False, training_READY=False)
    pending = run / 'asset-stage-complete.pending'
    receipt = write(pending, staged)
    pending.replace(run / 'asset-stage-complete.json')
    return dict(state=staged['state'], receipt_SHA256=receipt['SHA256'], native88_step=native['step'], model_files=10,
                framework_files=393, train_rows=88, media=22, physical_exposures=800, actual_GPU_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native88', type=Path, required=True)
    parser.add_argument('--native88-sha256', required=True)
    args = parser.parse_args()
    need(re.fullmatch('[0-9a-f]{64}', args.native88_sha256), 'expected native88 SHA-256')
    def timeout(*_):
        raise TimeoutError('target asset staging exceeded its original budget')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(240)
    try:
        raw = subprocess.check_output(['sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-19T00:00:00', '-o', COLUMNS], timeout=15)
        need(0 < len(raw) <= 2*1024**2, 'unexpected accounting output size')
        print(json.dumps(prepare(args.native88.absolute(), args.native88_sha256, raw), allow_nan=False))
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
