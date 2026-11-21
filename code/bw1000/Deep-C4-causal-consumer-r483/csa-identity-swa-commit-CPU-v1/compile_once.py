"""Build one DeepSeek HOST object in a claimed zero-GRES step."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from types import ModuleType


HERE = Path(__file__).resolve().parent
PLAN_SHA = '9c581edf061ab3536450fd1c13840e91c3d35f87a95297db6fd3572c8446207e'
TOOLCHAIN_SHA = '4bbb8cd130125f47cec62c943f4c4ade5980e288efab5d5c9116f1de452dd8a0'
ENV_SHA = 'd2b7f8d3370d5f5427b9e8d04324642d560305bad971fe429ab0372ad2cc250a'
TOOLS = ('/usr/bin/python3', '/usr/bin/g++', '/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus',
         '/usr/lib/gcc/x86_64-linux-gnu/11/collect2', '/usr/bin/as', '/usr/bin/ld',
         '/usr/bin/nm', '/usr/bin/readelf', '/opt/dtk/env.sh')
MASKS = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')
UNSET = ('LD_PRELOAD', 'LD_AUDIT', 'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'LIBRARY_PATH', 'CPATH',
         'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'PYTHONPATH')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, limit, wanted=None):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (wanted is None or sha256(raw).hexdigest() == wanted),
            f'frozen input bytes differ: {Path(path).name}')
    return raw


def stage(manifest_sha):
    names = {'compile_once.py', 'tu07_plan.py', 'TOOLCHAIN.json', 'ORIGINAL_ENV.sh', 'inside.sh',
             'native-src/SOURCE_FILES.json', 'native-src/SOURCE_PROOF.json',
             'native-src/deepseek_v4_csa_attention_batch_r784.cpp', 'native-src/identity_swa_commit.hpp'}
    manifest = json.loads(read(HERE / 'STAGE_FILES.json', 8192, manifest_sha))['files_SHA256']
    require(manifest.keys() == names, 'staged build inventory differs')
    for item, pin in manifest.items():
        read(HERE / item, 256 * 1024, pin)
    raw = read(HERE / 'tu07_plan.py', 64 * 1024, PLAN_SHA)
    recipe = ModuleType('frozen_tu07_plan')
    exec(compile(raw, str(HERE / 'tu07_plan.py'), 'exec'), recipe.__dict__)
    return recipe


def toolchain(recipe):
    chain = json.loads(read(HERE / 'TOOLCHAIN.json', 8192, TOOLCHAIN_SHA))
    read(HERE / 'ORIGINAL_ENV.sh', 8192, ENV_SHA)
    for name, expected in chain['required_environment'].items():
        require(os.environ.get(name) == expected, f'original environment differs: {name}')
    for name in TOOLS:
        pin = chain['tools'][name]
        require(str(Path(name).resolve(strict=True)) == pin['canonical'] and recipe.file_sha(name) == pin['sha256'],
                f'original tool bytes differ: {name}')
    for name in ('g++', 'as', 'ld'):
        found = shutil.which(name)
        require(found and str(Path(found).resolve(strict=True)) == chain['tools']['/usr/bin/' + name]['canonical'],
                f'PATH resolves a different tool: {name}')


def clock_and_placement():
    env = os.environ
    epoch, deadline = float(env['TUNING_CPU_EPOCH']), float(env['TUNING_CPU_DEADLINE'])
    now = time.time()
    require(math.isfinite(epoch) and math.isfinite(deadline) and deadline == epoch + 300
            and 0 < epoch <= now < deadline - 25, 'invalid original wall budget')
    parent, step, node = env['SLURM_JOB_ID'], env['SLURM_STEP_ID'], env['SLURMD_NODENAME']
    require(parent in {'1087656', '1103606'} and parent == env['TUNING_CPU_PARENT']
            and step.isdecimal() and re.fullmatch(r'[a-zA-Z0-9-]+', node) and node == env['TUNING_CPU_NODE']
            and env.get('SLURM_CPUS_PER_TASK') == '4'
            and not env.get('SLURM_STEP_GPUS') and env.get('SLURM_STEP_GRES', '') in ('', 'none')
            and all(env.get(key) == '-1' for key in MASKS)
            and all(key not in env for key in UNSET), 'expected the claimed four-CPU zero-GRES setup')
    for key in ('TUNING_CPU_CLAIM_SHA', 'TUNING_CPU_STAGE_SHA'):
        require(re.fullmatch(r'[0-9a-f]{64}', env[key]), f'expected an exact digest: {key}')
    end = time.monotonic() + deadline - now - 12

    def left():
        return min(end - time.monotonic(), deadline - time.time() - 12)

    return dict(original_step=parent + '.' + step, node=node, epoch=epoch, deadline=deadline, GRES='none',
                claim_SHA256=env['TUNING_CPU_CLAIM_SHA'], stage_manifest_SHA256=env['TUNING_CPU_STAGE_SHA']), left


def strong_symbols(raw):
    return {symbol: kind for kind, symbol in re.findall(r'(?m)^\s*[0-9a-fA-F]+\s+([TDBR])\s+(\S+)\s*$', raw)}


def needed(raw):
    return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]', raw)


def loader_paths(raw):
    require(not re.search(r'not found|undefined symbol', raw, re.I), 'ELF loader reported unresolved dependencies')
    return set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)', raw, re.M))


def check_elf(run, recipe, plan, prefix, library):
    tags = run(prefix + '-readelf', ['/usr/bin/readelf', '-d', library])
    require(needed(tags) == plan['parent_NEEDED'] and 'SYMBOLIC' in tags, 'ELF dynamic tags differ from original')
    syms = strong_symbols(run(prefix + '-nm', ['/usr/bin/nm', '-D', '--defined-only', library]))
    require(syms and all(syms.get(name) == 'T' for name in plan['parent_required_CABI']),
            'required parent exports are missing')
    paths = loader_paths(run(prefix + '-ldd', ['/usr/bin/ldd', '-r', library]))
    expected = plan['parent_loader_paths_SHA256']
    require(paths == expected.keys() and all(recipe.file_sha(path) == pin for path, pin in expected.items()),
            'ELF loader paths or library bytes differ from original')
    return syms


def build(recipe, plan, result, run):
    result['input_files_verified'] = recipe.verify_inputs(plan)
    parent_symbols = check_elf(run, recipe, plan, 'parent', plan['parent_library'])
    run('fresh-M', plan['preprocess'])
    result['fresh_M_paths'] = recipe.verify_new_depfile(plan, plan['fresh_depfile'])
    result['fresh_M_SHA256'] = recipe.file_sha(plan['fresh_depfile'])
    result['compile_invocations'] = 1
    run('compile', plan['compile'])
    result['actual_MD_paths'] = recipe.verify_new_depfile(plan, plan['actual_depfile'])
    result['actual_MD_SHA256'] = recipe.file_sha(plan['actual_depfile'])
    obj_sha = recipe.file_sha(plan['object'])
    reused = plan['reused_objects_SHA256']
    require(all(recipe.file_sha(path) == pin for path, pin in reused.items()), 'reused object changed before link')
    result['link_invocations'] = 1
    run('link', plan['link'])
    candidate_symbols = check_elf(run, recipe, plan, 'candidate', plan['library'])
    require(candidate_symbols == parent_symbols, 'candidate strong exports differ')
    require(recipe.file_sha(plan['object']) == obj_sha and
            all(recipe.file_sha(path) == pin for path, pin in reused.items()), 'object changed during link')
    result.update(passed=True, object_SHA256=obj_sha, library_SHA256=recipe.file_sha(plan['library']),
                  strong_exports=len(parent_symbols), NEEDED=plan['parent_NEEDED'],
                  loader_paths=len(plan['parent_loader_paths_SHA256']), reused_objects_verified=len(reused))


def save(path, result):
    with path.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def command_runner(output, left, result):
    def run(label, argv):
        seconds = left()
        require(seconds > 3, 'original CPU deadline exceeded before command')
        entry = dict(label=label, argv=argv, returncode=None)
        result['commands'].append(entry)
        with (output / (label + '.stdout')).open('xb') as out, (output / (label + '.stderr')).open('xb') as err:
            child = subprocess.Popen(argv, cwd='/', stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     start_new_session=True)
            try:
                entry['returncode'] = child.wait(timeout=min(120, seconds - 1))
            except BaseException:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=4)
                raise
        raw = []
        for suffix in ('stdout', 'stderr'):
            with (output / (label + '.' + suffix)).open('rb') as stream:
                data = stream.read(2 * 1024**2 + 1)
            require(len(data) <= 2 * 1024**2, f'{label} logs exceed the verification bound')
            raw.append(data)
        entry['log_SHA256'] = [sha256(item).hexdigest() for item in raw]
        require(entry['returncode'] == 0, f'{label} failed with rc {entry["returncode"]}')
        return b'\n'.join(raw).decode('utf8', errors='replace')
    return run


def main():
    identity, left = clock_and_placement()
    require(left() > 10, 'insufficient original CPU time')
    recipe = stage(identity['stage_manifest_SHA256'])
    require(HERE == Path(recipe.STAGED).parent, 'staged build path differs')
    plan = recipe.derive(read(recipe.PARENT_RECEIPT, 2 * 1024**2),
                         read(recipe.STAGED / 'SOURCE_FILES.json', 8192))
    output = Path(recipe.OUTPUT)
    output.mkdir(mode=0o700)
    result = dict(identity, completed=False, passed=False, planned_HOST_TUs=1, planned_GPU_TUs=0,
                  compile_invocations=0, link_invocations=0, actual_GPU_steps=0, explicit_Native_calls=0,
                  weight_reads=0, device_qualified=False, commands=[],
                  parent_receipt_SHA256=plan['parent_receipt_SHA256'], staged_manifest_SHA256=plan['rendered_manifest_SHA256'])
    save(output / 'intent.json', identity)

    def exhausted(*_):
        raise TimeoutError('original CPU worker deadline exceeded')

    signal.signal(signal.SIGALRM, exhausted)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(0.001, left()))
        toolchain(recipe)
        build(recipe, plan, result, command_runner(output, left, result))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result.update(passed=False, error_type=type(error).__name__, error=str(error)[:240])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['completed'] = True
        save(output / 'CPU-result.json', result)
    print(json.dumps({key: result.get(key) for key in ('original_step', 'passed', 'compile_invocations',
          'link_invocations', 'library_SHA256', 'error_type')}, allow_nan=False), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
