"""Create the one-TU build stage from exact repo bytes, without target calls."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PARENT = BASE.parent / 'Deep-C4-constant-slot-metadata-r482/Native-CPU-v1'
STAGE = '/refactor/incoming/Deep-CSA-identity-SWA-commit-CPU-v1'
FROZEN = {
    'csa-identity-swa-commit-source-v1/render.py':
        '27551623e83e6f68b038174fa649e037b2f7ef1cd8f4e4073f6ed91660796ad8',
    'csa-identity-swa-commit-source-v1/SOURCE_FILES.json':
        'd299540bc8d2b101ef2677c715628d7d57bc17dcf87fc1c56d24295b00487c10',
    'csa-identity-swa-commit-build-plan-v1/tu07_plan.py':
        '9c581edf061ab3536450fd1c13840e91c3d35f87a95297db6fd3572c8446207e',
}
TARGET_MF = '58fe99c729dc44509d5e32f12ce03760875815087982732a932d176f25406ea6'


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f'expected frozen bytes: {path.name}')
    return raw


def prepare(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for name, pin in FROZEN.items():
        pinned(BASE / name, pin)
    old_env = pinned(PARENT / 'CPU_inside.sh', 'd2b7f8d3370d5f5427b9e8d04324642d560305bad971fe429ab0372ad2cc250a')
    prefix, old_exec = old_env.rstrip(b'\n').rsplit(b'\n', 1)
    if not old_exec.endswith(b'/Deep-C4-constant-slot-metadata-r482/Native-CPU-v1/build_native.py'):
        raise ValueError('original final invocation differs')
    files = {
        'compile_once.py': (HERE / 'compile_once.py').read_bytes(),
        'tu07_plan.py': pinned(BASE / 'csa-identity-swa-commit-build-plan-v1/tu07_plan.py',
                              FROZEN['csa-identity-swa-commit-build-plan-v1/tu07_plan.py']),
        'TOOLCHAIN.json': pinned(PARENT / 'TOOLCHAIN.json',
                                 '4bbb8cd130125f47cec62c943f4c4ade5980e288efab5d5c9116f1de452dd8a0'),
        'ORIGINAL_ENV.sh': old_env,
        'inside.sh': prefix + f'\nexport LC_ALL=C\nexec /usr/bin/python3 -I -S -B -u {STAGE}/compile_once.py\n'.encode(),
    }
    with TemporaryDirectory(dir=output.parent, prefix='deep-csa-cpu-') as temporary:
        target = Path(temporary) / 'stage'
        target.mkdir()
        native = target / 'native-src'
        render = BASE / 'csa-identity-swa-commit-source-v1/render.py'
        subprocess.run([sys.executable, '-I', '-S', '-B', str(render), str(native)],
                       stdout=subprocess.DEVNULL, check=True, timeout=30)
        source_mf = pinned(native / 'SOURCE_FILES.json', TARGET_MF)
        members = json.loads(source_mf)['files_SHA256']
        for name, pin in members.items():
            files['native-src/' + name] = pinned(native / name, pin)
        files['native-src/SOURCE_FILES.json'] = source_mf
        manifest = (json.dumps({'files_SHA256': {path: digest(raw) for path, raw in files.items()}},
                               sort_keys=True, indent=2) + '\n').encode()
        for name, raw in files.items():
            if not name.startswith('native-src/'):
                (target / name).write_bytes(raw)
        (target / 'STAGE_FILES.json').write_bytes(manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        target.rename(output)
    return dict(stage_manifest_SHA256=digest(manifest), source_manifest_SHA256=TARGET_MF, members=len(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_output', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.new_output.expanduser().absolute()), sort_keys=True))


if __name__ == '__main__':
    main()
