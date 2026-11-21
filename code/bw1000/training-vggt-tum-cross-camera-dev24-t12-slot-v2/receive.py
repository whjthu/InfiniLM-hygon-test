"""Install the pinned scoring controller once and return its original terminal fields."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'training-vggt-tum-cross-camera-dev24-t12-slot-v2'
REQUEST = '744b6d8b851044debb2eae5500eac547'
STAGE_SHA = 'd221478fe51e14ccdf8ca858632c60aee67993e693c58f29b2c9357d373dc903'
INDEX = 'coordination/training/vggt-tum-cross-camera-controller-files-v1.json'
STAGED = ROOT / 'results/training-vggt-tum-cross-camera-dev24-v1-staged'


def run(payload):
    sources, context = payload['sources'], payload['context']
    if (context['request'], context['label'], context['stage_SHA256']) != (REQUEST, LABEL, STAGE_SHA):
        raise ValueError('unexpected original scoring request')
    install = ROOT / 'incoming' / LABEL
    if install.parent.resolve(strict=True) != install.parent or STAGED.resolve(strict=True) != STAGED:
        raise ValueError('aliased source or staging directory')
    if any((STAGED / name).exists() or (STAGED / name).is_symlink()
           for name in ('score-intent.json', 'before.json', 'score-outcome.json', 'score')):
        raise FileExistsError('original scoring namespace has a prior attempt')
    raw_index = sources[INDEX]
    if sha256(raw_index).hexdigest() != context['source_index_SHA256']:
        raise ValueError('unexpected source manifest')
    index = json.loads(raw_index)
    if len(index) != 15 or set(sources) != {*index, INDEX}:
        raise ValueError('unexpected source members')
    for name, data in sources.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not 0 < len(data) <= 2 * 1024**2:
            raise ValueError('unexpected source artifact')
        if name != INDEX and sha256(data).hexdigest() != index[name]:
            raise ValueError('transmitted source differs')
    install.mkdir(mode=0o700)
    for name, data in sources.items():
        path = install / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    command = [sys.executable, '-B', str(install / 'tools/control_vggt_tum_cross_camera_dev.py'),
               str(STAGED), '--stage-sha', STAGE_SHA, '--request', REQUEST,
               '--publication', context['publication'], '--manifest-sha', context['source_index_SHA256']]
    return subprocess.run(command, timeout=1510).returncode


if __name__ == '__main__':
    raise SystemExit(run(INPUT))
