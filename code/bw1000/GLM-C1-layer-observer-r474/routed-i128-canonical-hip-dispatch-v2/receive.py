"""Install the byte-exact corrected component depot and invoke it once."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
DEPOT = ROOT / 'incoming/GLM-I128-canonical-HIP-depot-v3'
CONTROL = 'tools/control_glm_i128_component.py'
INDEX = 'coordination/tuning/glm-i128-component-controller-files-v1.json'
LABEL = 'GLM-I128-four-routed-variants-canonical-HIP-v2'
LINK_SHA = '3afb5d888ea0fb74aefeded292469c3167ded68ed5c2da1a39497e2e85f05ce4'


def run(payload):
    claim = json.loads(payload['claim'])
    files = payload['files']
    request = sha256((LABEL + ':' + claim['caller_manifest_SHA256']).encode()).hexdigest()[:32]
    if (claim['request'], claim['label'], claim['parent'], claim['node'], claim['GPU_GO'],
            claim['replay_allowed'], claim['link_report_SHA256']) != (
            request, LABEL, '1103606', 'f13r1n08', True, False, LINK_SHA):
        raise ValueError('the revised scope does not match its claim')
    if (set(files) != set(claim['depot']) or DEPOT.parent.resolve(strict=True) != DEPOT.parent
            or (ROOT / 'results' / LABEL).exists()):
        raise ValueError('the original execution namespace may already exist')
    for name, raw in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or sha256(raw).hexdigest() != claim['depot'][name]:
            raise ValueError('published source member differs')
    DEPOT.mkdir(mode=0o700)
    for name, raw in files.items():
        path = DEPOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    return subprocess.run([sys.executable, '-B', str(DEPOT / CONTROL), '--mode', 'component',
                           '--publication', payload['publication'], '--manifest-sha256', claim['depot'][INDEX],
                           '--link-sha256', LINK_SHA], timeout=445).returncode


if __name__ == '__main__':
    raise SystemExit(run(PAYLOAD))
