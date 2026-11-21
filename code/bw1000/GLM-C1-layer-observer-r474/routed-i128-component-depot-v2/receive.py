"""Install the exact GLM depot and invoke the selected sealed phase on the original node."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
DEPOT = ROOT / 'incoming/GLM-I128-verified-component-depot-v2'
CONTROL = 'tools/control_glm_i128_component.py'
INDEX = 'coordination/tuning/glm-i128-component-controller-files-v1.json'
SCOPES = {
    'link': ('292132bded88250dd43e66708049793c', 'CPU_GO'),
    'component': ('6f808bade51aded8234e2dbaee8b2af5', 'GPU_GO'),
}


def run(payload):
    mode, files = payload['mode'], payload['files']
    raw_claim = payload['claim']
    claim = json.loads(raw_claim)
    request, grant = SCOPES[mode]
    if (claim['request'], claim['mode'], claim['parent'], claim['node']) != (
            request, mode, '1103606', 'f13r1n08') or claim[grant] is not True:
        raise ValueError('selected phase does not match the authorized original scope')
    if set(files) != set(claim['depot']) or DEPOT.parent.resolve(strict=True) != DEPOT.parent:
        raise ValueError('unexpected installed source set')
    for name, data in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or sha256(data).hexdigest() != claim['depot'][name]:
            raise ValueError('original source member differs')
    if mode == 'link':
        DEPOT.mkdir(mode=0o700)
        for name, data in files.items():
            path = DEPOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
    else:
        if DEPOT.resolve(strict=True) != DEPOT:
            raise ValueError('original installed source depot is not canonical')
        for name, data in files.items():
            path = DEPOT / name
            if path.resolve(strict=True) != path or path.read_bytes() != data:
                raise ValueError('depot source changed since the original link')
    cmd = [sys.executable, '-B', str(DEPOT / CONTROL), '--mode', mode, '--publication', payload['publication'],
           '--manifest-sha256', claim['depot'][INDEX]]
    if mode == 'link':
        audits = payload['original_audits']
        if set(audits) != {'gate', 'down'}:
            raise ValueError('two independently verified compiler results required')
        for arm, data in audits.items():
            if sha256(data).hexdigest() != claim['audit_SHA256_by_arm'][arm]:
                raise ValueError('original compile audit pin differs')
            path = DEPOT / (arm + '-audit.json')
            with path.open('xb') as stream:
                stream.write(data)
            cmd += ['--' + arm + '-audit', str(path), '--' + arm + '-sha256', sha256(data).hexdigest()]
    else:
        if payload['link_sha'] != claim['link_report_SHA256']:
            raise ValueError('original link report SHA differs')
        cmd += ['--link-sha256', payload['link_sha']]
    return subprocess.run(cmd, timeout=330 if mode == 'link' else 445).returncode


if __name__ == '__main__':
    raise SystemExit(run(PAYLOAD))
