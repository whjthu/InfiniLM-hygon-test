"""Build the sole ordinary-SSH stdin body; no SSH, imports of native libraries or dispatch."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import time

D = Path(__file__).resolve().parent
G = D.parents[2]


def need(ok, why):
    if not ok:
        raise ValueError(why)


def make_body(publication, request_id, epoch):
    def check():
        need(type(epoch) in (float, int) and math.isfinite(epoch) and
             0 < epoch <= time.time() < epoch + 240, 'same original CPU epoch exhausted')
    check()
    need(re.fullmatch('[0-9a-f]{40}', publication) and re.fullmatch('[0-9a-f]{32}', request_id),
         'published once context required')
    manifest_raw = (D / 'SOURCE_FILES.json').read_bytes()
    manifest = json.loads(manifest_raw)['files']
    need(set(manifest) == {p.name for p in D.iterdir() if p.is_file() and p.name != 'SOURCE_FILES.json'},
         'source inventory differs')
    package = {'SOURCE_FILES.json': base64.b64encode(manifest_raw).decode()}
    for name, pin in manifest.items():
        check()
        p = D / name
        need(p.name == name and not p.is_symlink(), 'source member differs')
        b = p.read_bytes()
        need(len(b) == pin['bytes'] and hashlib.sha256(b).hexdigest() == pin['SHA256'], 'source pin differs')
        package[name] = base64.b64encode(b).decode()
    claim_raw = (D / 'CLAIM.json').read_bytes()
    claim = json.loads(claim_raw)
    need(claim['root_GO'] is True and claim['CPU_GO'] is True and
         claim['GPU_GO'] is False and claim['training_GO'] is False, 'root adoption required')
    plan = json.loads((D / 'PLAN.json').read_bytes())
    mp = plan['generic64_metadata']
    metadata = G / mp['path']
    need(metadata.is_file() and not metadata.is_symlink() and metadata.stat().st_size == mp['bytes'] <= 32768,
         'fixed small original64 identity metadata required')
    b = metadata.read_bytes()
    need(len(b) == mp['bytes'] and hashlib.sha256(b).hexdigest() == mp['SHA256'], 'original64 metadata differs')
    payload = dict(case='T03.sft', epoch=epoch, request_id=request_id, publication=publication,
                   source_manifest_SHA256=hashlib.sha256(manifest_raw).hexdigest(),
                   claim_SHA256=hashlib.sha256(claim_raw).hexdigest(), package=package,
                   generic64_metadata_b64=base64.b64encode(b).decode())
    raw = json.dumps(payload, separators=(',', ':')).encode()
    need(len(raw) < 2 * 1024**2, 'bounded root payload required')
    check()
    return ('INSTALL_PAYLOAD_B64=' + repr(base64.b64encode(raw).decode()) + '\n' +
            (D / 'install_and_run.py').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--publication', required=True)
    p.add_argument('--request', required=True)
    p.add_argument('--origin', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # Root creates the unique ignored dispatch directory/intent before this function.
    need(a.output.is_absolute() and a.output.resolve().is_relative_to((G / 'local-only').resolve())
         and not a.output.exists(), 'exclusive ignored body output required')
    body = make_body(a.publication, a.request, a.origin).encode()
    with a.output.open('xb') as f:
        f.write(body)
