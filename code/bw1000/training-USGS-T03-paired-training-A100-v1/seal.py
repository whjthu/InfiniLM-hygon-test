"""Seal only this new package; optional root adoption keeps all eight execution flags aligned."""
import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def pin(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8'))


def main(adopt=False):
    if adopt:
        for name in ('CLAIM.json', 'CONFIG.template.json', 'INSTALL_PLAN.json', 'supervisor/CLAIM.json'):
            p = D / name; value = json.loads(p.read_bytes())
            value['root_GO'] = value['GPU_GO'] = True
            if name == 'CLAIM.json':
                value['CPU_GO'] = value['training_GO'] = True
            write(p, value)
    overlays = {p.relative_to(D / 'overlay').as_posix(): pin(p) for p in sorted((D / 'overlay').rglob('*')) if p.is_file()}
    assert len(overlays) == 5
    write(D / 'OVERLAYS.json', dict(files=overlays))
    base = json.loads((D / 'BASE_FRAMEWORK_FILES.json').read_bytes())
    assert len(base['files']) == 400
    base['files'].update(overlays)
    assert len(base['files']) == 404
    base['assembly'] = 'Qualified common396 -> unchanged T04 common400 -> explicit T03 opt-in/common404; no new training loop.'
    write(D / 'MERGED_FRAMEWORK_FILES.json', base)
    sub = D / 'supervisor'
    write(sub / 'SOURCE_FILES.json', dict(files={p.name: pin(p) for p in sorted(sub.iterdir()) if p.is_file() and p.name != 'SOURCE_FILES.json'}))
    plan = json.loads((D / 'INSTALL_PLAN.json').read_bytes())
    plan['supervisor_manifest'] = pin(sub / 'SOURCE_FILES.json')
    plan['supervisor_claim'] = pin(sub / 'CLAIM.json')
    plan['framework_manifest'] = pin(D / 'MERGED_FRAMEWORK_FILES.json')
    write(D / 'INSTALL_PLAN.json', plan)
    write(D / 'SOURCE_FILES.json', dict(files={p.relative_to(D).as_posix(): pin(p) for p in sorted(D.rglob('*'))
        if p.is_file() and p != D / 'SOURCE_FILES.json'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--adopt', action='store_true')
    main(p.parse_args().adopt)
