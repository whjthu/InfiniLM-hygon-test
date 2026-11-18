"""Refresh nested/outer pins; --adopt changes only the six/eight published GO fields."""
import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def manifest(root):
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p != root / 'SOURCE_FILES.json':
            raw = p.read_bytes(); files[p.relative_to(root).as_posix()] = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
    write(root / 'SOURCE_FILES.json', dict(schema='EXACT_SOURCE_FILES_V1', files=files))


def main(adopt=False):
    if adopt:
        for relative in ['CLAIM.json','INSTALL_PLAN.json','core/PLAN.json','core/CLAIM.json']:
            path = D / relative; value = json.loads(path.read_bytes())
            value.update(root_GO=True, GPU_GO=True); write(path, value)
    manifest(D / 'core')
    path = D / 'INSTALL_PLAN.json'; value = json.loads(path.read_bytes())
    value['core_manifest_SHA256'] = hashlib.sha256((D / 'core/SOURCE_FILES.json').read_bytes()).hexdigest(); write(path, value)
    manifest(D)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--adopt', action='store_true'); a = p.parse_args(); main(a.adopt)
