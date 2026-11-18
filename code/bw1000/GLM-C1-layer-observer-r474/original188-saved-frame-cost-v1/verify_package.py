"""Validate published source bytes and explicit review/adopted states."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, adopted=None):
    root = Path(root)
    mf = json.loads((root/'SOURCE_FILES.json').read_bytes())
    for name, pin in mf['files'].items():
        if Path(name).name != name:
            raise ValueError('FLAT_SOURCE_MEMBER')
        path = root/name
        if path.is_symlink():
            raise ValueError('SOURCE_SYMLINK')
        raw = path.read_bytes()
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['SHA256']:
            raise ValueError('SOURCE_BYTES')
    claim = json.loads((root/'CLAIM.json').read_bytes())
    pair = (claim['root_GO'], claim['CPU_GO'])
    if pair not in ((False, False), (True, True)) or (adopted is not None and pair != (adopted, adopted)):
        raise ValueError('REVIEW_ADOPTED_STATE')
    if any(claim[k] is not False for k in ('GPU_GO','Model_GO','performance_GO')):
        raise ValueError('NO_GPU_MODEL_PERFORMANCE_SCOPE')
    if claim['epoch_seconds'] != 180 or claim['owner_file_max_bytes'] != 128*1024**2 or claim['total_explicit_read_max_bytes'] != 2*1024**3:
        raise ValueError('EXACT_RESOURCE_SCOPE')
    return mf


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--adopted', action='store_true')
    a = p.parse_args()
    mf = verify(Path(__file__).resolve().parent, adopted=a.adopted)
    print(json.dumps(dict(source_files=len(mf['files']), verified=True, actual_saved_owner_IO=False)))
