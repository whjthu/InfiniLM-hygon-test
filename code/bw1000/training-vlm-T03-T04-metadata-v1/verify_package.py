"""Local published-byte and bounded metadata contract verifier; no transport imports."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

D = Path(__file__).resolve().parent
G = D.parents[2]
H = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    mf = json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name, pin in mf['files'].items():
        p = PurePosixPath(name)
        assert p.as_posix() == name and not p.is_absolute() and '..' not in p.parts
        raw = (D/name).read_bytes()
        assert len(raw) == pin['bytes'] and H(raw) == pin['SHA256']
        if name.endswith('.py'):
            ast.parse(raw)
        assert b'\r' not in raw
        assert all(line.rstrip(b' \t') == line for line in raw.splitlines())
    plan = json.loads((D/'PLAN.json').read_bytes())
    claim = json.loads((D/'CLAIM.json').read_bytes())
    assert claim['metadata_only'] and not claim['GPU_GO'] and not claim['Model_GO'] and not claim['full_weight_identity']
    assert claim['epoch_seconds'] == plan['epoch_seconds'] == 300
    assert plan['max_file_bytes'] == 12*1024**2 and plan['max_compressed_bytes'] == 32*1024**2
    assert plan['max_decompressed_bytes'] == 64*1024**2 and plan['output_members'] == 36
    assert len(plan['members']) == 35
    assert sum(p['kind'] == 'metadata' for p in plan['members'].values()) == 27
    assert sum(p['kind'] == 'header8plus' for p in plan['members'].values()) == 8
    assert sum(len(c['records']) for c in plan['cases'].values()) == 32
    assert plan['exact_source_read_bytes'] == 32327445
    assert plan['maximum_uncompressed_bundle_bytes'] == 32334233
    assert plan['maximum_uncompressed_bundle_bytes'] < plan['max_decompressed_bytes']
    assert all(p['bytes'] <= plan['max_file_bytes'] for p in plan['members'].values() if p['kind'] == 'metadata')
    for pin in plan['source_evidence']:
        raw = (G/pin['path']).read_bytes()
        assert len(raw) == pin['bytes'] and H(raw) == pin['SHA256']
    assert plan['cases']['internvl']['run_root'] != plan['cases']['llava']['run_root']
    print(json.dumps(dict(passed=True, files=len(mf['files']), source_manifest_SHA256=H((D/'SOURCE_FILES.json').read_bytes()),
                          exact_source_read_bytes=plan['exact_source_read_bytes'], remote_calls=0)))


if __name__ == '__main__':
    main()
