"""Local AST/manifest/static byte-budget verification. No endpoint or payload I/O."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
H = lambda value: hashlib.sha256(value).hexdigest()


def main():
    mf = json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name, pin in mf['files'].items():
        assert '/' not in name and '\\' not in name
        raw = (D/name).read_bytes()
        assert len(raw) == pin['bytes'] and H(raw) == pin['SHA256']
        if name.endswith('.py'):
            ast.parse(raw)
        elif name.endswith('.json'):
            json.loads(raw)
    plan = json.loads((D/'PLAN.json').read_bytes())
    claim = json.loads((D/'CLAIM.json').read_bytes())
    assert claim['case'] == plan['case'] == 'T04.sft'
    assert claim['GPU_GO'] is False and claim['root_GO'] is False
    assert claim['transfer_GO'] is False and claim['finalize_GO'] is False
    assert len(plan['files']) == 18 and len(plan['phases']) == 19
    assert plan['phase0_epoch_seconds'] == 90 and plan['epoch_seconds'] == 7200 and plan['bridge_block_bytes'] == 1<<20
    assert sum(row['bytes'] for row in plan['files'].values()) == 17070557079
    offsets = dict.fromkeys(plan['files'], 0)
    for index, phase in enumerate(plan['phases']):
        assert phase['id'] == index and 0 < phase['payload_bytes'] <= 1<<30
        assert sum(row['bytes'] for row in phase['segments']) == phase['payload_bytes']
        for row in phase['segments']:
            assert row['offset'] == offsets[row['name']]
            offsets[row['name']] += row['bytes']
    assert offsets == {name: row['bytes'] for name, row in plan['files'].items()}
    assert sum(row['payload_bytes'] for row in plan['phases']) == 17070557079
    assert len((D/'endpoint.py').read_bytes())+len(repr(plan).encode())+4096 <= 262144
    print(json.dumps(dict(passed=True, source_members=len(mf['files']), phases=19,
          model_bytes=17070557079, max_phase_payload_bytes=1<<30,
          root_GO=False, transfer_GO=False, finalize_GO=False, remote_calls=0)))


if __name__ == '__main__':
    main()
