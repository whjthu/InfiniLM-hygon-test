"""Local package closure only; never connects or launches."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

D = Path(__file__).resolve().parent


def main():
    raw = (D/'SOURCE_FILES.json').read_bytes()
    manifest = json.loads(raw)
    actual = {path.name for path in D.iterdir() if path.is_file() and path.name != 'SOURCE_FILES.json'}
    assert actual == set(manifest['files'])
    for name,pin in manifest['files'].items():
        value = (D/name).read_bytes()
        assert len(value) == pin['bytes'] and hashlib.sha256(value).hexdigest() == pin['SHA256']
        assert b'\r' not in value and not any(line.rstrip(b' \t') != line for line in value.splitlines())
        if name.endswith('.py'):
            ast.parse(value)
    plan = json.loads((D/'PLAN.json').read_bytes())
    claim = json.loads((D/'CLAIM.json').read_bytes())
    assert plan['whole_seconds'] == claim['whole_seconds'] == 21600
    assert claim['max_invocations'] == 1 and claim['provision_GO'] and not claim['GPU_GO'] and not claim['Model_GO']
    files = plan['files']
    assert len(files) == len({row['name'] for row in files}) == 26
    assert sum(row['bytes'] for row in files) == plan['stream_payload_bytes'] == 18471312799
    assert sum(row['bytes'] for row in files if row['source_kind']=='A100_original_file') == 18471287442
    assert len([row for row in files if row['name'].startswith('model/')]) == 17
    assert len([row for row in files if 'header_bytes' in row]) == 4
    initials = [row for row in files if row['name'].startswith('initial/')]
    assert len(initials) == len({row['SHA256'] for row in initials}) == 8
    assert all(row['bytes'] == 174821899 for row in initials)
    for row in files:
        path = PurePosixPath(row['name'])
        assert path.as_posix() == row['name'] and not path.is_absolute() and '..' not in path.parts and '\\' not in row['name']
        assert re.fullmatch('[0-9a-f]{64}',row['SHA256']) and row['bytes'] > 0
    assert all(PurePosixPath(plan[name]).is_absolute() and '\\' not in plan[name]
               for name in ('source_namespace','target_namespace'))
    print(json.dumps(dict(passed=True, files=len(actual), SOURCE_FILES_SHA256=hashlib.sha256(raw).hexdigest(),
                          payload_files=26,payload_bytes=18471312799,local_only=True,GPU_GO=False),sort_keys=True))


if __name__ == '__main__':
    main()
