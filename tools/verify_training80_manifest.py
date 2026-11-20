"""Verify package members from a pinned local training80 source manifest."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re


REPO = Path(__file__).resolve().parents[1]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256_value(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def member_name(value):
    if isinstance(value, str) and value and '\\' not in value and ':' not in value:
        path = PurePosixPath(value)
        if path.parts and not path.is_absolute() and '..' not in path.parts and value == path.as_posix():
            return path
    raise ValueError('source manifest path must be a normalized relative path')


def unique_pairs(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, 'duplicate source manifest key: ' + name)
        result[name] = value
    return result


def file_pin(path):
    digest = sha256()
    size = 0
    with path.open('rb') as handle:
        for raw in iter(lambda: handle.read(64 * 1024), b''):
            size += len(raw)
            digest.update(raw)
    return dict(SHA256=digest.hexdigest(), bytes=size)


def manifest_members(document):
    keys = set(document).intersection(('files', 'files_SHA256'))
    require(len(keys) == 1, 'source manifest requires exactly one supported member map')
    key = keys.pop()
    entries = document[key]
    require(isinstance(entries, dict) and bool(entries), 'source manifest member list is empty or invalid')
    result = {}
    for name, entry in entries.items():
        path = member_name(name)
        expected = dict(SHA256=entry) if key == 'files_SHA256' else entry
        require(isinstance(expected, dict) and sha256_value(expected.get('SHA256')),
                'invalid source member SHA256: ' + name)
        if 'bytes' in expected:
            require(type(expected['bytes']) is int and expected['bytes'] >= 0,
                    'invalid source member byte count: ' + name)
        result[path] = expected
    return result


def verify_package_members(repo, manifest_ref, *, required_files=()):
    """Verify package-relative bytes and any independently pinned member references."""
    repo = repo.resolve()
    relative = member_name(manifest_ref['path'])
    require(relative.parts[:2] == ('code', 'bw1000') and len(relative.parts) > 2,
            'manifest is outside code/bw1000')
    package = (repo / relative.parent).resolve(strict=True)
    require(package.is_relative_to(repo / 'code' / 'bw1000'), 'source package resolves outside code/bw1000')
    manifest = (repo / relative).resolve(strict=True)
    require(manifest.is_relative_to(package) and manifest.is_file() and manifest.stat().st_size <= 1024 * 1024,
            'manifest resolves outside package or exceeds the JSON size limit')
    raw = manifest.read_bytes()
    actual_manifest_sha = sha256(raw).hexdigest()
    require(sha256_value(manifest_ref.get('SHA256')) and actual_manifest_sha == manifest_ref['SHA256'],
            'source manifest SHA256 differs')
    if 'bytes' in manifest_ref:
        require(type(manifest_ref['bytes']) is int and manifest_ref['bytes'] == len(raw), 'source manifest byte count differs')
    document = json.loads(raw, object_pairs_hook=unique_pairs)
    require(isinstance(document, dict), 'source manifest must be a JSON object')
    expected_members = manifest_members(document)
    actual_members = {}
    resolved = set()
    for name, expected in expected_members.items():
        path = (package / name).resolve(strict=True)
        require(path.is_relative_to(package) and path.is_file() and path not in resolved and path != manifest,
                'source member escapes package or aliases another member: ' + name.as_posix())
        resolved.add(path)
        pin = file_pin(path)
        require(pin['SHA256'] == expected['SHA256'] and ('bytes' not in expected or pin['bytes'] == expected['bytes']),
                'source member bytes differ: ' + name.as_posix())
        actual_members[relative.parent / name] = pin
    for reference in required_files:
        name = member_name(reference['path'])
        member = actual_members.get(name)
        require(member is not None and member['SHA256'] == reference.get('SHA256') and
                ('bytes' not in reference or member['bytes'] == reference['bytes']),
                'required source reference is missing or differs: ' + name.as_posix())
    return dict(path=relative.as_posix(), SHA256=actual_manifest_sha, member_count=len(actual_members),
                member_bytes=sum(row['bytes'] for row in actual_members.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='repository-relative source manifest path')
    parser.add_argument('SHA256', help='manifest SHA-256 from the frozen claim')
    parser.add_argument('--require', action='append', nargs=2, default=[], metavar=('PATH', 'SHA256'))
    args = parser.parse_args()
    result = verify_package_members(REPO, dict(path=args.path, SHA256=args.SHA256),
                                    required_files=[dict(path=path, SHA256=pin) for path, pin in args.require])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
