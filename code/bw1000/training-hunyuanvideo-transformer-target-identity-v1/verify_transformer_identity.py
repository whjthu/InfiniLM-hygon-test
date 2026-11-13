"""Verify the exact published Transformer after both original endpoints succeed."""

import argparse
import hashlib
import json
from pathlib import Path
import time

PLAN_SHA = 'd2d58e2514b54be97f56def4d85fced1153536064c111c6c06a64e37601d681b'
REQUEST = '0a610475811d4237955e139ab9dda71c'
TERMINAL_READER_SHA = '3d0cb87060723bff8f4ced1cf525647732ee953f72249a0fd916e139a4807cc8'
STABLE_READER_SHA = '947a229b6becf393c09423c28fcf67dba69925a70244e6f2a4f3e6d1a8b0f7ab'


def load_reader(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('reader source differs')
    scope = {'__name__': 'pinned_reader'}
    exec(compile(raw, str(path), 'exec'), scope)
    return scope


def audit(plan, sender, receiver, launch, verify, stable_read):
    if launch['request_id'] != REQUEST:
        raise ValueError('wrong original Transformer request')
    terminal = verify('transformer', plan, sender, receiver, launch)
    if terminal['state'] != 'SENDER_AND_RECEIVER_SUCCESS':
        raise ValueError('original Transformer did not publish successfully')
    started = time.monotonic()
    root = Path(plan['root'])
    final = root / plan['final']
    folder = final / plan['folder']
    directories = [root, final, final / 'transformer', folder]
    if any(not p.is_dir() or p.is_symlink() for p in directories):
        raise ValueError('published Transformer directory is missing or indirect')
    partial = root / plan['stage']
    if partial.exists() or partial.is_symlink():
        raise ValueError('partial Transformer namespace still exists')
    config, _ = stable_read(folder / 'config.json', 968)
    if config != plan['config_SHA256']:
        raise ValueError('Transformer config differs')
    name = plan['partial'].removesuffix('.partial')
    weight, _ = stable_read(folder / name, plan['total'])
    if weight != plan['full_SHA256']:
        raise ValueError('Transformer whole-file hash differs')
    return dict(passed=True, request_id=REQUEST, bytes=plan['total'], weight_SHA256=weight,
                config_SHA256=config, GPU_calls=0, elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'sender_terminal', 'receiver_observation', 'launch_intent',
                 'terminal_reader', 'stable_reader'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    terminal = load_reader(args.terminal_reader, TERMINAL_READER_SHA)
    stable = load_reader(args.stable_reader, STABLE_READER_SHA)
    read = terminal['read']
    plan, plan_sha = read(args.plan)
    if plan_sha != PLAN_SHA:
        raise ValueError('original transfer plan differs')
    sender, sender_sha = read(args.sender_terminal)
    receiver, receiver_sha = read(args.receiver_observation)
    launch, launch_sha = read(args.launch_intent)
    result = audit(plan['assets']['transformer'], sender, receiver['assets']['transformer'],
                   launch, terminal['verify'], stable['stable_read'])
    print(json.dumps(dict(result, plan_SHA256=plan_sha, sender_SHA256=sender_sha,
                          receiver_SHA256=receiver_sha, launch_SHA256=launch_sha), allow_nan=False))


if __name__ == '__main__':
    main()
