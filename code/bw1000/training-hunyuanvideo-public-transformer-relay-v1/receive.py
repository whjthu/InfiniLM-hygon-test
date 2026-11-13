"""Receive the two public 480p T2V assets and publish only after full verification."""
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time

root = Path(PLAN['root'])
assert root.parent.is_dir() and not root.exists()
assert 0 < DEADLINE - time.time() <= PLAN['seconds']
assert shutil.disk_usage(root.parent).free > sum(p['bytes'] for p in PLAN['files'].values()) + 1024**3
def expired(*_):
    raise TimeoutError('public transfer expired')
signal.signal(signal.SIGALRM, expired)
signal.alarm(max(1, int(DEADLINE - time.time())))
root.mkdir(mode=0o700)
def save(name, value):
    tmp = root / (name + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, root/name)
base = dict(claim=CLAIM, request_id=REQUEST, deadline=DEADLINE)
save('intent.json', base)
stage = root / 'model.incomplete'
folder = stage / 'transformer/480p_t2v'
folder.mkdir(parents=True, mode=0o700)
print(json.dumps(dict(**base, ready=True)), flush=True)
files = {}
received = 0
try:
    for name, pin in PLAN['files'].items():
        partial = folder / (name + '.partial')
        digest = hashlib.sha256()
        position = 0
        next_progress = 64*1024**2
        with partial.open('xb') as f:
            while position < pin['bytes']:
                block = sys.stdin.buffer.read(min(1024**2, pin['bytes'] - position))
                if not block:
                    raise EOFError('public stream ended early')
                f.write(block)
                digest.update(block)
                position += len(block)
                received += len(block)
                if received >= next_progress:
                    save('progress.json', dict(**base, received_bytes=received))
                    next_progress = received + 64*1024**2
            f.flush()
            os.fsync(f.fileno())
        if digest.hexdigest() != pin['SHA256']:
            raise ValueError('public file digest differs')
        partial.rename(folder/name)
        files[name] = pin
    if sys.stdin.buffer.read(1):
        raise ValueError('unexpected trailing public bytes')
    final = root / 'hunyuanvideo-1.5'
    stage.rename(final)
    result = dict(**base, passed=True, files=files, received_bytes=received, model_directory=str(final))
except Exception as error:
    result = dict(**base, passed=False, files=files, received_bytes=received, error_type=type(error).__name__)
save('outcome.json', result)
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['passed'] else 1)
