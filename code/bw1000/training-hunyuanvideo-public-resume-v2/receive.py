"""Continue the public model after the original receiver's confirmed EOF."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import stat
import time

def timeout(*_):
    raise TimeoutError('continuation deadline')

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())

def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def pin(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

root = Path(PLAN['root'])
final = root / PLAN['final']
stage = root / PLAN['stage']
folder = stage / PLAN['folder']
partial = folder / PLAN['partial']
scope = root / 'retained-suffix-v2'
assert 0 < DEADLINE-time.time() <= PLAN['seconds']
signal.signal(signal.SIGALRM, timeout)
signal.alarm(max(1, int(DEADLINE-time.time())))
raw = (root / PLAN['prior']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == PLAN['prior_outcome_SHA256']
prior = json.loads(raw)
assert prior['request_id'] == PLAN['prior_request'] and prior['error_type']=='EOFError' and prior['passed'] is False and prior['offset'] == PLAN['offset']
assert not final.exists() and not final.is_symlink() and not partial.is_symlink()
assert shutil.disk_usage(root).free > PLAN['total']-PLAN['offset']+1024**3
assert pin(folder/'config.json')==PLAN['config_SHA256']
scope.mkdir(mode=0o700)
base = dict(claim=CLAIM, request_id=REQUEST, deadline=DEADLINE)
save(scope/'intent.json', base)
position = PLAN['offset']
try:
    with os.fdopen(os.open(partial, os.O_RDWR | os.O_NOFOLLOW), 'r+b') as f:
        before = os.fstat(f.fileno())
        assert stat.S_ISREG(before.st_mode) and before.st_size==position and os.path.samestat(partial.lstat(), before)
        digest=hashlib.file_digest(f, 'sha256')
        assert digest.hexdigest()==PLAN['prefix_SHA256']
        after=os.fstat(f.fileno())
        assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        f.seek(position)
        print(json.dumps(dict(**base, ready=True, offset=position)), flush=True)
        next_receipt = position + 64*1024**2
        while position < PLAN['total']:
            chunk=sys.stdin.buffer.read(min(1024**2,PLAN['total']-position))
            if not chunk:
                f.flush()
                os.fsync(f.fileno())
                save(scope/'last-fsynced.json', dict(**base, offset=position, prefix_SHA256=digest.hexdigest()))
                raise EOFError('continuation ended early')
            f.write(chunk)
            digest.update(chunk)
            position+=len(chunk)
            if position>=next_receipt:
                f.flush()
                os.fsync(f.fileno())
                temp=scope/'progress.tmp'
                save(temp,dict(**base,offset=position,prefix_SHA256=digest.hexdigest()))
                temp.replace(scope/'progress.json')
                next_receipt=position+64*1024**2
        f.flush()
        os.fsync(f.fileno())
    if sys.stdin.buffer.read(1) or digest.hexdigest()!=PLAN['full_SHA256']:
        raise ValueError('whole-file verification failed')
    weight = folder / 'diffusion_pytorch_model.safetensors'
    if partial != weight:
        if weight.exists() or weight.is_symlink():
            raise FileExistsError('weight name already in use')
        partial.rename(weight)
    fsync_directory(folder)
    stage.rename(final)
    fsync_directory(root)
    result=dict(**base,passed=True,offset=position,whole_SHA256=digest.hexdigest())
except Exception as e:
    result=dict(**base,passed=False,offset=position,error_type=type(e).__name__)
save(scope/'outcome.json',result)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 1)
