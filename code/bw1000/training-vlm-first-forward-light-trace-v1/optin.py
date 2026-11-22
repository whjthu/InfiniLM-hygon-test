"""Minimal callback-side entry. No config means no backend import or work."""
import hashlib
import json
import os
from pathlib import Path
import stat


def attach_if_requested(*,model=None,trainer=None,rank=None,check=None,environ=None):
    env=os.environ if environ is None else environ
    path=env.get('VLM_LIGHT_TRACE_CONFIG')
    sha=env.get('VLM_LIGHT_TRACE_CONFIG_SHA256')
    if path is None and sha is None:return None
    if not path or not sha:raise ValueError('both explicit trace config path/SHA required')
    p=Path(path);before=p.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=16384:
        raise ValueError('bounded regular trace config required')
    keys=lambda s:[getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]
    with p.open('rb') as f:
        opened=os.fstat(f.fileno());raw=f.read(16385);after=os.fstat(f.fileno())
    if not keys(before)==keys(opened)==keys(after)==keys(p.lstat()) or len(raw)!=before.st_size:
        raise ValueError('trace config identity changed')
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('trace config SHA differs')
    config=json.loads(raw)
    # An explicit disabled config also avoids importing the native bridge.
    if config.get('enabled',False) is False:return None
    from native_bridge import install
    return install(config,model=model,trainer=trainer,rank=rank,check=check)
