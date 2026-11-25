"""Exact qualified same-FD weights_only loader function; no RNG restore/eager Torch import."""
import hashlib
import os
from pathlib import Path
from effect_contract import need,identity

def bounded_load(torch,path,expected,check):
    path=Path(path);before=path.lstat()
    need(path.is_file() and not path.is_symlink() and 0<before.st_size<=128*1024**2 and
         before.st_size==expected['bytes'] and identity(before)==expected['file_identity'],
         'same exact .3538 target-created state extent and current lifetime required')
    with path.open('rb') as f:
        need(identity(os.fstat(f.fileno()))==identity(before),'original state opened a different lifetime')
        h=hashlib.sha256();count=0
        while True:
            check();raw=f.read(min(1<<20,before.st_size-count+1))
            if not raw:break
            count+=len(raw);need(count<=before.st_size,'source state grew beyond frozen length');h.update(raw)
        need(count==before.st_size and h.hexdigest()==expected['SHA256'],'target-created state SHA differs')
        f.seek(0)
        value=torch.load(f,map_location='cpu',weights_only=True)
        need(identity(os.fstat(f.fileno()))==identity(before),'state changed during restricted deserialization')
    need(identity(path.lstat())==identity(before),'state pathname changed during load')
    return value
