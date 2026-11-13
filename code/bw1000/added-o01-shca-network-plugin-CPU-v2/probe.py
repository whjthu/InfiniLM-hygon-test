"""Verify the installed SHCA plugin ABI without initializing communication."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

prefix=Path('/candidate-opt')
candidate=prefix/'rccl-shca-net/lib/librccl-net-shca.so'
rccl=Path('/opt/dtk/lib/librccl.so.1.0')
def identity(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):
            h.update(chunk)
    return dict(bytes=p.stat().st_size,SHA256=h.hexdigest())

def main():
    result={'plugin':str(candidate),'RCCL':str(rccl)}
    try:
        result['plugin_pin']=identity(candidate)
        assert result['plugin_pin']['SHA256']=='20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0'
        result['rccl_pin']=identity(rccl)
        run=subprocess.run(['ldd','-r',str(candidate)],capture_output=True,text=True,timeout=12)
        result['ldd_rc']=run.returncode
        result['loader_errors']=[line.strip() for line in (run.stdout+run.stderr).splitlines()
                                 if 'not found' in line or 'undefined symbol' in line][:32]
        handle=ctypes.CDLL(str(candidate),mode=os.RTLD_NOW|os.RTLD_LOCAL)
        versions={}
        for version in range(5,13):
            symbol=f'ncclNetPlugin_v{version}'
            try:
                name=ctypes.c_char_p.in_dll(handle,symbol).value
                versions[symbol]=name.decode() if name else None
            except ValueError:
                continue
        result['exports']=versions
        paths={}
        for line in Path('/proc/self/maps').read_text().splitlines():
            cols=line.split(maxsplit=5)
            if len(cols)!=6 or not cols[5].startswith('/'):
                continue
            p=Path(cols[5])
            if re.match(r'lib(?:(?:nccl|rccl)-net|uc[mpst]|ibverbs|shca|nl-)',p.name) and str(p) not in paths:
                paths[str(p)]=identity(p)
        result['mapped_dependencies']=paths
        result['plugin_v8_loaded']=bool(versions.get('ncclNetPlugin_v8'))
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)[-500:])
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
