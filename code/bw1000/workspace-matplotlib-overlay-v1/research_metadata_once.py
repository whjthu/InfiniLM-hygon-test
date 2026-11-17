"""Approved public metadata research only: seven exact PyPI JSONs, no wheel/data GET."""
import ctypes
import argparse
from ctypes import wintypes
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time
import urllib.request

D=Path(__file__).resolve().parent
G=D.parents[2]
VERSIONS={'matplotlib':'3.10.7','contourpy':'1.3.3','cycler':'0.12.1','fonttools':'4.60.1',
          'kiwisolver':'1.4.9','pyparsing':'3.2.5','six':'1.17.0'}


def two_cores():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
    kernel.SetProcessAffinityMask.argtypes=(wintypes.HANDLE,ctypes.c_size_t)
    handle=kernel.GetCurrentProcess();process=ctypes.c_size_t();system=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle,ctypes.byref(process),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    bits=[i for i in range(8*ctypes.sizeof(process)) if process.value & (1<<i)]
    if len(bits)<2:raise RuntimeError('two allowed CPU cores required')
    selected=sum(1<<i for i in bits[:2])
    if not kernel.SetProcessAffinityMask(handle,selected):raise ctypes.WinError(ctypes.get_last_error())
    actual=ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle,ctypes.byref(actual),ctypes.byref(system)) or actual.value!=selected:
        raise RuntimeError('actual research affinity differs')
    return bits[:2]


def main(output):
    cores=two_cores();started=time.monotonic();deadline=started+120
    if not output.startswith('workspace-matplotlib-overlay-v1-metadata-') or '/' in output or '\\' in output or '..' in output:
        raise ValueError('owned metadata-only output name required')
    out=G/'local-only'/output;out.mkdir(mode=0o700)
    total=0;rows={}
    try:
        for name,version in VERSIONS.items():
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('original120s metadata budget')
            url='https://pypi.org/pypi/'+name+'/'+version+'/json'
            request=urllib.request.Request(url,headers={'Accept':'application/json','User-Agent':'Hygon-workspace-dependency-metadata/1.0'})
            raw=bytearray()
            with urllib.request.urlopen(request,timeout=min(12,remaining)) as response:
                if response.status!=200 or response.url!=url:raise ValueError('official exact metadata response differs')
                while True:
                    if time.monotonic()>=deadline:raise TimeoutError('original120s metadata budget')
                    block=response.read(min(65536,1048576-total+1))
                    if not block:break
                    total+=len(block);raw.extend(block)
                    if total>1048576:raise ValueError('metadata total exceeds1MiB')
            raw=bytes(raw);value=json.loads(raw)
            if value['info']['version']!=version:raise ValueError('exact release version differs')
            with (out/(name+'-'+version+'.json')).open('xb') as stream:stream.write(raw)
            rows[name]=dict(url=url,version=version,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),
                            requires_python=value['info']['requires_python'],requires_dist=value['info']['requires_dist'])
        passed=True;error=None
    except Exception as exc:passed=False;error=type(exc).__name__
    result=dict(passed=passed,CPU_affinity=cores,metadata_GETs_completed=len(rows),total_metadata_bytes=total,
                metadata_limit_bytes=1048576,whole_seconds=120,elapsed_seconds=time.monotonic()-started,rows=rows,error_type=error,
                wheel_GETs=0,WDI_GETs=0,pip_installs=0,Torch_imports=0,GPU=0,
                python=sys.version,executable=sys.executable,
                installed={name:md.version(name) for name in ('numpy','Pillow','packaging','python-dateutil','pip')})
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result,sort_keys=True))
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='workspace-matplotlib-overlay-v1-metadata-first')
    raise SystemExit(main(parser.parse_args().output))
