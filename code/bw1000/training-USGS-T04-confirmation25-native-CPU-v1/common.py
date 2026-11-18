"""Pinned stdlib metadata helpers; no native imports or model code execution."""
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent


def need(ok,why):
    if not ok:raise ValueError(why)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def pinned(path,pin,cap=2*1024**2):
    path=Path(path);a=path.lstat();need(path.is_file() and not path.is_symlink() and 0<a.st_size==pin['bytes']<=cap,'BOUNDED_REGULAR_PIN')
    raw=path.read_bytes();b=path.lstat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
         and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'EXACT_STABLE_PIN');return raw


def verify_source():
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        need(Path(name).name==name,'FLAT_EXACT_SOURCE');pinned(D/name,pin)
