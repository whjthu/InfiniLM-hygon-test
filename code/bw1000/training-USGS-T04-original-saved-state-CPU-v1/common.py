"""Stdlib exact source and metadata helpers, no model or network."""
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent


def need(ok,why):
    if not ok:raise ValueError(why)

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def pinned(p,pin,cap=2*1024**2):
    need(p.is_file() and not p.is_symlink() and 0<pin['bytes']<=cap,'Bounded source member required');a=p.stat();raw=p.read_bytes();b=p.stat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'Exact source pin differs');return raw

def verify_source():
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():need(Path(name).name==name,'Flat own source only');pinned(D/name,pin)

def check_state_contract(c):
    need(c['schema']=='USGS_T04_ORIGINAL_SAVED_STATE29_CPU_V1' and c['state_files']==29 and c['state_total_bytes']==1223651001,'Exact29 originalstate scope')
    expected={'adapter_config.json','adapter_model.safetensors','args.json','optimizer.pt','scheduler.pt','trainer_state.json'}|{'rng_state_%d.pth'%i for i in range(8)}
    paths=[];total=0
    for step in ['20','25']:
        z=c['checkpoint'][step];need(Path(z['root']).name=='checkpoint-'+step and set(z['files'])==expected,'Full14 canonical checkpoint scope differs')
        for name,pin in z['files'].items():
            need(type(pin['bytes']) is int and 0<pin['bytes']<=512*1024**2 and len(pin['SHA256'])==64,'State pin size/type differs')
            paths.append(str(Path(z['root'])/name));total+=pin['bytes']
    need(Path(c['initial']['path']).name=='initial.pt' and c['initial']['bytes']==174808783,'Exact initialfile scope differs');paths.append(c['initial']['path']);total+=c['initial']['bytes']
    need(len(paths)==len(set(paths))==29 and total==c['state_total_bytes'],'Exact total path/byte denominator differs')
    return paths
