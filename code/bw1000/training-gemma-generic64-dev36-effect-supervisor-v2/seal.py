"""Root may adopt this NEW scope then reseal nested first. Default only seals sources."""
import argparse
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent

def pin(path):
    raw=path.read_bytes();return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())

def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')

def seal(adopt=False):
    if adopt:
        for folder in (D/'core',D):
            path=folder/'CLAIM.json';value=json.loads(path.read_bytes())
            for key in ('root_GO','GPU_GO','READ_GO','SCORE_GO'):value[key]=True
            value['state']='ROOT_ADOPTED_EXECUTION_ONLY_NOT_QUALIFICATION'
            write(path,value)
    core=D/'core'
    write(core/'SOURCE_FILES.json',dict(schema='GEMMA_DEV36_EFFECT_SOURCES_V1',files={p.name:pin(p) for p in sorted(core.iterdir())
        if p.is_file() and p.name!='SOURCE_FILES.json'}))
    files={p.relative_to(D).as_posix():pin(p) for p in sorted(D.rglob('*')) if p.is_file() and
        p!=D/'SOURCE_FILES.json' and '__pycache__' not in p.parts}
    write(D/'SOURCE_FILES.json',dict(schema='GEMMA_DEV36_OPERATIONAL_SOURCES_V1',files=files))
    return dict(outer=pin(D/'SOURCE_FILES.json'),core=pin(core/'SOURCE_FILES.json'),claim=pin(D/'CLAIM.json'),files=len(files)+1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopt',action='store_true');a=p.parse_args();print(json.dumps(seal(a.adopt),sort_keys=True))
