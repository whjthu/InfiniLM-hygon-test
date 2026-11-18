"""Root may adopt CPU-only GO; refresh exact manifest without touching original datasets."""
import argparse,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopt',action='store_true');a=p.parse_args()
    if a.adopt:
        for name in ['PLAN.json','CLAIM.json']:
            value=json.loads((D/name).read_bytes());value['root_GO']=value['CPU_GO']=True
            if name=='CLAIM.json':value['state']='ROOT_ADOPTED_PUBLISHED_BEFORE_ONLY_INVOCATION'
            (D/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    files={x.name:dict(bytes=len(x.read_bytes()),SHA256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sorted(D.iterdir()) if x.is_file() and x.name!='SOURCE_FILES.json'}
    (D/'SOURCE_FILES.json').write_text(json.dumps(dict(schema='EXACT_CPU_BLIND25_SOURCE_V1',files=files),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(source_members=len(files),manifest_SHA256=hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest(),claim_SHA256=hashlib.sha256((D/'CLAIM.json').read_bytes()).hexdigest())))
