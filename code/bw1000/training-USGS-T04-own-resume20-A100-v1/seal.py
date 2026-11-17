"""Recompute only owned preparation manifests after root adoption; no runtime/action."""
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent


def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(name,value):(D/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
def seal():
    from verify_package import adoption_gate
    adoption_gate(*[json.loads((D/name).read_bytes()) for name in ['CONFIG.template.json','CLAIM.json','supervisor/CLAIM.json']])
    nested={'files':{p.name:dict(bytes=p.stat().st_size,SHA256=sha(p.read_bytes())) for p in sorted((D/'supervisor').iterdir()) if p.is_file() and p.name!='SOURCE_FILES.json'}}
    write('supervisor/SOURCE_FILES.json',nested)
    config=json.loads((D/'CONFIG.template.json').read_bytes());plan=json.loads((D/'INSTALL_PLAN.json').read_bytes())
    raw=(D/'MERGED_FRAMEWORK_FILES.json').read_bytes();config['framework_manifest'].update(bytes=len(raw),SHA256=sha(raw));write('CONFIG.template.json',config)
    for key,name in [('framework_manifest','MERGED_FRAMEWORK_FILES.json'),('base_manifest','BASE_FRAMEWORK_FILES.json'),('supervisor_manifest','supervisor/SOURCE_FILES.json'),('supervisor_claim','supervisor/CLAIM.json'),('runtime_config_template','CONFIG.template.json')]:
        raw=(D/name).read_bytes();plan[key]=dict(bytes=len(raw),SHA256=sha(raw))
    write('INSTALL_PLAN.json',plan)
    write('SOURCE_FILES.json',{'files':{p.relative_to(D).as_posix():dict(bytes=p.stat().st_size,SHA256=sha(p.read_bytes())) for p in sorted(D.rglob('*')) if p.is_file() and p.relative_to(D).as_posix()!='SOURCE_FILES.json'}})
    return dict(source_manifest_SHA256=sha((D/'SOURCE_FILES.json').read_bytes()),claim_SHA256=sha((D/'CLAIM.json').read_bytes()))

if __name__=='__main__':print(json.dumps(seal(),sort_keys=True))
