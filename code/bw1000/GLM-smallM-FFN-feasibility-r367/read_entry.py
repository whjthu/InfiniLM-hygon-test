"""Resolve the already-linked layer entry against its original CPU receipt."""
from pathlib import Path
import json,hashlib,datetime,subprocess,re
R=Path.home()/'hygon-inference-refactor-20261014';host=lambda p:Path(str(p).replace('/refactor',str(R),1));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';a=json.loads(p.read_bytes());q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,GPU=False,dlopen=False,receipt=dict(path=str(p),SHA256=sha(p)),entry_objects=[])
link=next(cmd for cmd in a['commands']if isinstance(cmd,list)and'-shared'in cmd);objects=[host(p)for p in link if p.endswith('.o')]
text=subprocess.check_output(['nm','-A','-g','--defined-only',*[str(p)for p in objects]],text=True,timeout=45)
for line in text.splitlines():
 if re.search(r' glm_layer_(forward|create|destroy)_r5670$',line):
  o=Path(line.split(':',1)[0]);item=dict(symbol=line.split()[-1],object=str(o),object_SHA256=sha(o),source_files=[])
  md=o.with_suffix('.d')
  if md.exists():
   item['MD_SHA256']=sha(md)
   for n in re.findall(r'[^\s\\]+\.cpp',md.read_text()):
    s=host(n)
    if s.exists():item['source_files'].append(dict(path=str(s),SHA256=sha(s),text=s.read_text()))
  q['entry_objects'].append(item)
assert len(q['entry_objects'])==3
print(json.dumps(q))
