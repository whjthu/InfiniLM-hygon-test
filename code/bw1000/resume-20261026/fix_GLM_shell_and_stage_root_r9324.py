from pathlib import Path
import json,hashlib,subprocess,sys
r=Path.cwd();v=Path(__file__).resolve().parent;src=r/'glm-m15-model-entry-r129';dst=v/'glm-m15-model-entry-r129b';dst.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((src/'GPU-manifest.json').read_bytes());old='ccd7893dcbd48972a8bda1f710376f0d905dd3efb5e4af683ea1024373f61def';assert sha(src/'GPU-manifest.json')==old
for n,h in m['source_SHA256'].items():
 p=src/n;assert sha(p)==h;b=p.read_bytes()
 if n.endswith('.sh'):b=b.replace(b'\r\n',b'\n');assert b'\r'not in b
 q=dst/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);m['source_SHA256'][n]=sha(q)
assert [n for n,h in m['source_SHA256'].items()if sha(src/n)!=h]==['GPU_inside_r129.sh']
(dst/'GPU-manifest.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());pin=sha(dst/'GPU-manifest.json')
report={'scope':'Source-only line-ending repair before any remote staging or Slurm invocation','parent_manifest_SHA256':old,'manifest_SHA256':pin,'changed_semantics':False,'changed_source':['GPU_inside_r129.sh_CRLF_to_LF'],'remote_directory':'GLM-M15-Model-r129','original_unexecuted_source_preserved':True}
(dst/'source-repair-r129b.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
for p in dst.iterdir():
 q=r/'glm-m15-model-entry-r129b'/p.name;q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes())
prefix=(r/'publish_resume_r9230.py').read_text(encoding='utf-8-sig').split("packet('Kimi-adaptive-Model-AB-private-r9225/producer",1)[0].replace("'.inc'","'.inc','.patch'")
exec(compile(prefix,'publication-primitives','exec'),globals())
paths=json.loads((r/'model-source-files-r9323.json').read_bytes())
for p in(r/'glm-m15-model-entry-r129b').iterdir():archive(p)
rel='coordination/tasks/glm-m15-oproj-model.json';q=read(repo/rel);q.update(manifest_SHA256=pin,source_revision='r129b_LF_repair_before_remote_stage',original_r129_not_executed=True);write(rel,q)
for rel in ['manifests/bw1000/recent-source-files.json','manifests/bw1000/profile-evidence.json']:
 d=read(repo/rel);es={e['path']:e for e in d['files']}
 for name in list(dict.fromkeys(paths)):
  if name==rel or(rel.endswith('recent-source-files.json')and not name.startswith('code/')):continue
  e=es.get(name)
  if e is None:e={'path':name};d['files'].append(e);es[name]=e
  e['SHA256']=sha(repo/name)
  if rel.endswith('recent-source-files.json'):e['bytes']=(repo/name).stat().st_size
 write(rel,d)
(r/'model-source-files-r9323.json').write_bytes(json.dumps(list(dict.fromkeys(paths))).encode())
s=(v/'stage_models_root_r9323.py').read_text().replace("d=r/'glm-m15-model-entry-r129'","d=r/'glm-m15-model-entry-r129b'").replace(old,pin)
(v/'stage_models_LF_root_r9324.py').write_bytes(s.encode());print(json.dumps(report))
subprocess.run([sys.executable,'-B','-X','utf8',str(repo/'tools/verify_repository.py')],cwd=repo,check=True)
