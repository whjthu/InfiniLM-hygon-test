import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';conv=lambda p:r/pathlib.Path(p).relative_to('/refactor');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=r/'candidates/glm-attention-cohort-r9054/CPU-build.json';q=json.loads(p.read_bytes());assert q['completed'];linked=q['reused_objects_SHA256'];out={'CPU_only':True,'GPU':False,'build':False,'DSO_SHA256':q['library_SHA256'],'compiled_sources':[],'files':{}}
for folder in ['glm-PP2-B32-r8843','glm-resident-exact-r8852','glm-transient-layer-r5681']:
 doc=json.loads((r/'candidates'/folder/'CPU-build.json').read_bytes())
 for rec in doc.get('compiled',[]):
  if rec['object']not in linked:continue
  src=conv(rec['source']);assert sha(conv(rec['object']))==linked[rec['object']];out['compiled_sources'].append({'object':rec['object'],'object_SHA256':linked[rec['object']],'source':str(src),'source_SHA256':sha(src)})
  if any(t in src.name for t in ['cache_view','bridge_r8705','resident']):out['files'][str(src)]={'SHA256':sha(src),'text':src.read_text()}
  dep=conv(rec['object']).with_suffix('.d')
  if dep.exists():
   import shlex
   names=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
   for n in names:
    z=pathlib.Path(n)
    if z.name in ['glm_decode_leases_r1354.hpp','glm_projected_host_r4504.hpp','glm_transient_staging_r5680.hpp','glm_transient_geometry_r5680.hpp','glm_moe_dsa_projected_cache_r22.hpp']:
     f=conv(z);out['files'][str(f)]={'SHA256':sha(f),'text':f.read_text()}
print(json.dumps(out))
