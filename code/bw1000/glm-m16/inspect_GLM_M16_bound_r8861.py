import pathlib,json,hashlib,shlex
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
def host(p):return pathlib.Path(str(p).replace('/refactor/',str(r)+'/'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((r/'candidates/glm-exact-budget-r8854/CPU-build.json').read_text())
assert c['completed'] and sha(host(c['library']))==c['library_SHA256']
link=next(x for x in c['commands'] if '-shared' in x)
rows=[];sources={};headers={}
for obj in [x for x in link if x.endswith('.o')]:
 p=host(obj);d=p.with_suffix('.d')
 if not d.exists():rows.append({'object':obj,'MD_missing':True});continue
 deps=shlex.split(d.read_text().replace('\\\n',' ').split(':',1)[1]);src=next((v for v in deps if pathlib.Path(v).suffix in ('.cpp','.cc','.cu')),None)
 hit=[v for v in deps if pathlib.Path(v).name in ('glm_model_resident_policy_r5640.hpp','glm_moe_dsa_projected_cache_r22.hpp')]
 row={'object':obj,'object_SHA256':sha(p),'MD_SHA256':sha(d),'source':src,'resident_header_dependencies':hit}
 if src:
  sp=host(src);s=sp.read_text();row['source_SHA256']=sha(sp)
  if ('sparse_attention_exact_rows_r5555' in s and 'rt::Tensor sparse_attention_exact_rows_r5555' in s) or 'bool selected(Flags' in s:
   sources[src]={'SHA256':sha(sp),'text':s}
 for v in hit:
  hp=host(v);headers[v]={'SHA256':sha(hp),'text':hp.read_text()}
 rows.append(row)
print(json.dumps({'CPU_only':True,'GPU':False,'dlopen':False,'current_DSO':c['library'],'DSO_SHA256':c['library_SHA256'],'objects':rows,'matched_sources':sources,'bound_headers':headers}))
