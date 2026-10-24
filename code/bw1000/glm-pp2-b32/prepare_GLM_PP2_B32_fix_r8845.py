import pathlib,json,hashlib,ast
r=pathlib.Path('.');old=r/'GLM-PP2-B32-Model-private-r8844';d=r/'GLM-PP2-B32-Model-private-r8845';d.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in old.iterdir():
 if not p.is_file()or p.name.startswith(('dispatch_','review-package','GLM-CROSSREQUEST-','GLM-transient-Model-observer-')):continue
 name=p.name.replace('r8844','r8845');b=p.read_bytes()
 if p.suffix in('.py','.sh')and'r8844'in p.name:
  s=b.decode().replace('r8844','r8845')
  if p.name=='modern_glm_prefill_micro_r8844.py':
   original="LIBRARY=pathlib.Path(json.loads((r/'candidates/glm-PP2-B32-r8843/CPU-build.json').read_text())['library'])"
   assert original in s;s=s.replace(original,"MODEL_LIBRARY_METADATA=_continuous_preflight['model_library_metadata']\nLIBRARY=pathlib.Path(MODEL_LIBRARY_METADATA['library_actual_path'])")
   oldhash="assert report['library_sha256']=='bbda28394742232a977556f246c38e96a69d3eb365ab7845220d745960575819'";assert s.count(oldhash)==2;s=s.replace(oldhash,"assert report['library_sha256']==MODEL_LIBRARY_METADATA['library_SHA256']")
   assert 'bbda28394742232a977556f246c38e96a69d3eb365ab7845220d745960575819'not in s
  if p.name=='GLM_transient_preflight_r8844.py':
   s='\n'.join(line for line in s.splitlines()if not line.startswith(" assert m['model_SHA256']=="))+'\n'
   a=s.index(' receipt=r/');b=s.index(' assert (m[',a)
   s=s[:a]+" from glm_model_library_metadata_r8845 import read_model_library_metadata\n model_meta,proof=read_model_library_metadata(r,m);assert len(proof['reused_objects_SHA256'])==22\n"+s[b:]
   s=s.replace("return {'passed':True,","return {'model_library_metadata':model_meta,'passed':True,")
  if p.name=='GLM_transient_preengine_r8844.py':
   a=s.index(' private=pathlib.Path(');b=s.index(';lib=ctypes.CDLL(str(private));symbols=set()',a)
   s=s[:a]+" from glm_model_library_metadata_r8845 import read_model_library_metadata\n model_meta,_=read_model_library_metadata(r,json.loads((r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8845.json').read_text()))\n private=pathlib.Path(model_meta['library_actual_path'])"+s[b:]
   s=s.replace("assert sha(private.resolve())=='5fd8c3545da0b3d9adf5a8f76a0101e27e81d51bbfdcdb3eadce59d0d888cabf'","assert sha(private.resolve())==model_meta['library_SHA256']").replace("'library_sha256':'5fd8c3545da0b3d9adf5a8f76a0101e27e81d51bbfdcdb3eadce59d0d888cabf'","'library_sha256':model_meta['library_SHA256']")
  if name.endswith('.py'):ast.parse(s,filename=name)
  b=s.encode()
 (d/name).write_bytes(b)
# The frozen gold bytes retain their original filename/hash; no token edits.
(d/'GLM477-frozen-gold-r8844.json').write_bytes((old/'GLM477-frozen-gold-r8844.json').read_bytes())
(d/'glm_model_library_metadata_r8845.py').write_bytes((r/'glm_model_library_metadata_r8845.py').read_bytes())
for p in d.glob('*.py'):
 if 'r8845'not in p.name:continue
 s=p.read_text().replace('GLM477-frozen-gold-r8845.json','GLM477-frozen-gold-r8844.json');p.write_bytes(s.encode());ast.parse(s)
m=json.loads((old/'GLM-CROSSREQUEST-Model-source-manifest-r8844.json').read_text());m['observer_manifest_SHA256']=''
m['cpu_receipt_path']='/refactor/candidates/glm-PP2-B32-r8843/CPU-build.json'
m['scope']='One load TP8PP2 sameconstructor32/fixedactualadmittedmask B16M8(first16subset) thenB32M8(32distinct),I512O128/currentprefixvector/PVOFF/tile2quantreuseabsent;477fullgold/EOS/QA/userclocks/16normal/audit/post;n12+n13/c64each/240000Meach/35minhard2000inside.'
m['root_GO']='ROOT_FINITE_B32_SAME_GEOMETRY_MECHANICAL_REVISION_PENDING_CURRENT_SHA'
m['I512_fixture']={'path':'results/glm-I512x32-real-inputs-r5732.json','SHA256':m['input_provenance']['SHA256']}
m['CPU_raw_receipt_SHA256']=m['CPU_native_receipt']['SHA256']
observer=json.loads((old/'GLM-transient-Model-observer-manifest-r8844.json').read_text());observer['source_SHA256']={n.replace('r8844','r8845'):sha(d/n.replace('r8844','r8845'))for n in observer['source_SHA256']};op=d/'GLM-transient-Model-observer-manifest-r8845.json';op.write_bytes((json.dumps(observer,indent=2)+'\n').encode())
m['observer_manifest_SHA256']=sha(op);m['source_SHA256']={p.name:sha(p)for p in d.iterdir()if p.is_file()and not p.name.startswith(('dispatch_','GLM-CROSSREQUEST-','review-package'))};m['mechanical_revision']={'prior552':'GLM552-engineering-stop-compact-r8844.json','change':'worker ModelSHA expected from same LIBRARY_RECEIPT as actual path; namespace only, DSO/source/math/geometry/resources unchanged','new_native_compile':False,'new_GPU_submitted':False}
mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8845.json';mp.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode());mh=sha(mp)
entry="import pathlib,sys\nsys.argv=['launch_GLM_prefill_Model_r8845.py',%r]\nexec(compile((pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming/launch_GLM_prefill_Model_r8845.py').read_text(),'launch_GLM_prefill_Model_r8845.py','exec'))\n"%mh;(d/'dispatch_GLM_PP2_B32_Model_r8845.py').write_bytes(entry.encode())
q=json.loads((old/'review-package-r8844.json').read_text());q.update(status='MECHANICAL_FIX_READY_NO_NEW_GPU_GO',manifest_SHA256=mh,entry_SHA256=sha(d/'dispatch_GLM_PP2_B32_Model_r8845.py'),prior_failure='552 FAILED1/7:02;553postempty preserved',remaining='Root same-scope revisionack/sourceclaim/SHA/freshidle/uniquewatch; no compiled or mathchange');(d/'review-package-r8845.json').write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(q))
