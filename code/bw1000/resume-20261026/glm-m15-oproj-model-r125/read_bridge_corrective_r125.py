import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');host=lambda s:pathlib.Path(s.replace('/refactor',str(r),1));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={'scope':'boundedcurrentc650namedHOSTmetadata; noassertonoldsourcebeforeexport','source_copies':{}}
mp=r/'candidates/glm-M16-r8861/CPU-build.json';q=json.loads(mp.read_bytes());rec=next(x for x in q['compiled']if x['object']=='/refactor/candidates/glm-M16-r8861/private-3.o');out.update(M16_receipt_SHA256=sha(mp),bridge_record=rec);cmd=next(a for a in q['commands']if '-c'in a and rec['source']in a);out['bridge_compile_command']=cmd;md=host(cmd[cmd.index('-MF')+1]);out['bridge_MD_SHA256']=sha(md);out['bridge_MD_text']=md.read_text()
for name in [rec['source'],'/refactor/candidates/glm-exact-budget-r8854/src/modern_glm_cached_model_bridge_r809.cpp','/refactor/candidates/glm-M16-r8861/tree/candidates/glm-exact-budget-r8854/src/modern_glm_cached_model_bridge_r809.cpp']:
 p=host(name)
 if p.is_file():raw=p.read_bytes();assert len(raw)<180000;out['source_copies'][name]={'SHA256':sha(p),'text':raw.decode('utf8')}
pp=r/'candidates/GLM-down-row2-Model-r9284/CPU-build.json';out['current_parent_receipt_SHA256']=sha(pp);out['current_parent_receipt']=json.loads(pp.read_bytes())
old=r/'candidates/glm-transient-layer-r5681/CPU-build.json';a=json.loads(old.read_bytes());rr=next(x for x in a['compiled']if x['object'].endswith('/private-15.o'));out['runtime_record']=rr;out['runtime_compile_command']=next(c for c in a['commands']if '-c'in c and rr['source']in c)
out['bridge_object_actual_SHA256']=sha(host(rec['object']));print(json.dumps(out,separators=(',',':')))
