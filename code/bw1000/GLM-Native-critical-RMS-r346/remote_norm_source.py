import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';old=pathlib.Path.home()/'hygon-inference-20261013';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def host(name):return pathlib.Path(str(name).replace('/refactor',str(r),1).replace('/old',str(old),1))
cp=r/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(cp)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';q=json.loads(cp.read_bytes());assert q['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00';assert sha(host(q['library']))==q['library_SHA256']
out={'CPU_only':True,'GPU_initialized':False,'Native_library':q['library'],'Native_SHA256':q['library_SHA256'],'CPU_receipt_SHA256':sha(cp),'CPU_keys':list(q),'compiled':q['compiled'],'reused_objects':list(q['reused_objects_SHA256']),'dependency_paths':list(q['parent_actual_dependency_SHA256']),'norm_matches':{},'MD_paths':[]}
pins=q['parent_actual_dependency_SHA256']
for root in [r/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/csrc',r/'install/hygon-glm-5f91ad7-r3672/include']:
 for p in root.rglob('*'):
  if p.suffix not in ['.hpp','.h','.cc','.cpp']:continue
  text=p.read_text(errors='replace')
  if not any(x in text for x in ['RMSNorm','rms_norm','rmsNorm','RmsNorm']):continue
  lines=text.splitlines();indices=[i for i,s in enumerate(lines) if any(k in s for k in ['RMSNorm','rms_norm','rmsNorm','RmsNorm'])];decl='/refactor'+str(p)[len(str(r)):];pin=pins.get(decl);out['norm_matches'][decl]={'SHA256':sha(p),'pin_expected':pin,'pin_matches':pin==sha(p) if pin else None,'text':text if len(text)<45000 else None,'context':[{'line':i+1,'text':'\n'.join(lines[max(0,i-5):i+12])} for i in indices[:20]]}
print(json.dumps(out))
