import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';cp=r/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert hashlib.sha256(cp.read_bytes()).hexdigest()=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';q=json.loads(cp.read_bytes());pins={**q['parent_actual_dependency_SHA256'],**q['dependency_SHA256']};sdk=r/'install/hygon-glm-5f91ad7-r3672';lib=sdk/'lib/libinfiniops.so';h=hashlib.sha256(lib.read_bytes()).hexdigest();assert h=='0bd37990340f39bbcaf185abbc990ebeb9f56ea1b9992327eda96d627446a863'
out={'CPU_only':True,'GPU_initialized':False,'SDK_library_SHA256':h,'files':{},'directories':{}}
for suffix in ['include/infini/operator_call_instantiations.h','include/tensor.h','include/config.h']:
 p=sdk/suffix;name='/refactor'+str(p)[len(str(r)):];raw=p.read_bytes();hh=hashlib.sha256(raw).hexdigest();pin=pins.get(name);assert pin==hh,(name,pin,hh);lines=raw.decode(errors='replace').splitlines();idx=[i for i,s in enumerate(lines) if any(k in s for k in ['RmsNorm','RMSNorm','Cache','cache','HashCombine','data_','operator==','hash()'])];out['files'][name]={'SHA256':hh,'contexts':[{'line':i+1,'text':'\n'.join(lines[max(0,i-5):i+12])} for i in idx[:50]]}
for rel in ['candidates/rebased-5f91ad7-r3668/Common/GLM','candidates/rebased-5f91ad7-r3668/Model/GLM']:
 p=r/rel;out['directories'][rel]=[x.name for x in p.iterdir()][:40]
z=subprocess.run(['nm','-D','--defined-only',str(lib)],capture_output=True,text=True,timeout=30);out['nm_returncode']=z.returncode;out['cache_RMS_symbols']=[s for s in z.stdout.splitlines() if 'RmsNorm' in s][:150];out['nm_error']=z.stderr
proof=r/'results/latest-sdk-r3672-GLM-build.json';out['SDK_build']=json.loads(proof.read_bytes()) if proof.exists() else None
print(json.dumps(out))
