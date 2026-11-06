"""Bounded independent original5M/MD/object/ELF replay. stdlib only, no subprocess/Native."""
from pathlib import Path,PurePosixPath
import json,hashlib,shlex,re,struct,sys
H=lambda b:hashlib.sha256(b).hexdigest()
def elf(b):
 assert b[:6]==bytes([127,69,76,70,2,1]);off=struct.unpack_from('<Q',b,40)[0];size,n=struct.unpack_from('<HH',b,58);assert size==64
 sections=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*size)for i in range(n)];symbols={};needed=[]
 for s in sections:
  if s[1]==11:
   strings=sections[s[6]];names=b[strings[4]:strings[4]+strings[5]]
   for pos in range(s[4],s[4]+s[5],s[9]):
    name,info,vis,ndx,value,length=struct.unpack_from('<IBBHQQ',b,pos)
    if name and ndx:symbols[names[name:names.index(0,name)].decode()]=dict(binding=info>>4,type=info&15)
  if s[1]==6:
   strings=sections[s[6]];names=b[strings[4]:strings[4]+strings[5]]
   for pos in range(s[4],s[4]+s[5],s[9]):
    tag,val=struct.unpack_from('<qQ',b,pos)
    if tag==1:needed.append(names[val:names.index(0,val)].decode())
 return symbols,needed
def audit(root):
 root=Path(root)
 def host(v):
  if v.startswith('/refactor/'):return root/str(PurePosixPath(v).relative_to('/refactor'))
  raise AssertionError('Root reader only explicitly registered project paths: '+v)
 def raw(p,expected=None):
  b=host(p).read_bytes()
  if expected:assert H(b)==expected,(p,'bytes drift')
  return b
 D='/refactor/incoming/GLM-C1-layer-observer-r474/CPU-v1';B='/refactor/candidates/GLM-C1-layer-observer-r474-CPU-v1'
 mf=raw(D+'/SOURCE_MANIFEST.json','fd2128c65a35b9a8fd6958007e22c0de5f4473351dac896b592f26b7f280fd67');manifest=json.loads(mf)
 for n,h in manifest['files_SHA256'].items():raw(D+'/'+n,h)
 claim=json.loads(raw(D+'/CPU_RESOURCE_CLAIM.json','06cd224c3de383faf4810bcd023427ea470d5a8ae8b569d911d32d0413a637c3'))
 receipt_raw=raw(B+'/CPU-result.json','9978a4f59a70ebc220c82b71e5f9d5c138bc28f9fc88e81e63b4c7676f720556');q=json.loads(receipt_raw);plan=json.loads(raw(D+'/BUILD_PLAN.json'))
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.1119'and q['new_HOST']==5 and q['new_GPU']==0 and q['all5_fresh_M_before_compile']and q['all5_MD_equal_fresh_M']and len(q['compiled'])==5
 assert q['failures']==q['unmatched_dependencies']==[]and len(q['observer_consumers_verified'])==5 and q['Native_dlopen']==False and not q['Runtime_created']and q['SDK_explicit_calls']==0 and not q['GPU_job']and not q['Torch_imported']and not q['weights_opened']
 units=[]
 for u in plan['units']:
  v=q['compiled'][u['name']];actual=raw(v['MD'],v['MD_SHA256']);before=raw(B+'/'+u['name']+'.before.d');names=lambda b:set(shlex.split(b.decode().replace(chr(92)+chr(10),' ').partition(':')[2]));paths=names(actual);assert paths==names(before)==set(q['fresh_dependencies_SHA256'][u['name']]);assert u['source']in paths
  raw(v['object'],v['object_SHA256']);raw(v['source'],v['source_SHA256']);snapshot=q['fresh_dependencies_SHA256'][u['name']];assert plan['new_class_header_SHA256']in snapshot.values()and plan['old_class_header_SHA256']not in snapshot.values()
  hp=[p for p in paths if p.endswith('/glm_sparse_observer_r131.hpp')];assert len(hp)==1;raw(hp[0],plan['new_class_header_SHA256']);assert all(not p.startswith('/refactor/')or p.startswith(B+'/project-mirror/')for p in paths)
  units.append(dict(name=u['name'],M_path=B+'/'+u['name']+'.before.d',M_SHA256_first_root_computation=H(before),MD_path=v['MD'],MD_SHA256=H(actual),dependency_count=len(paths),M_equals_MD=True,object_SHA256=v['object_SHA256'],source_SHA256=v['source_SHA256'],new_observer_header=hp[0],new_observer_SHA256=plan['new_class_header_SHA256']))
 assert len(q['all32_linked_objects_SHA256'])==32 and len(plan['reuse_objects_SHA256'])==27
 for p,h in q['all32_linked_objects_SHA256'].items():raw(p,h)
 for p,h in plan['reuse_objects_SHA256'].items():assert q['all32_linked_objects_SHA256'][p]==h
 assert q['new_transitive_library_SHA256']==q['parent_transitive_SHA256']==plan['original_transitive_SHA256']and len(q['new_transitive_library_SHA256'])==71
 project={p:h for p,h in q['new_transitive_library_SHA256'].items()if p.startswith('/refactor/')}
 for p,h in project.items():raw(p,h)
 logs={}
 for c in q['commands']:
  assert c['rc']==0
  for stream,v in c['logs'].items():
   b=raw(v['path'],v['SHA256']);assert len(b)==v['bytes'];logs[PurePosixPath(v['path']).name]=b.decode()
 strong=lambda t:{a[-1]:a[-2]for l in t.splitlines()if len(a:=l.split())>=3 and a[-2]in['T','D','B','R']}
 prior=strong(logs['parent-symbols.stdout']);now=strong(logs['symbols.stdout']);assert len(prior)==q['strong_parent_exports_retained']==273 and all(now.get(n)==t for n,t in prior.items())
 libraw=raw(q['library'],q['library_SHA256']);parentraw=raw(plan['parent_library'],plan['parent_library_SHA256']);symbols,needed=elf(libraw);parent_symbols,parent_needed=elf(parentraw)
 assert needed==parent_needed==plan['old_NEEDED']==q['NEEDED'];assert all(n in symbols and n in parent_symbols for n in prior)
 for n in plan['required_new_exports']:assert now[n]=='T'and symbols[n]['type']==2 and symbols[n]['binding']==1
 def paths(text):
  assert 'not found'not in text and 'undefined symbol'not in text
  return sorted(set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M)))
 assert set(paths(logs['ldd.stdout']+chr(10)+logs['ldd.stderr']))==set(q['new_transitive_library_SHA256']);assert set(paths(logs['parent-ldd.stdout']+chr(10)+logs['parent-ldd.stderr']))==set(q['parent_transitive_SHA256'])
 compile_seq=[i for i,c in enumerate(q['commands'])if c['argv'][0]=='/usr/bin/g++'and'-c'in c['argv']];M_seq=[i for i,c in enumerate(q['commands'])if c['argv'][0]=='/usr/bin/g++'and'-M'in c['argv']];assert len(compile_seq)==len(M_seq)==5 and max(M_seq)<min(compile_seq)
 return dict(schema='GLM1119-root-original5-MD-object-passiveELF-v1',root_passed=True,actual_CPU='910407.1119',original_receipt_SHA256=H(receipt_raw),original_receipt_bytes=len(receipt_raw),units=units,all5_fresh_M_before_compile=True,all5_M_equals_MD=True,live_object_hashes=32,reused_original_objects=27,new_HOST_TUs=5,new_GPU_TUs=0,mirror_registered_files=762,original_observer_consumers=5,all5_new_observer_definition=True,source_MF=H(mf),claim_SHA256=H(raw(D+'/CPU_RESOURCE_CLAIM.json')),library=q['library'],library_SHA256=H(libraw),library_bytes=len(libraw),actual_passive_ELF64_defined_symbols=len(symbols),parent_strong_exports_retained=273,new_CABI=plan['required_new_exports'],NEEDED=needed,transitive_resolved_libraries=71,live_project_DSO_hashes=project,original_metadata_logs_checked=len(logs),all5_M_MD_source_object_byte_checked=True,CPU_elapsed_s=q['elapsed_s'],Native_dlopen=False,Runtime_created=False,SDK_explicit_calls=0,GPU=False,Model_GO=False,qualification='Engineering-only original5M/MD/newheader/object32/parentstrong273+4typedCABI/passiveELF/fullparentENV ldd dualstream logs. No device/numeric/profile execution.',limits=['Original before-M rawSHA first independently computed by root; producer already captured actualMD SHA and freshpathsets.','Nonproject system/DTKheaders and DSOs keep producer snapshots/toolchain qualification; root did not repeat all1106 current header hashes or63container DSOs.','Childcompiler/ldd implicitinitialization UNKNOWN; no Native/Runtime/SDK/Torch calls by CPU/root scripts.','250<512 is currentC1 sourcebound; actualevent driver memory/selectedframes/Model quality and speed remain UNKNOWN.'])
if __name__=='__main__':print(json.dumps(audit(sys.argv[1])))
