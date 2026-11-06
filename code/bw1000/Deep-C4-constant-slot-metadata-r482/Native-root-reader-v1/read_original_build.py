"""Bounded original1113 MD/object/ELF reader; stdlib, no compiler or SDK.
R project bytes and all50 linked objects are independently rehashed. Original
container/system headers and non-project DTK DSOs retain producer snapshots.
"""
from pathlib import Path
import hashlib,json,posixpath,re,shlex
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL='Deep-C4-constant-slots-r482-Native-CPU-v1'
D=R/'incoming/Deep-C4-constant-slot-metadata-r482/Native-CPU-v1';B=R/'candidates'/LABEL
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def project(p):
 if p.startswith('/refactor/'):return R/p[len('/refactor/'):]
 p=Path(p);assert p.is_absolute()and p.is_relative_to(R);return p

def dep_names(p):
 text=Path(p).read_text().replace('\\\r\n',' ').replace('\\\n',' ');names=set()
 for line in text.splitlines():
  if line.strip():a,colon,b=line.partition(':');assert colon;names.update(shlex.split(b))
 assert names;return names

def strong(text):
 result={}
 for line in text.splitlines():
  row=line.split()
  if len(row)>=3 and row[-2]in ['T','D','B','R']:result[row[-1]]=row[-2]
 return result

def main():
 rp=B/'CPU-result.json';assert H(rp)=='1d3938d699393a85ccddc65a6aeb34a687db14447c4b5ba8861386174d029a6b';q=json.loads(rp.read_bytes());assert q['completed']and q['passed']and q['actual_CPU']=='910407.1113'and q['new_HOST']==7 and q['new_GPU']==12 and q['all19_fresh_M_before_compile']and q['all19_MD_equal_fresh_M']and not q['failures']and not q['unmatched_dependencies']and not q['parent_project_maps']
 assert q['source_manifest_SHA256']==H(D/'SOURCE_MANIFEST.json')=='d3d4d0d52f0f66363fb29a7ccb02b5602c9cb9594c55c08f855aeda64ac60e43'and q['claim_SHA256']==H(D/'CPU_RESOURCE_CLAIM.json')=='9e9d1d9624bafd76e87c6ff6735c1fb6da8e5a7e39858ad75ee6916cc877949d'
 mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==v for n,v in mf['files_SHA256'].items());plan=json.loads((D/'BUILD_PLAN.json').read_bytes());assert q['source_v2_MF']==plan['source_v2_manifest_SHA256']=='4f702cbf48306daa9133d78a9e7fe402ebd6a49f587a936e73af205be50be515'
 units=[];class_units=[];objects={}
 for u in plan['units']:
  name=u['name'];c=q['compiled'][name];before=B/(name+'.before.d');md=project(c['MD']);snapshot=q['fresh_dependencies_SHA256'][name]
  assert dep_names(before)==dep_names(md)==set(snapshot)and H(md)==c['MD_SHA256'];assert H(project(c['source']))==c['source_SHA256']==u['source_SHA256'];assert c['original_object']==u['original_object']and c['kind']==u['kind'];obj=project(c['object']);assert H(obj)==c['object_SHA256'];objects[c['object']]=c['object_SHA256']
  if u['changes_class_definition']:
   assert plan['new_class_header_SHA256']in snapshot.values()and plan['old_class_header_SHA256']not in snapshot.values();class_units.append(name)
  else:assert plan['new_class_header_SHA256']not in snapshot.values()and plan['old_class_header_SHA256']not in snapshot.values()
  units.append(dict(name=name,kind=c['kind'],original_object=u['original_object'],source_SHA256=c['source_SHA256'],object_SHA256=c['object_SHA256'],MD_SHA256=c['MD_SHA256'],original_M_observed_SHA256=H(before),dependency_count=len(snapshot),changes_class_definition=u['changes_class_definition']))
 assert len(units)==19 and set(class_units)==set(q['class_consumers_verified'])and len(class_units)==5
 for p,h in plan['reuse_objects_SHA256'].items():assert H(project(p))==h;objects[p]=h
 assert len(objects)==50 and objects==q['all50_linked_objects_SHA256']
 link=next(c for c in q['commands']if'-shared'in c['argv']);assert link['rc']==0 and set(p for p in link['argv']if p.endswith('.o'))==set(objects)
 actual_headers=[]
 for p,h in q['mirror_file_SHA256'].items():
  if h==plan['new_class_header_SHA256']:assert H(project(p))==h;actual_headers.append(p)
 assert len(actual_headers)==3
 def original_log(label,stream):
  c=next(v for v in q['commands']if v['logs']['stdout']['path'].endswith('/'+label+'.stdout'));assert c['rc']==0
  info=c['logs'][stream];p=project(info['path']);assert H(p)==info['SHA256']and p.stat().st_size==info['bytes'];return p.read_text()
 now=strong(original_log('symbols','stdout'));old=strong(original_log('parent-symbols','stdout'));assert all(now.get(n)==t for n,t in old.items())and len(old)==q['strong_parent_exports_retained'];assert all(now.get(n)=='T'for n in plan['required_new_exports'])
 n=lambda t:re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',t);elf=original_log('ELF','stdout');assert n(elf)==n(original_log('parent-ELF','stdout'))==q['NEEDED']==plan['old_NEEDED']and'SYMBOLIC'in elf
 dynamic_out=original_log('ldd','stdout')+'\n'+original_log('ldd','stderr');assert 'not found'not in dynamic_out and 'undefined symbol'not in dynamic_out;assert q['new_transitive_library_SHA256']==q['parent_transitive_SHA256'];paths=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',dynamic_out,re.M));assert paths==set(q['new_transitive_library_SHA256'])
 checked_project={}
 for p,h in q['new_transitive_library_SHA256'].items():
  if p.startswith('/refactor/')or p.startswith(str(R)+'/'):assert H(project(p))==h;checked_project[p]=h
 library=project(q['library']);assert H(library)==q['library_SHA256']
 return dict(schema=1,actual_CPU=q['actual_CPU'],passed=True,scope='Original19freshM/MD sets and live50objects/new3classheaders/projectDSOs/Native bytes plus originaldualstreamELF/symbol/loader logs independently checked. No Native/SDK/compiler/Slurm/GPU execution.',receipt_SHA256=H(rp),source_manifest_SHA256=q['source_manifest_SHA256'],claim_SHA256=q['claim_SHA256'],source_v2_MF=q['source_v2_MF'],library=q['library'],library_SHA256=q['library_SHA256'],library_bytes=library.stat().st_size,new_HOST=7,new_GPU_translation_units=12,compiled_units=units,reuse_objects=31,all50_object_SHA256=objects,class_consumers=class_units,new_class_header_copies=actual_headers,new_class_header_SHA256=plan['new_class_header_SHA256'],old_class_header_seen_in_changed_TUs=False,parent_strong_exports_retained=len(old),new_CABI=plan['required_new_exports'],NEEDED=q['NEEDED'],resolved_transitive_count=len(paths),project_DSO_live_SHA256=checked_project,elapsed_inside_s=q['elapsed_s'],Native_dlopen=False,SDK_explicit_calls=0,Model_qualified=False,numeric_qualified=False,GPU_GO=False,limits=['No independent current non-projectcontainer/header/DTKDSO rehash. Parent/new producerpath/hash snapshot and qualifiedtoolchain preserved.', 'Original_M_observed_SHA256 first computed by thisreader; producer saved MD SHA and completefreshdependency sets.', 'Class-layout source consumers consistently rebuilt, not runtimeTensor/SDKnumeric qualification; old14missingMD objects were freshly compiled, no blanketreuse.', 'Metadata childimplicitinitialization UNKNOWN, no explicitNative/SDK calls; GPUsource TU compilation is not card compute or a performance result.'])
if __name__=='__main__':print(json.dumps(main()))
