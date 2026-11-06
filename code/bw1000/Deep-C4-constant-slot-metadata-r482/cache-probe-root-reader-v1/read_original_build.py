"""Original1116 build reader, including independent passive ELF64 dynsym/NEEDED.
No compiler, subprocess, Native load, SDK, Slurm, or device operation.
"""
from pathlib import Path
import hashlib,json,shlex,struct,re
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Deep-C4-constant-slot-metadata-r482/cache-probe-CPU-v1';B=R/'candidates/Deep-C4-constant-slots-r482-cache-probe-CPU-v1'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def path(p):
 if p.startswith('/refactor/'):return R/p[10:]
 p=Path(p);assert p.is_relative_to(R);return p
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').partition(':')[2]))
def elf(p):
 b=Path(p).read_bytes();assert b[:6]==b'\x7fELF\x02\x01';off=struct.unpack_from('<Q',b,40)[0];stride,n=struct.unpack_from('<HH',b,58);assert stride==64;sects=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*stride)for i in range(n)]
 def string(sec,i):a=sec[4]+i;return b[a:b.index(0,a)].decode()
 symbols={};needed=[];symbolic=False
 for sec in sects:
  if sec[1]==11:
   assert sec[9]==24;strings=sects[sec[6]]
   for pos in range(sec[4],sec[4]+sec[5],24):
    name,info,other,section,value,size=struct.unpack_from('<IBBHQQ',b,pos)
    if name:symbols[string(strings,name)]={'binding':info>>4,'type':info&15,'section':section,'value':value,'size':size}
  if sec[1]==6:
   strings=sects[sec[6]]
   for pos in range(sec[4],sec[4]+sec[5],16):
    tag,value=struct.unpack_from('<qQ',b,pos)
    if tag==1:needed.append(string(strings,value))
    if tag==16 or(tag==30 and value&2):symbolic=True
 return symbols,needed,symbolic

def main():
 rp=B/'CPU-result.json';assert H(rp)=='f381d487d07c58f97197351daea76917bf416ff69d72a0379308f45cd60250f1';q=json.loads(rp.read_bytes());assert q['completed']and q['passed']and q['actual_CPU']=='910407.1116'and q['new_HOST']==1 and q['new_GPU']==0 and q['all_fresh_M_before_compile']and q['all_fresh_M_equal_MD']and not q['unmatched_dependencies']and not q['project_maps']
 assert H(D/'SOURCE_MANIFEST.json')==q['source_manifest_SHA256']=='03a1bb2fe6d09c9936d39085945343bcc5ea35f39455d089d9abda98283a9651'and H(D/'CPU_RESOURCE_CLAIM.json')==q['claim_SHA256']=='d176411ab32eca883469c0b3d441372751def5528c37f22005222dcc73e22b51'
 mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in mf['files_SHA256'].items());p=json.loads((D/'BUILD_PLAN.json').read_bytes());u=q['compiled']['probe'];snapshot=q['fresh_dependencies_SHA256']['probe'];assert deps(path(u['MD']))==deps(B/'probe.before.d')==set(snapshot)and H(path(u['MD']))==u['MD_SHA256'];assert H(path(u['source']))==u['source_SHA256']==mf['files_SHA256']['native-src/probe.cpp'];assert H(path(u['object']))==u['object_SHA256'];headers=[a for a,h in snapshot.items()if h==p['class_header_SHA256']];assert headers and p['forbidden_old_class_header_SHA256']not in snapshot.values();assert all(H(path(a))==p['class_header_SHA256']for a in headers)
 v=q['libraries']['probe'];lib=path(v['path']);assert H(lib)==v['SHA256']=='0ea995d46fbfdc5642fa3ae9ea0a3d0c51733a66d93f12ba66100f55c15f387d';syms,needed,symbolic=elf(lib);item=p['links'][0];assert symbolic and needed==v['NEEDED']and needed.count(item['required_NEEDED'])==1;assert all(a==item['required_NEEDED']or a in item['allowed_other_NEEDED']for a in needed);assert all(syms[a]['binding']==1 and syms[a]['section']!=0 for a in item['required_exports'])
 native=path(item['required_NEEDED']);assert H(native)=='90ba080a7666c35e4160003412be2f0c127b176eba1fc6c5fa0f1eb21ea244b1';nsyms,_,_=elf(native);imports=[a for a,d in syms.items()if'PagedIndexerCache'in a and d['section']==0];assert len(imports)>=4 and all(a in nsyms and nsyms[a]['section']!=0 for a in imports);assert all(any(k in a for a in imports)for k in ['C1E','store','gather','reset']);assert any('Runtime6create'in a and d['section']==0 for a,d in syms.items());assert not any('C4IndexerC1E'in a and d['section']==0 for a,d in syms.items())
 def log(name,stream):
  d=q['logs'][name+'.'+stream];f=path(d['path']);assert H(f)==d['SHA256']and f.stat().st_size==d['bytes'];return f.read_text()
 ldd=log('probe-ldd','stdout')+'\n'+log('probe-ldd','stderr');assert 'not found'not in ldd and 'undefined symbol'not in ldd;resolved=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',ldd,re.M));assert resolved==set(v['all_transitive_resolved_library_SHA256'])==set(p['dynamic_SHA256'])and len(resolved)==69;project={}
 for a,h in v['all_transitive_resolved_library_SHA256'].items():
  assert h==p['dynamic_SHA256'][a]
  if a.startswith('/refactor/'):assert H(path(a))==h;project[a]=h
 link=next(c['argv']for c in q['commands']if c['name']=='probe-link');assert [a for a in link if a.endswith('.o')]==[u['object']]and link.index(u['object'])<link.index(item['required_NEEDED'])<min(i for i,a in enumerate(link)if a.startswith('-l'))
 return dict(schema=1,actual_CPU=q['actual_CPU'],passed=True,receipt_SHA256=H(rp),source_manifest_SHA256=q['source_manifest_SHA256'],claim_SHA256=q['claim_SHA256'],elapsed_inside_s=q['elapsed_s'],source_SHA256=u['source_SHA256'],object_SHA256=u['object_SHA256'],MD_SHA256=u['MD_SHA256'],original_M_first_observed_SHA256=H(B/'probe.before.d'),dependency_count=len(snapshot),live_new_class_headers=headers,new_class_header_SHA256=p['class_header_SHA256'],probe_path=v['path'],probe_SHA256=v['SHA256'],probe_bytes=lib.stat().st_size,Native_SHA256=H(native),new_CABI=item['required_exports'],passive_ELF64_dynsym_NEEDED_verified=True,actual_NEEDED=needed,imported_actual_cache_methods=imports,Runtime_create_imported=True,C4Indexer_or_RankInfo_instantiated=False,transitive_count=69,live_project_DSO_SHA256=project,Native_dlopen=False,SDK_calls=0,GPU_GO=False,Model_GO=False,numeric_qualified=False,limits=['Parent337Nativeexports belong to Native90ba, not newly defined in the nineCAPI probe.','Non-projectheaders/DTKDSO/compiler bytes retain producer snapshots, not a fresh independent fullrehash.','Passive ELF import resolution and newclassMD closure are engineering evidence, not Runtime/cacheaddress/SDKnumeric/performance qualification.'])
if __name__=='__main__':print(json.dumps(main()))
