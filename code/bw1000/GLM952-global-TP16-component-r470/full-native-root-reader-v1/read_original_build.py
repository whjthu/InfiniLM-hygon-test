
from pathlib import Path,PurePosixPath
import os,json,hashlib,shlex,re,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'containers/dtk26-sandbox-r1';H=lambda b:hashlib.sha256(b).hexdigest()
def backed(name):
 if name.startswith('/refactor/'):return R/name[len('/refactor/'):]
 if name.startswith(str(R)+'/'):return Path(name)
 assert name.startswith('/'),name
 pending=list(PurePosixPath(name).parts[1:]);done=[];links=0
 while pending:
  part=pending.pop(0)
  if part in ('','.'):continue
  if part=='..':
   assert done;done.pop();continue
  candidate=S.joinpath(*done,part)
  if candidate.is_symlink():
   links+=1;assert links<40;target=os.readlink(str(candidate))
   if target.startswith('/'):done=[]
   pending=list(PurePosixPath(target).parts[(1 if target.startswith('/')else 0):])+pending
  else:done.append(part)
 return S.joinpath(*done)
O=R/'candidates/GLM-global-TP16-full-native-r470-CPU-v4';p=O/'CPU-result.json';raw=p.read_bytes();assert H(raw)=='bb4db38050b127d92a74d82ae4cef83dd10da400a0e6c8b00c86030a71303c94';q=json.loads(raw);assert q['completed']and q['passed']and q['actual_CPU']=='910407.1087';assert(q['new_HOST'],q['new_GPU'],q['final_object_count'])==(8,5,31)
commands=q['commands'];M=[i for i,v in enumerate(commands)if v['name'].endswith('-M')];C=[i for i,v in enumerate(commands)if v['name'].endswith('-compile')];assert len(M)==len(C)==13 and max(M)<min(C)
records=[];union={};cache={};missing=[];mismatch=[]
for name,v in q['compiled'].items():
 obj=backed(v['object']);assert H(obj.read_bytes())==v['object_SHA256'];md=backed(v['MD']);assert H(md.read_bytes())==v['MD_SHA256'];stem=Path(name).stem;before=O/(stem+'-M.stdout');before_text=before.read_text();paths=lambda text:set(shlex.split(text.replace(chr(92)+chr(10),' ').partition(':')[2]));a=paths(before_text);b=paths(md.read_text());expected=q['fresh_dependencies_SHA256'][name];assert a==b==set(expected);source=R/'incoming/GLM952-global-TP16-component-r470/full-native-source-v1'/name;assert H(source.read_bytes())==v['source_SHA256']
 for path,h in expected.items():
  if path in union:assert union[path]==h
  union[path]=h
 records.append(dict(source=name,object_SHA256=v['object_SHA256'],MD_SHA256=v['MD_SHA256'],fresh_M_SHA256=H(before.read_bytes()),dependency_count=len(expected),M_equals_MD=True))
P=R/'incoming/GLM952-global-TP16-component-r470/full-native-CPU-v4';mf=json.loads((P/'SOURCE_MANIFEST.json').read_bytes());assert H((P/'SOURCE_MANIFEST.json').read_bytes())==q['source_manifest_SHA256'];assert all(H((P/n).read_bytes())==h for n,h in mf['files_SHA256'].items());allowed=json.loads((P/'DEPENDENCY_PINS.json').read_bytes())['dependency_SHA256'];D=R/'incoming/GLM952-global-TP16-component-r470/full-native-source-v1';source_mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert H((D/'SOURCE_MANIFEST.json').read_bytes())==q['parent_source_manifest_SHA256'];assert all(H((D/n).read_bytes())==h for n,h in source_mf['files_SHA256'].items())
for path,h in union.items():
 source_prefix='/refactor/incoming/GLM952-global-TP16-component-r470/full-native-source-v1/'
 if path.startswith(source_prefix):assert source_mf['files_SHA256'].get(Path(path).name)==h
 else:assert allowed.get(path)==h,path
for path,h in q['reused_objects_SHA256'].items():assert H(backed(path).read_bytes())==h
lib=backed(q['library']);assert H(lib.read_bytes())==q['library_SHA256']=='22345abd96774752f5e33f048da3e4a75ff9fb54f88007d72ac2958422db295c';link=next(v['argv']for v in commands if v['name']=='full-Native-link');objects=[x for x in link if x.endswith('.o')];expected_objects=set(q['reused_objects_SHA256'])|{v['object']for v in q['compiled'].values()};assert len(objects)==len(set(objects))==31 and set(objects)==expected_objects;assert len(q['reused_objects_SHA256'])==18
for name in ['full_bridge_tp16.cpp','routed_ffn_tp16.cpp']:
 obj=q['compiled'][name]['object'];assert link.index(obj)<next(i for i,v in enumerate(link)if v.startswith('-l'))
def run(args):
 z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert z.returncode==0,z.stderr[-1000:];return z.stdout
symbols=run(['nm','-D','--defined-only',str(lib)]);exports={l.split()[-1]:l.split()[-2]for l in symbols.splitlines()if len(l.split())>=3};parent_names={l.split()[-1]:l.split()[-2]for l in (O/'parent-exports.stdout').read_text().splitlines()if len(l.split())>=3};strong={k:v for k,v in parent_names.items()if v not in ['W','V','u','w','v']};assert len(strong)==q['original_strong_count']and all(exports.get(k)==v for k,v in strong.items());assert all(exports.get(n)=='T'for n in q['new_CABI'])
elf=run(['readelf','-d',str(lib)]);needed=re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',elf);assert needed==q['NEEDED_unchanged']
ldd=(O/'candidate-ldd.stdout').read_text()+(O/'candidate-ldd.stderr').read_text();assert 'not found'not in ldd and 'undefined symbol:'not in ldd;dynamic_paths=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',ldd,re.M));assert dynamic_paths==set(q['actual_dynamic_dependency_SHA256']);dynmissing=[];dynchecked=0
for path,h in q['actual_dynamic_dependency_SHA256'].items():
 if path.startswith('/refactor/'):
  p=R/path[len('/refactor/'):];assert H(p.read_bytes())==h;dynchecked+=1
 else:dynmissing.append(dict(path=path,scope='Container/mount-backed currentbytes not rehashed by this HOSTaudit; originalactualldd resolved and receiptpins retained'))
print(json.dumps(dict(scope='Independent original13raw-M/MD/31object/ELF and publishedsnapshot binding; livebytes projectDSOs only;0compiler/Runtime/SDK/Slurm/GPU, no broadheader rehash',receipt_SHA256=H(raw),actual_CPU='910407.1087',records=records,headers_unique_paths=len(union),headers_unique_backing_files=len(cache),missing_headers=missing,changed_headers=mismatch,all13_raw_M_MD_objects_passed=True,header_snapshots_match_published_pins=True,current_header_byte_rehash='Not repeated; previous75s readonlyreader timed out during broadcanonicalNFSwalk, originalCPU1551producerbyteproof retained not independentcurrentverification',raw_command_order_all13M_before_compile=True,exact_link_object_set31=True,reused_objects18=True,original_strong_exports_retained=len(strong),new_CABI=len(q['new_CABI']),NEEDED=needed,library_SHA256=H(lib.read_bytes()),dynamic_from_original_ldd=len(dynamic_paths),dynamic_live_bytes_verified=dynchecked,dynamic_unavailable_backing=dynmissing,dynamic_full_current_byteproof=not dynmissing,independent_qualification='Compiler inputs and ELF only; typed CFUNCTYPE remains recording; no Model/fit/numeric/performance',Model_GO=False)))
