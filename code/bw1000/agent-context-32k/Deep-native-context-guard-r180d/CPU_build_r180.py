"""Guard-only HOST closure. No Native dlopen, Kernel rebuild, weights or GPU."""
import pathlib,json,hashlib,subprocess,shlex,os,time,traceback,re
from CPU_contract_r180 import check
from dependency_mirror_r180d import Mirror,spelled_deps
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-native-context-guard-r180d';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic();deadline=start+900
assert not B.exists();B.mkdir();q=dict(completed=False,CPU_only=True,GPU_initialized=False,Native_loaded=False,weights_opened=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
OLD_MESSAGE=b'Native sparse cache requires rows1..128 and capacity1..4096';NEW_MESSAGE=b'Native sparse cache requires rows1..128 and capacity1..65536'
def run(a,label,limit=120):
 assert time.monotonic()<deadline,'CPU900s hard deadline';q['commands'].append(a)
 z=subprocess.run(a,text=True,capture_output=True,timeout=min(limit,deadline-time.monotonic()));(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-2500:]);return z.stdout
def prefix(a):
 out=[];i=0
 while i<len(a):
  x=a[i]
  if x in ['-o','-MF','-MT','-MQ']:i+=2;continue
  if x in ['-c','-MD','-MMD','-M','-MM','-shared']or x.endswith('.cpp')or x.endswith('.o'):i+=1;continue
  out.append(x);i+=1
 assert pathlib.Path(out[0]).name in ['g++','c++','clang++']
 assert not any(x.startswith(('-fplugin','-specs','-wrapper'))for x in out)
 return out
def deps(s):return {str(pathlib.Path(p).resolve())for p in shlex.split(s.replace('\\\n',' ').split(':',1)[1])}
def dep_rows(u):
 out={}
 for v in u.get('dependencies',[]):
  if isinstance(v,dict):
   path=v.get('path',v.get('file'));h=v.get('sha256',v.get('SHA256'))
   if path and h:out[path]=h
 return out
def find_records(x,out):
 if isinstance(x,dict):
  obj=x.get('object',x.get('object_path'));src=x.get('source',x.get('src'));h=x.get('source_sha256',x.get('source_SHA256'));oh=x.get('object_sha256',x.get('object_SHA256'))
  if isinstance(obj,str)and isinstance(src,str)and src.endswith('.cpp')and pathlib.Path(src).is_file()and pathlib.Path(obj).is_file():
   h=h or sha(src);oh=oh or sha(obj)
  if isinstance(obj,str)and isinstance(src,str)and src.endswith('.cpp')and isinstance(h,str):
   out.append(dict(object=obj,source=src,source_SHA256=h,object_SHA256=oh,compile_argv=x.get('compile_argv',x.get('argv'))))
  for v in x.values():find_records(v,out)
 elif isinstance(x,list):
  if x and all(isinstance(v,str)for v in x)and '-c'in x and '-o'in x:
   sources=[v for v in x if v.endswith('.cpp')]
   if len(sources)==1 and pathlib.Path(sources[0]).is_file():
    obj=x[x.index('-o')+1]
    if pathlib.Path(obj).is_file():out.append(dict(object=obj,source=sources[0],source_SHA256=sha(sources[0]),object_SHA256=sha(obj),compile_argv=x))
  for v in x:find_records(v,out)
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 m=json.loads((D/'manifest-r180.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 q['source_contract']=check();c=json.loads((D/'source-context-r180.json').read_bytes())
 cp=R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';assert sha(cp)==c['current_CPU_receipt_SHA256'];parent=json.loads(cp.read_bytes());old=c['current_link']['objects'];assert old==parent['link']['full']['objects']and len(old)==48 and all(sha(n)==h for n,h in old.items());assert sha(parent['libraries']['full']['path'])==c['Native_parent_SHA256'];assert sha(c['original_header_path'])==c['original_header_current_SHA256'];assert sha(c['request_metadata_header_path'])==c['request_metadata_header_SHA256']
 for n,h in parent['actual_DTK_bindings'].items():assert sha(n)==h['SHA256']
 rp=R/'results/deep-c4-projection-model-r5920-build.json';assert sha(rp)=='9fd84fb279396985f8ad2fe56a4f081adeccad823e95ca1dd881bfdd82706c70';base=json.loads(rp.read_bytes());assert base['completed']and len(base['units'])==37
 script=R/'candidates/deep-c4-projection-model-r5920/compile.sh';assert sha(script)=='991eb415e5abb123cafe68dedf5ae067ad9e020f5b9a773ff08e5c5be4b41c2e'
 text=script.read_text().replace('\\\n',' ');vars={'BASE':str(script.parent),'LM':str(R/'candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM'),'COMMON':str(R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM'),'SDK':str(R/'install/hygon-deep-5f91ad7-r3672'),'HELPER':str(R/'candidates/models-5f91ad7-r3675/Deep/helpers'),'ABI':'1'}
 for k,v in vars.items():text=text.replace('${'+k+'}',v).replace('$'+k,v)
 a=shlex.split(text);assert not any('$'in x for x in a);baseflags=prefix(a)
 known={u['object']:u for u in c['known_units']};mapping={}
 # Reuse the sealed baseline unit's real source/object/backtime dep records.
 for u in base['units']:
  if u['object']in old:
   assert old[u['object']]==u['object_sha256'];mapping[u['object']]=dict(object=u['object'],source=u['source'],source_SHA256=u['source_sha256'],object_SHA256=u['object_sha256'],compile_argv=baseflags,exact_compile_context=False,backtime_dependencies=dep_rows(u))
 for obj,u in known.items():
  if obj in old:
   assert old[obj]==u['object_SHA256'];mapping[obj]=dict(u,exact_compile_context=True,backtime_dependencies=mapping.get(obj,{}).get('backtime_dependencies',{}))
 # Exact original7169 receipt is known from the retained build script.
 combined_path=R/'results/deep-combined-r7169-build.json';combined=json.loads(combined_path.read_bytes());assert combined['completed']and combined['parent_manifest_SHA256']=='9fd84fb279396985f8ad2fe56a4f081adeccad823e95ca1dd881bfdd82706c70'
 expected7169={
  'unit-2.o':('deepseek_v4_attention_output.cpp','31e80fb1109308809a030896681ceacc17fbe5b76f1a61b528389456f683f960','b6d87d042a48665fee3310249506a3b1f6768001fc3b564f0a4fda69afa4dd99'),
  'unit-3.o':('deepseek_v4_hyper_connection_r177.cpp','4b178cd98cd9e8dd2e24614bf022c1297d6332f22b2c790530fbb06866c5d313','3269f7e6aea58fcfd961ab26b0d85841be36ffe12bb2b6fa4525da80264ef558'),
  'unit-4.o':('deep_model_layout_api_r7052.cpp','f8334083065dd6245322fbe76aefcbcea4443009aed090e2ad6e99ff2bb40ff9','46cc4ce3cc47a8d651c3d95cfb964edc7f65b0ee0e269366e2b3218b99515b3f'),
 }
 for object_name,(source_name,source_sha,object_sha)in expected7169.items():
  obj=str(R/'candidates/deep-combined-r7169/objects'/object_name);src=str(R/'candidates/deep-combined-r7169/incoming'/source_name);unit=next(x for x in combined['compiled_units']if x['object']==obj)
  assert unit['source']==src and unit['source_SHA256']==source_sha and unit['object_SHA256']==object_sha and old[obj]==object_sha and sha(src)==source_sha and sha(obj)==object_sha
  assert combined['full_link_objects'][obj]==object_sha
  # This reproduces7169's added private include for classification. If any
  # unexpectedly consumes the changed Config header, exact-context recovery
  # remains required; never rebuild it speculatively.
  flags=[baseflags[0],'-I'+str(pathlib.Path(src).parent),*baseflags[1:]]
  mapping[obj]=dict(object=obj,source=src,source_SHA256=source_sha,object_SHA256=object_sha,compile_argv=flags,exact_compile_context=False,backtime_dependencies=unit['deps'])
 q['combined7169_mapping']={'receipt':str(combined_path),'SHA256':sha(combined_path),'parent_manifest_SHA256':combined['parent_manifest_SHA256'],'source_object_SHA_checked':True,'classification_only_no_recompile_permission':True}
 # Inspect receipt metadata beside later replacement objects; never execute it.
 records=[];metadata=[]
 directories={pathlib.Path(n).parent for n in old if n not in mapping};directories|={p.parent for p in list(directories)if p.name=='objects'}
 metadata_files={file for directory in directories for file in directory.glob('*.json')}
 # Exact known provenance paths only. Never glob/scandir the huge results root.
 exact_metadata=[
  R/'results/deep-owner-device-CPU-r8918.json',
  R/'results/deep-attention-inner-CPU-r9193a.json',
  R/'candidates/deep-c4-smallt-r9252/CPU-build.json',
  R/'candidates/deep-mode1-sink-r9207/CPU-build.json',
 ]
 metadata_files.update(file for file in exact_metadata if file.is_file())
 for file in sorted(metadata_files):
  if not re.search(r'CPU|build|manifest|source|binding|producer|compile',file.name,re.I)or file.stat().st_size>16*1024**2:continue
  try:data=json.loads(file.read_bytes())
  except (ValueError,UnicodeError):continue
  find_records(data,records);metadata.append({'path':str(file),'SHA256':sha(file)})
 for u in records:
  obj=u['object']
  if obj in old and obj not in mapping and u['object_SHA256']==old[obj]and pathlib.Path(u['source']).is_file():
   mapping[obj]=dict(u,compile_argv=u['compile_argv']if isinstance(u['compile_argv'],list)else baseflags,exact_compile_context=isinstance(u['compile_argv'],list),backtime_dependencies={})
 # For a source-named replacement object, current source can be used only to
 # classify deps. It is never eligible for recompile without exact known pins.
 for obj in old:
  if obj in mapping:continue
  parentdir=pathlib.Path(obj).parent;name=pathlib.Path(obj).name.removesuffix('.o')
  if not name.endswith('.cpp'):name+='.cpp'
  candidates=[p for p in [parentdir/name,parentdir/'src'/name,parentdir/'incoming'/name]if p.is_file()]
  if len(candidates)==1:
   src=candidates[0];f=[baseflags[0],'-I'+str(src.parent),'-I'+str(parentdir/'src'),'-I/refactor/candidates/deep-owner-events-r8900/src',*baseflags[1:]]
   mapping[obj]=dict(object=obj,source=str(src),source_SHA256=sha(src),object_SHA256=old[obj],compile_argv=f,exact_compile_context=False,backtime_dependencies={})
 q['replacement_receipt_metadata']=metadata
 scans=[];affected=[];unresolved=[]
 # Current private header is not relabelled as original object backtime input.
 for i,(obj,h)in enumerate(old.items()):
  symbols=run(['nm','-C',obj],'symbols-'+str(i),30);has_type='NativeSparseCacheConfig'in symbols;has_old=OLD_MESSAGE in pathlib.Path(obj).read_bytes();u=mapping.get(obj)
  if u:
   assert sha(u['source'])==u['source_SHA256'],('current source differs from pinned TU',obj)
   flags=prefix(u['compile_argv']);raw_M=run(flags+['-M',u['source']],'M-old-'+str(i),60);ds=deps(raw_M);u['spelled_dependencies']=spelled_deps(raw_M);consumes=any(pathlib.Path(n).name=='native_sparse_cache_r22.hpp'for n in ds)
   if consumes:
    assert u['exact_compile_context'],('header consumer lacks exact original compile context',obj)
    old_headers={n for n in ds if pathlib.Path(n).name=='native_sparse_cache_r22.hpp'};assert len(old_headers)==1
    u['resolved_original_header']=next(iter(old_headers));assert sha(u['resolved_original_header'])==c['original_header_current_SHA256']
    affected.append((u,flags,ds))
   assert consumes or not(has_type or has_old),('class/guard object absent from current source deps',obj)
   scans.append(dict(object=obj,source=u['source'],source_SHA256=u['source_SHA256'],fresh_dependency_count=len(ds),exact_compile_context=u['exact_compile_context'],nonexact_context_purpose='Classification only; never eligible to recompile a guard consumer',header_consumer=consumes,NativeSparseCacheConfig_symbols=has_type,old_guard_message=has_old,recorded_backtime_header={n:h for n,h in u['backtime_dependencies'].items()if pathlib.Path(n).name=='native_sparse_cache_r22.hpp'}))
  else:
   # Opaque GPU/helper .o inputs are never compiled. They may not own ANY
   # class constructor/copy/type/vtable/guard code. Unknown HOST .cpp.o fails.
   recognized_gpu=(obj.endswith('.cu.o')or('/Deep/helpers/'in obj and pathlib.Path(obj).name in ['routed.o','sinkhorn.o','gather.o','fused.o','post.o','scatter.o','aux_glm_routed.o']))
   if not recognized_gpu or has_type or has_old:unresolved.append(dict(object=obj,NativeSparseCacheConfig_symbols=has_type,old_guard_message=has_old))
   scans.append(dict(object=obj,opaque_gpu_helper=recognized_gpu,NativeSparseCacheConfig_symbols=has_type,old_guard_message=has_old))
 q['all_object_scans']=scans;q['unresolved_objects']=unresolved
 assert not unresolved,('unresolved current HOST source/context; no partial build',unresolved)
 assert affected and all(u['object']in known for u,_,_ in affected)
 # CPU-only real metadata constructor/copy tests, compiled against actual base.
 fixture_flags=[x for x in baseflags if x.startswith(('-I','-D','-std=','-O'))]+['-pthread']
 fixtures={}
 for label,include,maximum in [('old',str(pathlib.Path(c['original_header_path']).parent),4096),('new',str(D/'include'),65536)]:
  exe=B/('config-'+label);md=B/('config-'+label+'.d');f=['g++','-I'+include,*fixture_flags,'-DEXPECT_CAP_MAX='+str(maximum)];source=str(D/'config_contract_r180.cpp');ds=deps(run(f+['-M',source],'fixture-'+label+'-M',60));assert {n for n in ds if pathlib.Path(n).name=='native_sparse_cache_r22.hpp'}=={str((pathlib.Path(include)/'native_sparse_cache_r22.hpp').resolve())};assert str(pathlib.Path(c['request_metadata_header_path']).resolve())in ds;pins={n:sha(n)for n in ds};argv=f+['-MD','-MF',str(md),source,'-o',str(exe)];run(argv,'fixture-'+label+'-compile',90);assert deps(md.read_text())==ds and all(sha(n)==h for n,h in pins.items())
  dynamic=run(['readelf','-d',str(exe)],'fixture-'+label+'-dynamic',30);needed=set(re.findall(r'Shared library: \[(.*?)\]',dynamic));assert needed<= {'libstdc++.so.6','libgcc_s.so.1','libc.so.6','libm.so.6'},needed
  fixtures[label]=json.loads(run([str(exe)],'fixture-'+label+'-run',15));assert fixtures[label]['passed']and fixtures[label]['cases']==35
 assert fixtures['old']['class_size']==fixtures['new']['class_size'];q['metadata_class_fixture']=fixtures
 compiled=[];spelled=set().union(*(u['spelled_dependencies']for u,_,_ in affected));mirror=Mirror(R,B,spelled,D/'include/native_sparse_cache_r22.hpp',c['original_header_current_SHA256']);q['project_dependency_mirror']=mirror.summary
 for i,(u,flags,original_deps)in enumerate(affected):
  src=mirror.target(u['source']);assert src.is_file()and sha(src)==u['source_SHA256']
  # Ancestor relative includes now stay inside the same private project tree.
  f=mirror.rewrite_flags(flags);obj=B/(str(i)+'.o');md=B/(str(i)+'.d');ds=deps(run(f+['-M',str(src)],'M-new-'+str(i),60))
  resolved={n for n in ds if pathlib.Path(n).name=='native_sparse_cache_r22.hpp'};assert resolved and all(sha(n)==c['new_header_SHA256']for n in resolved),resolved
  assert mirror.normalize_and_check(ds)=={mirror.logical(n)for n in original_deps},('canonical dependency graph changed',u['source'])
  pins={n:sha(n)for n in ds};cmd=f+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)];run(cmd,'compile-'+str(i),180);assert deps(md.read_text())==ds and all(sha(n)==h for n,h in pins.items())
  mirror.normalize_and_check(ds);compiled.append(dict(source=str(src),source_SHA256=sha(src),original_source=u['source'],replaces=u['object'],object=str(obj),object_SHA256=sha(obj),compile_argv=cmd,dependency_SHA256=pins,fresh_M_equal_MD=True,original_dependency_graph_equal_after_private_mapping=True))
 replace={u['replaces']:u['object']for u in compiled};assert all(not r['NativeSparseCacheConfig_symbols']and not r['old_guard_message']for r in scans if r['object']not in replace)
 link=[replace.get(x,x)for x in c['current_link']['argv']];lib=B/'libDeep_native_context_guard_r180.so';link[link.index('-o')+1]=str(lib);run(link,'link',120);binary=lib.read_bytes();assert NEW_MESSAGE in binary and OLD_MESSAGE not in binary
 oldnm=run(['nm','-D','--defined-only',parent['libraries']['full']['path']],'old-exports',30);newnm=run(['nm','-D','--defined-only',str(lib)],'new-exports',30);names=lambda s:{x.split()[-1]:x.split()[-2]for x in s.splitlines()if len(x.split())>=3};a,b=names(oldnm),names(newnm);missing=set(a)-set(b);knownweak={'_ZNK3fmt3v1217basic_format_argsINS0_7contextEE3getEi','_ZZN3fmt3v126detail12format_floatIdEEiT_iRKNS0_12format_specsEbRNS1_6bufferIcEEENKUljPcE_clEjSA_'};assert missing<=knownweak and all(a[n]=='W'for n in missing),missing
 dynamic=run(['readelf','-d',str(lib)],'dynamic',30);assert 'SYMBOLIC'in dynamic;ld=run(['ldd','-r',str(lib)],'ldd-r',60);assert 'not found'not in ld and 'undefined symbol:'not in ld
 assert all(sha(n)==h for n,h in old.items())and sha(c['original_header_path'])==c['original_header_current_SHA256']
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),HOST_TUs_compiled=len(compiled),compiled=compiled,link=link,reused_objects_SHA256={n:h for n,h in old.items()if n not in replace},all_old_guard_code_removed=True,exact_weak_export_exceptions={n:a[n]for n in missing},actual_DTK_bindings=parent['actual_DTK_bindings'],source_manifest_SHA256=sha(D/'manifest-r180.json'),new_Native_GPU_qualification=False,scope='Guard-only current-source HOST/object/dependency/link + real CPU metadata-class fixture. No GPU create/weights/kernel/long-prefill/agent/quality qualification.')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_step','HOST_TUs_compiled','GPU_TUs_compiled','library_SHA256','error','seconds']}));assert q['completed']
