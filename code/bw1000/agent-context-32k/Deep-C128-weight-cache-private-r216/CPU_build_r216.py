"""Explicit CPU-only HOST adaptation to actual232; no launch, dlopen or GPU."""
import pathlib,json,hashlib,subprocess,shlex,os,time,traceback,sys
from CPU_static_r216 import check
from dependency_mirror_r180d import Mirror,spelled_deps
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-C128-weight-cache-r216';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic();deadline=start+900
assert not B.exists();B.mkdir();q=dict(completed=False,CPU_only=True,GPU_initialized=False,Native_loaded=False,weights_opened=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,commands=[],actual_step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''))
def run(argv,label,limit=90):
 left=deadline-time.monotonic();assert left>0,'900s CPU deadline';q['commands'].append(argv)
 z=subprocess.run(argv,capture_output=True,text=True,timeout=min(left,limit));(B/(label+'.log')).write_bytes((z.stdout+z.stderr).encode());assert z.returncode==0,(label,z.stderr[-2000:]);return z.stdout
def deps(text):return {str(pathlib.Path(p).resolve())for p in shlex.split(text.replace('\\\n',' ').split(':',1)[1])}
def prefix(argv):
 out=[];i=0
 while i<len(argv):
  x=argv[i]
  if x in ['-o','-MF','-MT','-MQ']:i+=2;continue
  if x in ['-c','-M','-MM','-MD','-MMD','-shared']or x.endswith(('.cpp','.o')):i+=1;continue
  out.append(x);i+=1
 assert pathlib.Path(out[0]).name in ['g++','c++','clang++']and not any(x.startswith(('-fplugin','-specs','-wrapper'))for x in out)
 return out
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 q['source_contract']=check();c=json.loads((D/'source-context-r216.json').read_bytes());old=json.loads((D/'source-context-r172.json').read_bytes())
 receipt=pathlib.Path(c['parent_receipt']);assert sha(receipt)==c['parent_receipt_SHA256'];p=json.loads(receipt.read_bytes())
 assert p['completed']and p['CPU_only']and p['all_old_guard_code_removed']and p['HOST_TUs_compiled']==3 and p['GPU_TUs_compiled']==0
 assert p['source_manifest_SHA256']==c['parent_source_manifest_SHA256']and sha(p['library'])==p['library_SHA256']==c['parent_library_SHA256']
 replacements={u['replaces']:u for u in p['compiled']};current=dict(p['reused_objects_SHA256']);current.update({u['object']:u['object_SHA256']for u in p['compiled']})
 assert len(current)==48 and len(p['reused_objects_SHA256'])==45
 for proof in c['actual_parent_compiled']:
  u=replacements[proof['replaces']]
  for key in ['source','source_SHA256','original_source','object','object_SHA256']:assert u[key]==proof[key]
  assert u['fresh_M_equal_MD']and u['original_dependency_graph_equal_after_private_mapping']
  for path,h in u['dependency_SHA256'].items():assert sha(path)==h,path
  for path,h in proof['relevant_dependency_SHA256'].items():assert u['dependency_SHA256'][path]==h
 assert all(sha(path)==h for path,h in current.items())
 assert {x for x in p['link']if x.endswith('.o')}==set(current),'Use actual232 link, never ac937 link'
 for path,proof in p['actual_DTK_bindings'].items():assert sha(path)==proof['SHA256']
 units=[]
 for u in old['units']:
  prior=replacements.get(u['object']);obj=prior['object']if prior else u['object']
  assert obj in current and sha(u['source'])==u['source_SHA256']
  if prior:assert prior['original_source']==u['source']and prior['source_SHA256']==u['source_SHA256']
  else:assert current[obj]==u['object_SHA256']
  units.append(dict(u,current_object=obj))
 expected={u['current_object']for u in units};assert len(expected)==9
 # Actual parent recorded every -M command. Reuse those commands for exclusion,
 # rather than invent flags or treating nm absence as sufficient ABI proof.
 scans={replacements[x['object']]['object']if x['object']in replacements else x['object']:x for x in p['all_object_scans']}
 exclusions=[]
 for i,obj in enumerate(current):
  symbols=run(['nm','-C',obj],'symbols-'+str(i),20)
  if obj in expected:continue
  assert 'TemporalCompressor'not in symbols,('extra layout-symbol consumer; no compile',obj)
  scan=scans[obj]
  actual=next((u for u in p['compiled']if u['object']==obj),None)
  if actual:
   source=actual['source'];argv=prefix(actual['compile_argv'])+['-M',source];assert sha(source)==actual['source_SHA256']
  elif scan.get('source'):
   source=scan['source'];assert sha(source)==scan['source_SHA256']
   recipes=[x for x in p['commands']if isinstance(x,list)and '-M'in x and source in x and '-c'not in x]
   assert len(recipes)==1,('Need exact recorded parent fresh-M command for reused HOST',obj,source,len(recipes));argv=recipes[0];prefix(argv)
  else:
   assert scan.get('opaque_gpu_helper')is True,('Unclassified reused object',obj)
   exclusions.append(dict(object=obj,opaque_GPU_helper=True,layout_symbols=False));continue
  ds=deps(run(argv,'reuse-M-'+str(i),60));assert not any(pathlib.Path(x).name=='deepseek_v4_temporal_compressor.hpp'for x in ds),('extra TemporalCompressor dependency; no compile',obj)
  if actual:assert ds==set(actual['dependency_SHA256']),('Retained guard object graph differs',obj)
  exclusions.append(dict(object=obj,source=source,source_SHA256=sha(source),fresh_dependencies_SHA256={x:sha(x)for x in ds},no_TemporalCompressor_header=True))
 q['reused_layout_exclusions']=exclusions
 # Extend the proven180d mirror with the9 TUs; only guard + the2 original cache
 # patch files change. Never use flat-I shadowing that reintroduces old guard.
 contexts=[];spelled=set()
 for i,u in enumerate(units):
  flags=prefix(u['compile_argv']);raw=run(flags+['-M',u['source']],'old-M-'+str(i),60);ds=deps(raw);spelled|=spelled_deps(raw);contexts.append((u,flags,ds))
 mirror=Mirror(R,B,spelled,D/'include/native_sparse_cache_r22.hpp',c['original_guard_SHA256'])
 changed=[]
 for private_name in ['deepseek_v4_temporal_compressor.hpp','deepseek_v4_compressor_fp32_c128_r19.cpp']:
  targets=[path for path,original in mirror.back.items()if pathlib.Path(original).name==private_name];assert targets,private_name
  expected_old=old['temporal_header_original_SHA256']if private_name.endswith('.hpp')else old['units'][2]['source_SHA256']
  for path in targets:
   original=mirror.back[path];assert mirror.original_sha[original]==expected_old,(private_name,original)
   pathlib.Path(path).write_bytes((D/'src'/private_name).read_bytes());mirror.pins[path]=sha(path);changed.append(dict(original=original,private=path,SHA256=sha(path)))
 q['mirror_changes']=changed;q['project_mirror']={k:v for k,v in mirror.summary.items()if k not in ['only_changed_basename','ordinary_project_bytes_unchanged']}
 q['project_mirror'].update(changed_basenames=['native_sparse_cache_r22.hpp','deepseek_v4_temporal_compressor.hpp','deepseek_v4_compressor_fp32_c128_r19.cpp'],ordinary_project_bytes_unchanged_excluding_named_patches=True)
 compiled=[]
 for i,(u,flags,original_deps)in enumerate(contexts):
  src=mirror.target(u['source']);f=mirror.rewrite_flags(flags);obj=B/(str(i)+'.o');md=B/(str(i)+'.d');ds=deps(run(f+['-M',str(src)],'new-M-'+str(i),60))
  assert mirror.normalize_and_check(ds)=={mirror.logical(x)for x in original_deps},('Dependency graph changed beyond approved patches',u['source'])
  for path in ds:
   if pathlib.Path(path).name=='native_sparse_cache_r22.hpp':assert sha(path)==c['current_guard_SHA256']
  if pathlib.Path(u['source']).name not in ['deepseek_v4_swa_attention.cpp','deepseek_v4_paged_latent_cache_public.cpp']:
   assert any(pathlib.Path(x).name=='deepseek_v4_temporal_compressor.hpp'and sha(x)==sha(D/'src/deepseek_v4_temporal_compressor.hpp')for x in ds)
  pins={x:sha(x)for x in ds};argv=f+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)];run(argv,'compile-'+str(i),180)
  assert deps(md.read_text())==ds and all(sha(x)==h for x,h in pins.items());mirror.normalize_and_check(ds)
  compiled.append(dict(source=str(src),source_SHA256=sha(src),original_source=u['source'],replaces=u['current_object'],object=str(obj),object_SHA256=sha(obj),compile_argv=argv,dependency_SHA256=pins,fresh_M_equal_MD=True,original_dependency_graph_equal_after_private_mapping=True))
 replace={u['replaces']:u['object']for u in compiled};link=[replace.get(x,x)for x in p['link']];lib=B/'libDeep_C128_weight_cache_r216.so';link[link.index('-o')+1]=str(lib);run(link,'link',120)
 data=lib.read_bytes();assert b'Native sparse cache requires rows1..128 and capacity1..65536'in data and b'Native sparse cache requires rows1..128 and capacity1..4096'not in data
 exports=lambda s:{x.split()[-1]:x.split()[-2]for x in s.splitlines()if len(x.split())>=3}
 before=exports(run(['nm','-D','--defined-only',p['library']],'parent-exports'));after=exports(run(['nm','-D','--defined-only',str(lib)],'new-exports'));missing=set(before)-set(after)
 allowed={'_ZNK3fmt3v1217basic_format_argsINS0_7contextEE3getEi','_ZZN3fmt3v126detail12format_floatIdEEiT_iRKNS0_12format_specsEbRNS1_6bufferIcEEENKUljPcE_clEjSA_'}
 assert missing<=allowed and all(before[x]=='W'for x in missing),missing
 assert {'modern_deep_c128_weight_cache_mode_r172','modern_deep_c128_weight_cache_stats_r172'}<=set(after)
 assert 'SYMBOLIC'in run(['readelf','-d',str(lib)],'dynamic');ld=run(['ldd','-r',str(lib)],'ldd-r',60);assert 'not found'not in ld and 'undefined symbol:'not in ld
 assert all(sha(path)==h for path,h in current.items())
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),HOST_TUs_compiled=9,compiled=compiled,link=link,reused_objects_SHA256={x:h for x,h in current.items()if x not in expected},actual_DTK_bindings=p['actual_DTK_bindings'],parent_receipt_SHA256=sha(receipt),parent_Native_SHA256=p['library_SHA256'],guard65536_retained=True,all_old_guard_code_removed=True,default_OFF=True,new_resident_bytes_per_owner=335544320,source_manifest_SHA256=sha(D/'manifest-r216.json'),scope='CPU-only HOST/source/layout/dependency/link closure. C1 numerical/GPU/Model/performance NOT_RUN.')
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
q.update(seconds=time.monotonic()-start,Torch_imported='torch'in sys.modules,Native_imported='_infinilm'in sys.modules);(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','HOST_TUs_compiled','library_SHA256','error','seconds']}));assert q['completed']
