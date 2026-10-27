"""Metadata-only recovery of CPU441's already linked DSO. Never compile/relink."""
import pathlib,json,hashlib,subprocess,shlex,time,os
R=pathlib.Path('/refactor');SRC=R/'incoming/Kimi-TP32-I96-MoE-r120';C=R/'candidates/Kimi-TP32-I96-MoE-r120c'
FAILED_SHA='798a1a00d5a3704999b9b04c81527047543fc27024472e27160b2ba25610c302'
MANIFEST_SHA='d8c1491ba823c70e6f07b3a503b1a3c2cee63e9be42325379dd9d604e002cc39'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def deps(p):return sorted(set(shlex.split(pathlib.Path(p).read_text().replace('\\\n',' ').split(':',1)[1])))
def fresh_command(actual,source,output):
 i=actual.index(str(source));prefix=actual[:i];out=[];j=0
 assert pathlib.Path(prefix[0]).name in ['g++','c++']
 while j<len(prefix):
  if prefix[j] in ['-MD','-MMD']:j+=1;continue
  if prefix[j]=='-MF':assert j+1<len(prefix);j+=2;continue
  out.append(prefix[j]);j+=1
 assert all(flag in out for flag in ['-O2','-fPIC','-shared','-pthread','-fno-omit-frame-pointer'])
 assert '-o' not in out and not any(v in out for v in ['-c','-S','-E','-M','-MM'])
 return [*out,'-M','-MF',str(output),str(source)]
def main():
 b=R/'candidates/Kimi-TP32-I96-MoE-r120d';b.mkdir(exist_ok=False)
 q={'completed':False,'CPU_only':True,'GPU':False,'GPU_initialized':False,'dlopen':False,'Runtime_create':False,'HOST_TUs_compiled':0,'reused_compiled_HOST_TUs':1,'GPU_TUs_compiled':0,'relink':False,'preprocessor_dependency_only':True,'commands':[],'producer':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')};start=time.monotonic()
 try:
  failed_path=C/'CPU-build.json';assert sha(failed_path)==FAILED_SHA
  old=json.loads(failed_path.read_bytes());assert old['completed'] is False and old['producer']=='910407.441' and old['error']=="AssertionError('actual/fresh dependency mismatch')"
  manifest=SRC/'source-manifest-r120.json';assert sha(manifest)==MANIFEST_SHA;m=json.loads(manifest.read_bytes())
  for name,h in m['package_source_SHA256'].items():assert sha(SRC/name)==h,name
  for path,h in m['pinned_libraries'].items():assert sha(path)==h,path
  source=SRC/'TP32_I96_MoE_r120.cpp';actual=old['commands'][0];assert actual.count(str(source))==1 and '-Wl,--no-undefined' in actual
  library=pathlib.Path(actual[actual.index('-o')+1]);assert library==C/'libKimi_TP32_I96_MoE_r120.so' and library.is_file()
  original_md=pathlib.Path(actual[actual.index('-MF')+1]);assert original_md==C/'adapter.d'
  stale_md=C/'adapter.fresh.d';all_deps=deps(original_md)
  watched={failed_path,manifest,source,SRC/'contract_r120.hpp',library,original_md,stale_md,pathlib.Path(m['native'])}
  watched.update(pathlib.Path(p) for p in all_deps)
  before={str(p):sha(p) for p in sorted(watched)}
  fresh=b/'adapter.complete-prefix.fresh.d';cmd=fresh_command(actual,source,fresh);q['commands'].append(cmd)
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(b/'preprocess.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-8000:]
  new_deps=deps(fresh);q['dependency_set_difference']={'only_original':sorted(set(all_deps)-set(new_deps)),'only_fresh':sorted(set(new_deps)-set(all_deps))}
  assert all_deps==new_deps,'actual/fresh exact dependency equality required'
  for suffix,expected in m['header_source_suffix_SHA256'].items():
   hits=[p for p in all_deps if p.endswith(suffix)];assert hits and all(sha(p)==expected for p in hits),(suffix,hits)
  metadata={}
  for name,cmd in [('defined_symbols',['nm','-D','--defined-only',str(library)]),('dynamic_section',['readelf','-d',str(library)]),('ldd',['ldd',str(library)])]:
   q['commands'].append(cmd);p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);(b/(name+'.txt')).write_text(p.stdout+p.stderr);assert p.returncode==0,(name,p.stderr[-2000:]);metadata[name]=p.stdout
  for symbol in ['modern_kimi_TP32_moe_r120','modern_kimi_TP32_moe_result_r120']:assert symbol in metadata['defined_symbols']
  assert 'not found' not in metadata['ldd']
  assert '/usr/local/lib/python3.10/dist-packages/torch/lib' in os.environ.get('LD_LIBRARY_PATH',''),'fixed Torch library search path required'
  after={str(p):sha(p) for p in sorted(watched)};assert before==after,'compiledDSO/MD/source/Native/dependencies changed during metadata verification'
  assert sha(failed_path)==FAILED_SHA and json.loads(failed_path.read_bytes())['completed'] is False
  q.update(completed=True,library=str(library),library_SHA256=sha(library),native=m['native'],native_SHA256=m['native_SHA256'],source_manifest_SHA256=MANIFEST_SHA,original_failed_receipt=str(failed_path),original_failed_receipt_SHA256=FAILED_SHA,original_MD_SHA256=sha(original_md),preserved_old_fresh_MD_SHA256=sha(stale_md),fresh_complete_prefix_MD_SHA256=sha(fresh),actual_fresh_MD_set_equal=True,dependency_SHA256={p:before[p] for p in all_deps},watched_before_SHA256=before,watched_after_SHA256=after,ldd=metadata['ldd'],normal_CPU_exit=True,elapsed_s=time.monotonic()-start,repair='KeepcompleteactualprefixbeforeTU,removeonlyMD/MFpair; noCXXrecompile/noDSOmutation/noMathChange')
 except BaseException as e:q['error']=repr(e)
 (b/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k) for k in ['completed','error','library','library_SHA256','actual_fresh_MD_set_equal','reused_compiled_HOST_TUs']}));assert q['completed']
if __name__=='__main__':main()
