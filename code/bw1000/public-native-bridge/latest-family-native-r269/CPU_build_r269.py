"""Carry upstream activation at the three actual worker placements; CPU only."""
import pathlib,json,hashlib,subprocess,shlex,os,time,signal,traceback,argparse
D=pathlib.Path('/refactor/incoming/latest-family-native-r269');R=pathlib.Path('/refactor');B=R/'candidates/latest-family-native-r269'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def words(s):return set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);a=ap.parse_args();mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==a.manifest_sha
 manifest=json.loads(mf.read_bytes())
 for name,h in manifest['files_SHA256'].items():assert sha(D/name)==h,name
 m=json.loads((D/'build-contexts.json').read_bytes());assert all(os.environ.get(v)==''for v in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 B.mkdir(exist_ok=False);start=time.monotonic();deadline=start+205
 q={'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU':False,'Native_dlopen':False,'weights_or_Model_loaded':False,'upstream_commit':m['upstream_commit'],'source_manifest_SHA256':a.manifest_sha,'completed':False,'passed':False,'new_HOST_TUs':0,'new_GPU_TUs':0,'families':{},'logs':{}}
 def run(cmd,label,cwd,cap):
  assert pathlib.Path(cmd[0]).name in ['g++','nm','readelf'],'Only reviewed HOST compiler/symbol commands permitted'
  assert deadline-time.monotonic()>0,'CPU absolute deadline exhausted'
  proc=subprocess.Popen(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
  try:out=proc.communicate(timeout=min(cap,deadline-time.monotonic()))[0]
  except BaseException:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:out=proc.communicate(timeout=3)[0]
   except subprocess.TimeoutExpired:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    out=proc.communicate(timeout=3)[0]
   (B/(label+'.log')).write_text(out);raise RuntimeError('CPU_BOUNDED_COMMAND_TIMEOUT:'+label)
  p=B/(label+'.log');p.write_text(out);q['logs'][label]={'returncode':proc.returncode,'bytes':p.stat().st_size,'SHA256':sha(p)}
  assert proc.returncode==0,'CPU_COMMAND_FAILED:'+label
  return out
 def fresh_md(flags,source,dest,cwd,label):
  run([*flags,'-M','-MF',str(dest),'-MT','worker.o',str(source)],label,cwd,30)
  return {(pathlib.Path(v)if pathlib.Path(v).is_absolute()else cwd/v).resolve()for v in words(dest.read_text())}
 def mirror_worker(original,where,expected):
  assert sha(original)==expected
  old=original.read_text();needle='            runtime_ = infinilm::runtime::Runtime::create(rank_info_.device);'
  assert old.count(needle)==1 and 'runtime_->activate();'not in old
  new=old.replace(needle,needle+chr(10)+'            runtime_->activate();')
  srcroot=original.parent.parent;where.mkdir(parents=True)
  for p in srcroot.iterdir():
   if p.name!='engine':(where/p.name).symlink_to(p,target_is_directory=p.is_dir())
  engine=where/'engine';engine.mkdir()
  for p in original.parent.iterdir():
   if p.name!=original.name:(engine/p.name).symlink_to(p,target_is_directory=p.is_dir())
  source=engine/original.name;source.write_text(new)
  assert source.read_text().replace('            runtime_->activate();'+chr(10),'',1)==old
  return source
 def compile_worker(flags,original,source,target,cwd,label):
  assert pathlib.Path(flags[0]).name=='g++'and '-fPIC'in flags and '-std=c++17'in flags
  parent=fresh_md(flags,original,target.with_suffix('.parent.d'),cwd,label+'-parent-M')
  actual=fresh_md(flags,source,target.with_suffix('.fresh.d'),cwd,label+'-new-M')
  assert {original.resolve()if v==source.resolve()else v for v in actual}==parent,'effective dependencies differ except patched source'
  for p in parent:assert p.is_file(),p
  before={str(p):sha(p)for p in parent}
  run([*flags,'-MD','-MF',str(target.with_suffix('.d')),'-c',str(source),'-o',str(target)],label+'-compile',cwd,100);q['new_HOST_TUs']+=1
  compiled={(pathlib.Path(v)if pathlib.Path(v).is_absolute()else cwd/v).resolve()for v in words(target.with_suffix('.d').read_text())}
  assert compiled==actual,'compiled and fresh dependency sets differ'
  assert all(sha(p)==h for p,h in before.items()),'dependency changed while compiling'
  return {'original_source':str(original),'original_source_SHA256':sha(original),'new_source':str(source),'new_source_SHA256':sha(source),'new_object':str(target),'new_object_SHA256':sha(target),'dependency_count':len(parent),'all_project_and_system_dependencies_equal_except_worker_source':True,'dependency_SHA256':before,'compile_argv':flags}
 def symbols(binary,label,cwd):
  text=run(['nm','-D','--defined-only',str(binary)],label,cwd,15)
  return {s.split()[-1]for s in text.splitlines()if len(s.split())>=3}
 try:
  for fam in ['GLM','Deep']:
   q['stage']=fam+'-Common';common=R/('candidates/rebased-5f91ad7-r3668/Common/'+fam+'/InfiniLM');ctx=m['contexts'][fam];where=B/fam;where.mkdir();row={};q['families'][fam]=row
   old=common/'python/_infinilm.cpython-310-x86_64-linux-gnu.so';assert sha(old)==m['old_Common_extension_SHA256'][fam]
   dep=common/'build/.deps/_infinilm/linux/x86_64/release/csrc/engine/rank_worker.cpp.o.d';linkdep=common/'build/.deps/_infinilm/linux/x86_64/release/_infinilm.cpython-310-x86_64-linux-gnu.so.d'
   assert sha(dep)==ctx['worker_dependency_metadata_SHA256']and sha(linkdep)==ctx['link_metadata_SHA256']
   original=common/'csrc/engine/rank_worker.cpp';source=mirror_worker(original,where/'Common-source',m['old_Common_worker_source_SHA256']);newobj=where/'Common-worker.o';row['Common_worker']=compile_worker(ctx['compile_argv'],original,source,newobj,common,fam+'-Common')
   objects=[common/v for v in ctx['objects']];oldobj=common/'build/.objs/_infinilm/linux/x86_64/release/csrc/engine/rank_worker.cpp.o';assert objects.count(oldobj)==1
   reused={str(p):sha(p)for p in objects if p!=oldobj};objects[objects.index(oldobj)]=newobj
   extdir=where/'python';extdir.mkdir();newext=extdir/old.name;cmd=[ctx['link_argv'][0],*map(str,objects),'-o',str(newext),*ctx['link_argv'][1:]];run(cmd,fam+'-extension-link',common,35)
   assert all(sha(p)==h for p,h in reused.items());oldsyms=symbols(old,fam+'-old-extension-symbols',common);newsyms=symbols(newext,fam+'-new-extension-symbols',common);assert oldsyms<=newsyms,'extension exports removed'
   runtime=common/'python/infinilm/_lib/libinfinilm_runtime.so';assert 'infinilm::runtime::Runtime::activate()'in run(['nm','-D','-C','--defined-only',str(runtime)],fam+'-runtime-activate',common,15)
   row['Common_extension']={'original':str(old),'original_SHA256':sha(old),'new':str(newext),'new_SHA256':sha(newext),'reused_objects_SHA256':reused,'retained_public_symbols':len(oldsyms),'new_defined_symbol_count':len(newsyms),'runtime_library_SHA256':sha(runtime),'link_context_metadata_SHA256':sha(linkdep),'link_argv':cmd,'Native_loaded':False}
  q['stage']='Deep-Native';p=R/'candidates/Deep-native-context-guard-r180d/CPU-build.json';assert sha(p)==m['Deep_CPU_receipt_SHA256'];receipt=json.loads(p.read_bytes());old=pathlib.Path(receipt['library']);assert sha(old)==receipt['library_SHA256']=='232fb7e2566f7d1961fa433d9e45660bfe760ec726676dcb7f1f406d857c801f'
  cmd=m['Deep_native_original_worker_fresh_command'];assert cmd.count('-M')==1 and cmd[-1].endswith('Model/Deep/InfiniLM/csrc/engine/rank_worker.cpp');flags=[v for v in cmd[:-1]if v!='-M'];original=pathlib.Path(cmd[-1]);source=mirror_worker(original,B/'Deep/Native-source',m['old_Deep_native_worker_source_SHA256']);target=B/'Deep/Native-worker.o';row=q['families']['Deep'];row['Native_worker']=compile_worker(flags,original,source,target,R,'Deep-Native')
  link=receipt['link'][:];assert pathlib.Path(link[0]).name=='g++'and '-shared'in link
  oldobj='/refactor/candidates/deep-c4-projection-model-r5920/objects/unit-8.o';assert link.count(oldobj)==1
  expected=dict(receipt['reused_objects_SHA256'])
  for item in receipt['compiled']:expected[item['object']]=item['object_SHA256']
  assert {v for v in link if v.endswith('.o')}==set(expected)
  assert all(sha(p)==h for p,h in expected.items());link[link.index(oldobj)]=str(target);new=B/'Deep/libDeep_latest_native_r269.so';link[link.index('-o')+1]=str(new);run(link,'Deep-Native-link',R,35)
  oldsyms=symbols(old,'Deep-old-Native-symbols',R);newsyms=symbols(new,'Deep-new-Native-symbols',R);assert oldsyms<=newsyms,'Native exports removed'
  row['Native']={'original':str(old),'original_SHA256':sha(old),'new':str(new),'new_SHA256':sha(new),'unchanged_objects_SHA256':{p:h for p,h in expected.items()if p!=oldobj},'all_unchanged_object_bytes_verified':True,'retained_public_symbols':len(oldsyms),'new_defined_symbol_count':len(newsyms),'link_argv':link}
  q.update(passed=True,stage='complete',scope='Exactly three HOST worker activation patches, two extension relinks and one Native relink; all other compiled objects retained. No dlopen, weights, GPU or Model qualification.',GPU_GO=False)
 except BaseException as exc:q.update(error_type=type(exc).__name__,error=str(exc)[:1500],failure_locations=[{'file':t.filename,'line':t.lineno,'function':t.name}for t in traceback.extract_tb(exc.__traceback__)])
 finally:
  q['completed']=True;q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['actual_step','completed','passed','stage','new_HOST_TUs','elapsed_s','error_type','error']}))
 assert q['passed']
if __name__=='__main__':main()
