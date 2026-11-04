"""0compile: strict explicit-alias metadata first; link only after all checks.

The successful CPU985 fixed-sidecar contract and both production objects are
reused. Original failure985 and metadata-only TIMEOUT986 remain failures.
"""
from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess,time,traceback
D=Path(__file__).resolve().parent;P=D.parent
B=Path('/refactor/candidates/Kimi-vocab-parallel-embedding-r461-probe-link-CPU-v2')
O=Path('/refactor/candidates/Kimi-vocab-parallel-embedding-r461-probe-CPU-v1')
def main():
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
 assert not B.exists();B.mkdir();start=time.monotonic();cache={};hash_records={}
 q=dict(completed=False,passed=False,CPU_only=True,compile=False,new_HOST_TUs=0,new_GPU_TUs=0,Native_dlopen=False,SDK_execution=False,GPU=0,Model=0,
  actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[],logs={},metadata_completed=False,link_started=False)
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def progress(phase,**values):
  q['phase']=phase
  with(B/'progress.jsonl').open('a')as f:f.write(json.dumps(dict(phase=phase,elapsed_s=time.monotonic()-start,**values))+'\n');f.flush();os.fsync(f.fileno())
  save();assert time.monotonic()-start<150
 def H(path):
  p=Path(path);before=p.stat();key=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
  if key not in cache:
   h=hashlib.sha256(p.read_bytes()).hexdigest();after=p.stat()
   assert (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==key,('changed while hashing',str(p));cache[key]=h
  h=cache[key];hash_records[str(p)]=dict(SHA256=h,bytes=before.st_size,device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns)
  return h
 def run(args,label,cap=30):
  progress(label);q['commands'].append(args);save()
  p=subprocess.run(args,capture_output=True,text=True,timeout=min(cap,150-(time.monotonic()-start)))
  log=B/(label+'.log');log.write_text(p.stdout+p.stderr);q['logs'][label]=dict(rc=p.returncode,SHA256=H(log));save()
  if p.returncode:raise RuntimeError(label+' failed; original '+str(log))
  return p.stdout
 def deps(path):return set(shlex.split(Path(path).read_text().replace('\\\n',' ').split(':',1)[1]))
 try:
  q['source_manifest_SHA256']=H(D/'CPU_SOURCE_MANIFEST.json');progress('source-and-parent-pins')
  for root in [D,P]:
   m=json.loads((root/'CPU_SOURCE_MANIFEST.json').read_bytes())
   for name,h in m['files_SHA256'].items():assert H(root/name)==h,(root,name)
  pins=json.loads((D/'PARENT.json').read_bytes());aliases=json.loads((D/'EXACT_ALIASES.json').read_bytes())
  assert H(O/'CPU-result.json')==pins['CPU985_receipt']['SHA256'];old=json.loads((O/'CPU-result.json').read_bytes())
  assert old['actual_CPU']=='910407.985'and not old['passed']and old['phase']=='worker-compile'and old['source_contract']['passed']
  assert all(old['logs'][label]['rc']==0 and H(O/(label+'.log'))==old['logs'][label]['SHA256']for label in ['pair_probe-compile','worker-compile','actual-fixed-sidecar-contract'])
  assert json.loads((O/'actual-fixed-sidecar-contract.log').read_bytes())['cases']==6
  base_path=Path('/refactor/candidates/Kimi-vocab-parallel-embedding-r461-CPU-v2/CPU-result.json')
  parent_path=Path('/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v2/CPU-result.json')
  assert H(base_path)=='052818841ad38d97114e8a5e16634f67f67b743e43070fcccfbfeab20233327e'
  assert H(parent_path)=='cfe2237631aab314d8f410aceef029de486361edbc76c80ab421bb7655bb0185'
  base=json.loads(base_path.read_bytes());parent=json.loads(parent_path.read_bytes())
  alias_proof={};q['explicit_alias_proof']=alias_proof;progress('26-explicit-alias-pairs')
  for index,row in enumerate(aliases):
   left,right=Path(row['old_path']),Path(row['actual_path']);expected=row['SHA256']
   assert left.samefile(right)and H(left)==H(right)==expected,(str(left),str(right))
   canonical_left,canonical_right=str(left.resolve()),str(right.resolve())
   assert canonical_left==canonical_right
   alias_proof[row['old_path']]=dict(actual_path=row['actual_path'],SHA256=expected,samefile=True,
    canonical=canonical_left,old_stat=hash_records[str(left)],actual_stat=hash_records[str(right)])
   if index%4==3:progress('alias-pairs-progress',checked=index+1)
  assert len(alias_proof)==26
  q['explicit_alias_proof']=alias_proof;progress('full-original-MD-and-object-identity')
  units={};source_origins={}
  for unit,receipt,oldmd in [('pair_probe',base,base['compiled']['pair_probe']['all_new_MD']),('worker',parent,parent['HOST']['worker']['all_new_MD'])]:
   argv=next(c for c in receipt['commands']if '-c'in c and Path(c[c.index('-c')+1]).name==unit+'.cpp')
   origin=argv[argv.index('-c')+1];source_origins[unit]=origin
   actual=deps(O/(unit+'.d'));assert actual==deps(O/(unit+'.before.d'))
   for suffix in ['.d','.before.d']:assert H(O/(unit+suffix))==pins['original_MD'][unit+suffix]['SHA256']
   actual_h={}
   for index,p in enumerate(sorted(actual)):
    actual_h[p]=H(p)
    if index%64==63:progress('full-MD-hash-progress',unit=unit,checked=index+1,last_path=p)
   mapped=set()
   for p,h in oldmd.items():
    if p==origin:continue
    target=p if p in actual else alias_proof[p]['actual_path']
    assert target in actual and actual_h[target]==h,(unit,p,target);mapped.add(target)
   # Exact additional-dependency list, not a generic basename/../ exception.
   extra=set(actual)-mapped
   assert extra==set(pins['exact_additional_MD'][unit]),(unit,sorted(extra))
   for p,h in pins['exact_additional_MD'][unit].items():assert actual_h[p]==h
   obj=O/(unit+'.o');assert H(obj)==pins['objects'][unit]['SHA256']and obj.stat().st_size==pins['objects'][unit]['bytes']
   compiled_argv=next(c for c in old['commands']if '-c'in c and Path(c[c.index('-c')+1]).name==unit+'.cpp')
   assert compiled_argv[compiled_argv.index('-c')+1]==str(P/'native'/ (unit+'.cpp'))
   assert compiled_argv[compiled_argv.index('-o')+1]==str(obj)
   units[unit]=dict(object=pins['objects'][unit],actual_compile_argv=compiled_argv,
    source=dict(path=str(P/'native'/(unit+'.cpp')),SHA256=H(P/'native'/(unit+'.cpp')),rawtext=(P/'native'/(unit+'.cpp')).read_text()),
    actual_MD_paths_SHA256=actual_h,MD_before_after_same=True,
    original_MD={s:dict(path=str(O/(unit+s)),SHA256=H(O/(unit+s)),rawtext=(O/(unit+s)).read_text())for s in ['.d','.before.d']})
   progress('full-unit-MD-identity-passed',unit=unit,paths=len(actual))
  for p,h in base['all_linked_object_SHA256'].items():assert H(p)==h,p
  for v in base['actual_dependency_library_SHA256'].values():assert H(v['path'])==v['SHA256']
  compiler=Path(shutil.which('g++'));q['compiler_current_container']=dict(path=str(compiler),canonical=str(compiler.resolve()),SHA256=H(compiler))
  q['compiler_scope']='Current same-image g++ binary; CPU985 records its exact argv/PATH envelope, not a contemporaneous binary-file hash.'
  q['original985_metadata']=units;q['metadata_completed']=True;q['all_metadata_path_records']=hash_records.copy()
  (B/'metadata-before-link.json').write_text(json.dumps(q,indent=2)+'\n');progress('all-identity-PASS-before-link')
  link=next(c[:]for c in base['commands']if '-shared'in c)
  replacements={str(base_path.parent/'pair_probe.o'):str(O/'pair_probe.o'),str(parent_path.parent/'worker.o'):str(O/'worker.o')}
  assert set(replacements)<=set(link);link=[replacements.get(v,v)for v in link]
  native=B/'libKimi_vocab_parallel_embedding_probe_r461.so';link[link.index('-o')+1]=str(native)
  assert '-c'not in link and not any(v.endswith(('.cpp','.cu','.cc'))for v in link)
  q['link_started']=True;run(link,'private-Native-link',35)
  objects=[p for p in link if p.endswith('.o')];assert len(objects)==34
  def strong(path,label):
   rows=run(['nm','-D','--defined-only',str(path)],label,15)
   return {w[-1]for line in rows.splitlines()if len(w:=line.split())>=3 and w[-2]in ['T','D','B','R']}
  old_symbols=strong(base['native'],'CPU981-strong-symbols');new=strong(native,'new-strong-symbols');assert old_symbols<=new
  required={'modern_kimi_embedding_probe_load_r461','modern_kimi_embedding_probe_r461'};assert required<=new
  assert 'SYMBOLIC'in run(['readelf','--wide','-d',str(native)],'Native-dynamic',10)
  ldd=run(['ldd','-r',str(native)],'Native-ldd-relocations-only',20);assert 'not found'not in ldd and 'undefined symbol'not in ldd
  q.update(completed=True,passed=True,native=str(native),native_SHA256=H(native),all_linked_object_SHA256={p:H(p)for p in objects},
   old_strong_exports_retained=len(old_symbols),old_CAPI_retained=len({n for n in old_symbols if n.startswith('modern_')}),new_CAPI=sorted(required),
   actual_dependency_library_SHA256=base['actual_dependency_library_SHA256'],original985FAIL_preserved=True,metadata986TIMEOUT_preserved=True,
   successful_CPU985_contract_reused_not_rerun=True,original985_objects_reused_not_recompiled=True)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  q.update(elapsed_s=time.monotonic()-start,unique_files_hashed=len(cache));save()
  print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','phase','error','elapsed_s','native_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
