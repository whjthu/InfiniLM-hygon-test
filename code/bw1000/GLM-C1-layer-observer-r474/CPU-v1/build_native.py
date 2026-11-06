"""Finite GLM474 CPU-only exact5HOST/27reuse Native build. No dlopen/SDK calls.
All5 fresh-M and new class-header closure precede any compilation. Every
mirrored file has a registered source/hash; no directory whitelist or old-MD
assumption. Failed compiled artifacts are retained, never silently replayed.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,os,posixpath,re,shlex,shutil,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):
 text=Path(p).read_text().replace('\\\r\n',' ').replace('\\\n',' ');result=set()
 for line in text.splitlines():
  if line.strip():left,colon,right=line.partition(':');assert colon;result.update(shlex.split(right))
 assert result and all(p.startswith('/')for p in result);return result

def strong(text):return {a[-1]:a[-2]for l in text.splitlines()if len(a:=l.split())>=3 and a[-2]in ['T','D','B','R']}
def needed(text):return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',text)
def resolved(text):
 assert 'not found'not in text and 'undefined symbol'not in text,'loader unresolved'
 return sorted(set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M)))

def main():
 start=time.monotonic();plan=json.loads((D/'BUILD_PLAN.json').read_bytes());B=Path(plan['candidate']);assert not B.exists(),'Existing candidate: reconcile original, never replay';B.mkdir(mode=0o700);(B/'objects').mkdir();M=B/'project-mirror'
 q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,Native_dlopen=False,Runtime_created=False,SDK_explicit_calls=0,GPU_job=False,Torch_imported=False,weights_opened=False,Model_qualified=False,new_HOST=0,new_GPU=0,commands=[],compiled={},libraries={},failures=[],fresh_dependencies_SHA256={},unmatched_dependencies=[],observer_consumers_verified=[],original_reuse_objects=27,ldd_child_initialization='UNKNOWN; metadata-only children under documented parent ENV recipe, no explicit Runtime/SDK calls')
 def save():
  p=B/'CPU-result.tmp';p.write_text(json.dumps(q,indent=2)+'\n');os.replace(p,B/'CPU-result.json')
 def remaining():return plan['work_seconds']-(time.monotonic()-start)
 def run(args,label,timeout=300):
  left=remaining();assert left>2,'Original560s work deadline';z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=min(timeout,left-1));logs={}
  for k,s in [('stdout',z.stdout),('stderr',z.stderr)]:
   p=B/(label+'.'+k);p.write_text(s);logs[k]=dict(path=str(p),bytes=p.stat().st_size,SHA256=H(p))
  return dict(argv=args,rc=z.returncode,logs=logs),z.stdout,z.stderr
 def checked(args,label,timeout=300):
  record,out,err=run(args,label,timeout);q['commands'].append(record);save();assert record['rc']==0,(label,record['rc'],err[-1400:]);return out,err
 try:
  claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and claim['Native_compile_GO']and not claim['GPU_GO']and H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['TUNING_CPU_CLAIM_SHA'];assert claim['parent']==os.environ['SLURM_JOB_ID']and claim['new_HOST_TUs']==5 and claim['new_GPU_translation_units']==0 and claim['work_seconds']==plan['work_seconds']==560
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==v for n,v in mf['files_SHA256'].items());q.update(source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'),source_parent_MF=plan['source_v2_manifest_SHA256'])
  assert H(D/'native-src/SOURCE_MANIFEST.json')==plan['source_v2_manifest_SHA256'];sm=json.loads((D/'native-src/SOURCE_MANIFEST.json').read_bytes());assert all(H(D/'native-src'/n)==h for n,h in sm['files_SHA256'].items())
  tools=json.loads((D/'TOOLCHAIN.json').read_bytes());seen={}
  for path,pin in tools['tools'].items():
   canonical=str(Path(path).resolve());assert canonical==pin['canonical'],('tool canonical',path)
   if canonical not in seen:seen[canonical]=H(path)
   assert seen[canonical]==pin['sha256'],('tool bytes',path)
  for k,v in tools['required_environment'].items():assert os.environ.get(k)==v,(k,'environment drift')
  assert H(plan['parent_receipt'])==plan['parent_receipt_SHA256'];parent=json.loads(Path(plan['parent_receipt']).read_bytes());assert parent['completed']and parent['passed']and parent['actual_CPU']=='910407.1101'and parent['all_fresh_M_equal_MD'];assert H(plan['parent_library'])==plan['parent_library_SHA256']==parent['libraries']['GLM471']['SHA256']
  for path,h in plan['reuse_objects_SHA256'].items():assert H(path)==h,('original27 object bytes',path)
  for path,pin in plan['parent_DSO_bindings'].items():assert H(path)==pin['SHA256'],('original DSO bytes',path)
  # Resolve the old complete binary before spending on compilation, under the
  # exact documented parent ENV recipe. This is not an SDK initialization proof.
  old_out,old_err=checked(['/usr/bin/ldd','-r',plan['parent_library']],'parent-ldd',30);old_dynamic={p:H(p)for p in resolved(old_out+'\n'+old_err)};q['parent_transitive_SHA256']=old_dynamic;assert old_dynamic==plan['original_transitive_SHA256']
  snapshot=json.loads((D/'FRESH_DEPENDENCY_SNAPSHOT.json').read_bytes());assert H(D/'FRESH_DEPENDENCY_SNAPSHOT.json')==plan['snapshot_SHA256'];oldpins=snapshot['path_SHA256'];allowed=dict(oldpins);mirror_pins={}
  for rel,row in sorted(plan['mirror_files'].items()):
   assert rel and not rel.startswith('/')and '..'not in Path(rel).parts
   dest=M/rel;assert str(dest).startswith(str(M)+'/');dest.parent.mkdir(parents=True,exist_ok=True)
   if row['original_paths']:
    raw=row['original_paths'][0];assert H(raw)==row['original_SHA256'];data=Path(raw).read_bytes()
    anchor=os.stat(raw)
    for alias in row['original_paths']:
     astat=os.stat(alias);assert oldpins[alias]==row['original_SHA256']and posixpath.normpath(alias)==('/refactor/'+rel);assert(astat.st_dev,astat.st_ino)==(anchor.st_dev,anchor.st_ino),('explicit mirror alias differs',alias)
   else:data=None
   if row.get('replacement'):data=(D/row['replacement']).read_bytes()
   assert data is not None and hashlib.sha256(data).hexdigest()==row['effective_SHA256']
   with dest.open('xb')as f:f.write(data)
   mirror_pins[str(dest)]=row['effective_SHA256'];allowed[str(dest)]=row['effective_SHA256']
  q['mirror_files']=len(mirror_pins);q['mirror_file_SHA256']=mirror_pins
  # Register every mirrored raw alias individually, plus normalized generated
  # files. New paths may bind only to an exact registered file+SHA, never a dir.
  for raw,h in oldpins.items():
   if raw.startswith('/refactor/'):
    candidate=str(M)+'/'+raw[len('/refactor/'):];norm=posixpath.normpath(candidate)
    if norm in mirror_pins:allowed[candidate]=mirror_pins[norm]
  normalized_original={}
  for raw,h in oldpins.items():normalized_original.setdefault(posixpath.normpath(raw),[]).append((raw,h))
  observed={};registered_aliases=[]
  def pin_for(path):
   if path in allowed:return allowed[path]
   norm=posixpath.normpath(path)
   if norm in mirror_pins:
    assert path.startswith(str(M)+'/');registered_aliases.append(dict(path=path,registered_file=norm,SHA256=mirror_pins[norm],kind='generated plain-file mirror alias'));return mirror_pins[norm]
   stat=os.stat(path)
   for old,h in normalized_original.get(norm,[]):
    other=os.stat(old)
    if(stat.st_dev,stat.st_ino)==(other.st_dev,other.st_ino):registered_aliases.append(dict(path=path,registered_file=old,SHA256=h,kind='original exact same dev/inode'));return h
   return None
  def fresh(unit):
   source=unit['source'];assert H(source)==unit['source_SHA256'];assert H(unit['original_source'])==unit['original_source_SHA256'];compiler=shutil.which(unit['prefix'][0]);assert compiler==tools['compiler_selection'][unit['prefix'][0]]
   prefix=unit['prefix'];assert '-c'not in prefix and '-o'not in prefix and '-MD'not in prefix
   md=B/(unit['name']+'.before.d');obj=B/'objects'/(unit['name']+'.o');record,out,err=run(prefix+['-M','-MF',str(md),'-MT',str(obj),source],unit['name']+'-M',120);return unit,record,md,obj,err
  with concurrent.futures.ThreadPoolExecutor(max_workers=plan['max_parallel'])as pool:
   futures=[pool.submit(fresh,u)for u in plan['units']]
   for future in concurrent.futures.as_completed(futures):
    unit,record,md,obj,err=future.result();q['commands'].append(record)
    if record['rc']!=0:q['failures'].append(dict(name=unit['name'],phase='M',rc=record['rc'],tail=err[-1400:]));save();continue
    names=deps(md);sample={}
    for path in names:
     if path not in observed:observed[path]=H(path)
     value=observed[path];expected=pin_for(path);sample[path]=value
     if value!=expected:q['unmatched_dependencies'].append(dict(TU=unit['name'],path=path,SHA256=value,expected=expected))
    assert unit['source']in sample and sample[unit['source']]==unit['source_SHA256']
    actual_project={posixpath.normpath(p)for p in names if p.startswith(str(M)+'/')}
    expected_project={posixpath.normpath(p)for p in unit['expected_project_MD_paths']}
    if actual_project!=expected_project or any(p.startswith('/refactor/')and not p.startswith(str(M)+'/')for p in names):q['failures'].append(dict(name=unit['name'],phase='M-projectgraph',missing=sorted(expected_project-actual_project),extra=sorted(actual_project-expected_project),escaped=[p for p in names if p.startswith('/refactor/')and not p.startswith(str(M)+'/')]))
    if unit['changes_class_definition']:

     if plan['new_class_header_SHA256']in sample.values()and plan['old_class_header_SHA256']not in sample.values():q['observer_consumers_verified'].append(unit['name'])
     else:q['failures'].append(dict(name=unit['name'],phase='M-observer',error='old/missing candidate observer definition'))
    elif plan['old_class_header_SHA256']in sample.values()or plan['new_class_header_SHA256']in sample.values():q['failures'].append(dict(name=unit['name'],phase='M-observer',error='unexpected observer consumer'))
    q['fresh_dependencies_SHA256'][unit['name']]=sample;save()
  q['registered_explicit_aliases']=registered_aliases;save();assert len(q['fresh_dependencies_SHA256'])==5 and len(q['observer_consumers_verified'])==5 and not q['failures']and not q['unmatched_dependencies'],('All5 fresh-M must close before first compile',len(q['unmatched_dependencies']));q['all5_fresh_M_before_compile']=True;save()
  def compile_one(unit):
   obj=B/'objects'/(unit['name']+'.o');md=B/(unit['name']+'.actual.d');record=dict(argv=unit['prefix']+['-MD','-MF',str(md),'-c',unit['source'],'-o',str(obj)],rc=-1)
   try:
    record,out,err=run(record['argv'],unit['name']+'-compile',300)
    if record['rc']!=0:return unit,record,None,err
    expected=q['fresh_dependencies_SHA256'][unit['name']];assert deps(md)==set(expected),('MD differs from freshM',unit['name']);assert all(H(p)==v for p,v in expected.items()),('header bytes changed while compiling',unit['name'])
    return unit,record,dict(object=str(obj),object_SHA256=H(obj),MD=str(md),MD_SHA256=H(md),source=unit['source'],source_SHA256=H(unit['source']),kind=unit['kind'],original_object=unit['original_object']),err
   except BaseException as e:
    record['exception']=repr(e)
    if obj.exists():record['unqualified_partial_object']=dict(path=str(obj),SHA256=H(obj))
    if md.exists():record['unqualified_partial_MD']=dict(path=str(md),SHA256=H(md))
    return unit,record,None,repr(e)
  with concurrent.futures.ThreadPoolExecutor(max_workers=plan['max_parallel'])as pool:
   futures=[pool.submit(compile_one,u)for u in plan['units']]
   for future in concurrent.futures.as_completed(futures):
    unit,record,built,err=future.result();q['commands'].append(record)
    if built:q['compiled'][unit['name']]=built;q['new_'+unit['kind']]+=1
    else:q['failures'].append(dict(name=unit['name'],phase='compile',rc=record['rc'],tail=err[-1400:]))
    save()
  assert len(q['compiled'])==5 and q['new_HOST']==5 and q['new_GPU']==0 and not q['failures'];q['all5_MD_equal_fresh_M']=True
  replaced={u['original_object']:q['compiled'][u['name']]['object']for u in plan['units']};args=[replaced.get(v,v)for v in plan['link_argv']];library=B/'libGLM_C1_layer_observer_r474.so';args[args.index('-o')+1]=str(library);objects=[p for p in args if p.endswith('.o')];assert len(objects)==len(set(objects))==32 and set(objects)==set(plan['reuse_objects_SHA256'])|set(replaced.values());assert max(args.index(p)for p in objects)<min(i for i,v in enumerate(args)if v.startswith('-l'))
  checked(args,'link',120);new_syms=strong(checked(['/usr/bin/nm','-D','--defined-only',str(library)],'symbols',25)[0]);old_syms=strong(checked(['/usr/bin/nm','-D','--defined-only',plan['parent_library']],'parent-symbols',25)[0]);assert all(new_syms.get(n)==t for n,t in old_syms.items()),'Parent strong symbol lost';assert all(new_syms.get(n)=='T'for n in plan['required_new_exports'])
  elf=checked(['/usr/bin/readelf','-d',str(library)],'ELF',25)[0];old_elf=checked(['/usr/bin/readelf','-d',plan['parent_library']],'parent-ELF',25)[0];assert needed(elf)==needed(old_elf)==plan['old_NEEDED']and'SYMBOLIC'in elf
  out,err=checked(['/usr/bin/ldd','-r',str(library)],'ldd',30);dynamic={p:H(p)for p in resolved(out+'\n'+err)};assert dynamic==old_dynamic,'Parent/new exact transitive path/bytes differ';assert all(H(p)==h for p,h in plan['reuse_objects_SHA256'].items());maps=[s for s in Path('/proc/self/maps').read_text().splitlines()if re.search(r'lib(?:infiniops|infiniccl|infinirt|infinilm_runtime|torch|cudart|amdhip64|galaxyhip)',s)];assert not maps and 'torch'not in sys.modules
  q.update(passed=True,parent_project_maps=maps,all32_linked_objects_SHA256={p:H(p)for p in objects},library=str(library),library_SHA256=H(library),strong_parent_exports_retained=len(old_syms),new_CABI=plan['required_new_exports'],NEEDED=needed(elf),new_transitive_library_SHA256=dynamic,qualification='Actual32originalMD graph/5consistent inlineobserver consumers, all5freshM==MD/27exactobjects/32link/parentstrongexports+4newCABI/ELF/fullparentENV dualstreamldd only. No originalcontract replay or Native/SDK/device/Model execution; eventdriver memory, numeric and performance UNKNOWN.')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','new_HOST','new_GPU','elapsed_s','error']}));return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
