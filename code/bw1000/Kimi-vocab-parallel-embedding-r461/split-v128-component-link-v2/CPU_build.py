from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time,traceback
C=Path(__file__).resolve().parent;D=C.parent/'split-v128-component-source-v1';B=Path('/refactor/candidates/Kimi-split-V128-component-link-v2')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))
def repair_link(old,obj,native,output,mapfile):
 assert old.count(obj)==1 and old.count(native)==1
 libs=[x for x in old if x.startswith('-l')]
 link=[x for x in old if x not in [obj,native]]
 link[1:1]=[obj,native]
 link[link.index('-o')+1]=output
 link=[('-Wl,-Map,'+mapfile)if x.startswith('-Wl,-Map,')else x for x in link]
 assert link[1:3]==[obj,native]and [x for x in link if x.startswith('-l')]==libs
 assert link.index(obj)<min(i for i,x in enumerate(link)if x.startswith('-l'))
 return link

def main():
 assert not B.exists(),'existing candidate: reconcile neverrepeat';B.mkdir();start=time.monotonic();q=dict(completed=False,passed=False,GPU=0,new_HOST=0,new_GPU=0,production_recompiled=0,SDK_device_execution=False,Native_dlopen=False,commands=[],logs={})
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(args,label,cap=90):
  left=270-(time.monotonic()-start);assert left>0;q['phase']=label;q['commands'].append(args);save();z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=min(left,cap));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['logs'][label]=dict(rc=z.returncode,bytes=p.stat().st_size,SHA256=H(p));save()
  if z.returncode:raise RuntimeError(label+' failed; original log retained')
  return z.stdout
 try:
  assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  assert H(C/'SOURCE_MANIFEST.json')==os.environ['KIMI_SPLITV_COMPONENT_SHA256'];m=json.loads((C/'SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert H(C/n)==h,n
  assert H(D/'SOURCE_MANIFEST.json')==m['component_source_manifest_SHA256']
  for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert H(D/n)==h,n
  pin=json.loads((D/'PARENT.json').read_bytes());rp=pin['CPU1075_receipt'];assert H(rp['path'])==rp['SHA256'];prior=json.loads(Path(rp['path']).read_bytes());assert prior['passed']and prior['actual_CPU']=='910407.1075'
  assert H(pin['Native']['path'])==pin['Native']['SHA256']==prior['native_SHA256'];assert H(pin['build_contract']['path'])==pin['build_contract']['SHA256'];c=json.loads(Path(pin['build_contract']['path']).read_bytes())
  for obj,h in prior['all_linked_object_SHA256'].items():assert H(obj)==h,obj
  for value in prior['actual_dependency_library_SHA256'].values():assert H(value['path'])==value['SHA256']
  local=json.loads((D/'local-entry-result.json').read_bytes());assert local['returncode']==0 and local['actual_stdout']['passed']and local['actual_stdout']['actual_Native_leaf']=='recording'
  for n,h in local['source_SHA256'].items():assert H(D/n)==h
  q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=H(C/'SOURCE_MANIFEST.json'),parent_receipt=rp,parent_Native=pin['Native'],actual_dependency_library_SHA256=prior['actual_dependency_library_SHA256'],local_entry_reused=local,old_CPU_contracts_reexecuted=False);save()
  original=json.loads((C/'PARENT.json').read_bytes());pin1076=original['receipt1076'];assert H(pin1076['path'])==pin1076['SHA256'];failed=json.loads(Path(pin1076['path']).read_bytes());assert failed['completed']and not failed['passed']and failed['actual_CPU']=='910407.1076'and failed['phase']=='probe-link'
  compiled=failed['compiled'];obj=Path(original['object1076']['path']);assert H(obj)==compiled['object_SHA256']==original['object1076']['SHA256'];assert H(D/'probe.cpp')==compiled['source_SHA256']
  md=obj.with_suffix('.d');before=obj.parent/'probe.before.d';assert H(md)==compiled['MD_SHA256']and deps(md)==set(compiled['all_MD'])and deps(before)==set(compiled['all_MD']);assert H(before)==original['before_MD_SHA256']
  assert all(H(p)==h for p,h in compiled['all_MD'].items())
  q['reused1076']=dict(receipt=pin1076,object=str(obj),object_SHA256=H(obj),MD_SHA256=H(md),MD_records=len(compiled['all_MD']),source_SHA256=H(D/'probe.cpp'),passed=False,successful_HOST_recompiled=False);save()
  output=B/'libKimi_split_v128_component.so';link=repair_link(failed['commands'][-1],str(obj),pin['Native']['path'],str(output),str(B/'link.map'))
  run(link,'probe-link',45);symbols=run(['nm','-D',str(output)],'probe-symbols',15)
  required=['kimi_split_probe_'+x for x in ['create','begin','reset','seed','query','observe','audit_pages','destroy','error','symbol_path']]
  assert all(any(line.split()[-1]==n and line.split()[-2]=='T'for line in symbols.splitlines()if len(line.split())>=3)for n in required)
  for marker in ['_ZN15kimi_split_v1288allocate','_ZN15kimi_split_v1287forward','kimi_split_store']:assert any(marker in line and ' U 'in line for line in symbols.splitlines()),marker
  elf=run(['readelf','-d',str(output)],'probe-ELF',10);assert 'SYMBOLIC'in elf and 'libKimi_split_v128_private.so'in elf
  run(['ldd','-r',str(output)],'probe-relocation-full-streams',30);ldd=(B/'probe-relocation-full-streams.log').read_text();assert 'not found'not in ldd and 'undefined symbol'not in ldd
  q['project_maps']=[l for l in Path('/proc/self/maps').read_text().splitlines()if any(s in l for s in ['libKimi','libinfiniops','libinfinilm','libcudart'])];assert not q['project_maps'];q.update(passed=True,probe=str(output),probe_SHA256=H(output),CAPI=required,qualification='0compile link-only reusing1076HOST/allMD/localentry; source declarations call existing c78 allocate/forward; no SDK/Runtime/FA/device/numeric/fit qualification')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save();print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','phase','error','elapsed_s','probe_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
