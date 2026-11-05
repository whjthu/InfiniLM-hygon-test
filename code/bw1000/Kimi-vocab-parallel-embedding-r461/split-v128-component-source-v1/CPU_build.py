from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-split-V128-component-CPU-v1')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert not B.exists(),'existing candidate: reconcile neverrepeat';B.mkdir();start=time.monotonic();q=dict(completed=False,passed=False,GPU=0,new_HOST=1,new_GPU=0,production_recompiled=0,SDK_device_execution=False,Native_dlopen=False,commands=[],logs={})
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(args,label,cap=90):
  left=270-(time.monotonic()-start);assert left>0;q['phase']=label;q['commands'].append(args);save();z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=min(left,cap));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['logs'][label]=dict(rc=z.returncode,bytes=p.stat().st_size,SHA256=H(p));save()
  if z.returncode:raise RuntimeError(label+' failed; original log retained')
  return z.stdout
 try:
  assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  assert H(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_SPLITV_COMPONENT_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert H(D/n)==h,n
  pin=json.loads((D/'PARENT.json').read_bytes());rp=pin['CPU1075_receipt'];assert H(rp['path'])==rp['SHA256'];prior=json.loads(Path(rp['path']).read_bytes());assert prior['passed']and prior['actual_CPU']=='910407.1075'
  assert H(pin['Native']['path'])==pin['Native']['SHA256']==prior['native_SHA256'];assert H(pin['build_contract']['path'])==pin['build_contract']['SHA256'];c=json.loads(Path(pin['build_contract']['path']).read_bytes())
  for obj,h in prior['all_linked_object_SHA256'].items():assert H(obj)==h,obj
  for value in prior['actual_dependency_library_SHA256'].values():assert H(value['path'])==value['SHA256']
  local=json.loads((D/'local-entry-result.json').read_bytes());assert local['returncode']==0 and local['actual_stdout']['passed']and local['actual_stdout']['actual_Native_leaf']=='recording'
  for n,h in local['source_SHA256'].items():assert H(D/n)==h
  q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),parent_receipt=rp,parent_Native=pin['Native'],actual_dependency_library_SHA256=prior['actual_dependency_library_SHA256'],local_entry_reused=local,old_CPU_contracts_reexecuted=False);save()
  prefix=[c['HOST_prefix'][0],'-I'+str(D),*c['HOST_prefix'][1:]];source=D/'probe.cpp';obj=B/'probe.o';before=B/'probe.before.d';after=B/'probe.d'
  run(prefix+['-M','-MF',str(before),'-MT',str(obj),str(source)],'probe-before-MD',30);snapshot={p:H(p)for p in sorted(deps(before))}
  run(prefix+['-MD','-MF',str(after),'-c',str(source),'-o',str(obj)],'probe-compile',100);assert deps(after)==set(snapshot)and all(H(p)==h for p,h in snapshot.items());q['compiled']=dict(source_SHA256=H(source),object_SHA256=H(obj),all_MD=snapshot,MD_SHA256=H(after));save()
  link=[a for a in c['baseline_link_argv']if not a.endswith('.o')];output=B/'libKimi_split_v128_component.so';i=link.index('-o');link[i:i]=[str(obj),pin['Native']['path']];link[link.index('-o')+1]=str(output);link=[('-Wl,-Map,'+str(B/'link.map'))if a.startswith('-Wl,-Map,')else a for a in link]
  run(link,'probe-link',45);symbols=run(['nm','-D',str(output)],'probe-symbols',15)
  required=['kimi_split_probe_'+x for x in ['create','begin','reset','seed','query','observe','audit_pages','destroy','error','symbol_path']]
  assert all(any(line.split()[-1]==n and line.split()[-2]=='T'for line in symbols.splitlines()if len(line.split())>=3)for n in required)
  for marker in ['_ZN15kimi_split_v1288allocate','_ZN15kimi_split_v1287forward','kimi_split_store']:assert any(marker in line and ' U 'in line for line in symbols.splitlines()),marker
  elf=run(['readelf','-d',str(output)],'probe-ELF',10);assert 'SYMBOLIC'in elf and 'libKimi_split_v128_private.so'in elf
  run(['ldd','-r',str(output)],'probe-relocation-full-streams',30);ldd=(B/'probe-relocation-full-streams.log').read_text();assert 'not found'not in ldd and 'undefined symbol'not in ldd
  q['project_maps']=[l for l in Path('/proc/self/maps').read_text().splitlines()if any(s in l for s in ['libKimi','libinfiniops','libinfinilm','libcudart'])];assert not q['project_maps'];q.update(passed=True,probe=str(output),probe_SHA256=H(output),CAPI=required,qualification='new1HOST+link only; source declarations call existing c78 allocate/forward; no SDK/Runtime/FA/device/numeric/fit qualification')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save();print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','phase','error','elapsed_s','probe_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
