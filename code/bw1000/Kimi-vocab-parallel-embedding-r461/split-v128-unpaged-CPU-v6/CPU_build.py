"""Only one changed faHOST TU and two links, exact original objects/argv/ENV."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time,traceback,re
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-splitV-unpaged-CPU-v6');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert not B.exists(),'existing candidate; reconcile only';B.mkdir();start=time.monotonic();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],new_HOST_TUs=1,new_GPU_TUs=0,old_contracts_reexecuted=False,GPU_job=False,Native_dlopen=False,explicit_SDK_calls=0,weights=False,commands=[],logs={},SDK_planning_supports_executed=False,preload_child_initialization='UNKNOWN; exact GPU preload supplied only to childldd, no independent constructortrace')
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(args,label,cap=60,env=None):
  left=265-(time.monotonic()-start);assert left>1;q['phase']=label;q['commands'].append(args);save()
  z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=min(left,cap),env=env)
  for stream,value in [('stdout',z.stdout),('stderr',z.stderr)]:
   p=B/(label+'.'+stream);p.write_text(value);q['logs'][label+'.'+stream]=dict(path=str(p),bytes=p.stat().st_size,SHA256=H(p))
  save();assert z.returncode==0,(label,z.stderr[-1500:]);return z.stdout+z.stderr
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['KIMI_V6_CLAIM_SHA']and claim['CPU_GO']and not claim['GPU_GO'];assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in m['files_SHA256'].items());q.update(source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'))
  parent=json.loads((D/'PARENT.json').read_bytes());ps={}
  for key in ['CPU1075_receipt','CPU1080_receipt','loader1089_receipt']:
   a=parent[key];assert H(a['path'])==a['SHA256'];ps[key]=json.loads(Path(a['path']).read_bytes());assert ps[key]['completed']and ps[key]['passed']
  native=ps['CPU1075_receipt'];probe=ps['CPU1080_receipt'];loader=ps['loader1089_receipt'];assert native['actual_CPU']=='910407.1075'and probe['actual_CPU']=='910407.1080'and loader['actual_CPU']=='910407.1089'
  assert H(native['native'])==native['native_SHA256']and H(probe['probe'])==probe['probe_SHA256'];assert len(native['all_linked_object_SHA256'])==36
  for p,h in native['all_linked_object_SHA256'].items():assert H(p)==h,p
  oldfa=native['compiled']['fa_backend'];assert H(oldfa['source'])==oldfa['source_SHA256']==parent['old_fa_source_SHA256'];assert H(oldfa['object'])==oldfa['object_SHA256'];assert H(Path(oldfa['object']).with_suffix('.d'))==oldfa['MD_SHA256']
  for p,h in oldfa['all_MD'].items():assert H(p)==h,p
  for v in native['actual_dependency_library_SHA256'].values():assert H(v['path'])==v['SHA256']
  srcpin=parent['SDK_provider_header'];assert H(srcpin['path'])==srcpin['SHA256']
  originalcompile=next(a for a in native['commands']if'-c'in a and oldfa['source']in a);prefix=originalcompile[:originalcompile.index('-MD')];assert prefix[0]=='g++'and '-std=c++17'in prefix
  obj=B/'fa_backend.o';before=B/'fa_backend.before.d';md=B/'fa_backend.d';src=D/'fa_backend.cpp'
  run(prefix+['-M','-MF',str(before),'-MT',str(obj),str(src)],'fa-before-M',30);actual=deps(before);allowed=set(oldfa['all_MD'])-{oldfa['source']}|{str(src)};assert actual<=allowed,sorted(actual-allowed);snapshot={p:H(p)for p in sorted(actual)};assert snapshot[str(src)]==H(src)
  run(prefix+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],'fa-HOST-compile',90);assert deps(md)==actual and all(H(p)==h for p,h in snapshot.items());q['compiled']=dict(source=str(src),source_SHA256=H(src),object=str(obj),object_SHA256=H(obj),MD_SHA256=H(md),fresh_M_SHA256=H(before),all_MD=snapshot)
  link=next(a[:]for a in native['commands']if'-shared'in a and native['native']in a);assert link.count(oldfa['object'])==1;lib=B/'libKimi_split_v128_private.so';link=[str(obj)if p==oldfa['object']else p for p in link];link[link.index('-o')+1]=str(lib);link=[('-Wl,-Map,'+str(B/'Native-link.map'))if p.startswith('-Wl,-Map,')else p for p in link]
  assert set(p for p in link if p.endswith('.o'))==set(native['all_linked_object_SHA256'])-{oldfa['object']}|{str(obj)};run(link,'Native-relink',40)
  oldlog=Path('/refactor/candidates/Kimi-split-V128-CPU-v5/new-symbols.log');assert H(oldlog)==native['logs']['new-symbols']['SHA256'];oldstrong={l.split()[-1]for l in oldlog.read_text().splitlines()if len(l.split())>=3 and l.split()[-2]in ['T','D','B','R']};symbols=run(['nm','-D','--defined-only',str(lib)],'Native-new-symbols',15);now={l.split()[-1]for l in symbols.splitlines()if len(l.split())>=3 and l.split()[-2]in ['T','D','B','R']};assert oldstrong<=now
  elf=run(['readelf','-d',str(lib)],'Native-ELF',10);assert 'SYMBOLIC'in elf
  reusedprobe=probe['reused1076'];assert H(reusedprobe['object'])==reusedprobe['object_SHA256']and H(Path(reusedprobe['object']).with_suffix('.d'))==reusedprobe['MD_SHA256'];rp=reusedprobe['receipt'];assert H(rp['path'])==rp['SHA256'];failed=json.loads(Path(rp['path']).read_bytes());assert all(H(p)==h for p,h in failed['compiled']['all_MD'].items())
  plink=next(a[:]for a in probe['commands']if'-shared'in a and probe['probe']in a);oldnative=probe['parent_Native']['path'];assert plink.count(oldnative)==1;plink=[str(lib)if p==oldnative else p.replace(str(Path(oldnative).parent),str(B))if p.startswith('-Wl,-rpath,')else p for p in plink];output=B/'libKimi_split_v128_component.so';plink[plink.index('-o')+1]=str(output);plink=[('-Wl,-Map,'+str(B/'probe-link.map'))if p.startswith('-Wl,-Map,')else p for p in plink];assert plink.index(str(lib))<min(i for i,p in enumerate(plink)if p.startswith('-l'));run(plink,'probe-relink',35)
  syms=run(['nm','-D',str(output)],'probe-symbols',15);assert all(any(line.split()[-1]==n and line.split()[-2]=='T'for line in syms.splitlines()if len(line.split())>=3)for n in probe['CAPI']);dynamic=run(['readelf','-d',str(output)],'probe-ELF',10);assert 'SYMBOLIC'in dynamic and 'libKimi_split_v128_private.so'in dynamic
  env=os.environ.copy();env['LD_LIBRARY_PATH']=str(B)+':'+parent['qualified_LD_LIBRARY_PATH'];env['LD_PRELOAD']=parent['qualified_LD_PRELOAD'];assert env['LD_PRELOAD']==os.environ['KIMI_V6_EXPECTED_PRELOAD'];ldd=run(['/usr/bin/ldd','-r',str(output)],'probe-ldd-full-streams',35,env);assert 'not found'not in ldd and 'undefined symbol'not in ldd;paths=sorted(set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',ldd,re.M)));resolved={p:H(p)for p in paths};assert resolved[str(lib)]==H(lib)
  expected=loader['all_transitive_resolved_library_SHA256'];oldpath=native['native'];assert oldpath not in resolved;assert set(resolved)==set(expected)-{oldpath}|{str(lib)}
  for p,h in resolved.items():assert p==str(lib)or h==expected[p],p
  q['project_maps']=[line for line in Path('/proc/self/maps').read_text().splitlines()if any(x in line for x in ['libKimi','libinfiniops','libinfinilm_runtime','libcudart'])];assert not q['project_maps']and 'torch'not in sys.modules;q.update(passed=True,native=str(lib),native_SHA256=H(lib),probe=str(output),probe_SHA256=H(output),parent_Native=dict(path=str(lib),SHA256=H(lib)),all_linked_object_SHA256={p:H(p)for p in link if p.endswith('.o')},reused_production_objects=35,reused_GPU_objects=True,reused_probe_object=reusedprobe,actual_dependency_library_SHA256=native['actual_dependency_library_SHA256'],CAPI=probe['CAPI'],old_strong_exports_retained=len(oldstrong),all_transitive_resolved_library_SHA256=resolved,ENV_SHA256=H(D/'ENV.sh'),LD_LIBRARY_PATH=env['LD_LIBRARY_PATH'],LD_PRELOAD=env['LD_PRELOAD'],preload_SHA256=H(env['LD_PRELOAD']),qualification='one faHOST recompile+production/probe relinks and exactENVldd, no SDK operator/Runtime/device/fit/numeric/perf executed; no oldcontractrerun')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save();print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','phase','error','elapsed_s','native_SHA256','probe_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
