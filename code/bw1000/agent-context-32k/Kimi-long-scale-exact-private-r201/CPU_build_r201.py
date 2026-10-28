"""One HOST TU using locked compile/link contexts. No argv export or GPU work."""
import argparse,ast,hashlib,json,os,pathlib,shlex,subprocess,sys,time
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Kimi-long-scale-exact-r201'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
BASES={'pair_probe':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/pair_probe.o','worker':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/worker.o','delta':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/delta.o','trace':'/refactor/candidates/Kimi-speculative-batch-Model-r9122c/trace.o','infer_engine':'/refactor/candidates/kimi-compact-cache-r4974/objects/infer_engine.o','allocate':'/refactor/candidates/kimi-compact-cache-r4974/objects/kimi_k3_allocate_cache.o'}
def semantic(c):
 out=[c[0]];i=1
 while i<len(c):
  z=c[i]
  if z in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
  if z in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if z.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or z.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  out.append(z);i+=1
 return out
def dependencies(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--cpu-build-authorized',action='store_true');a=p.parse_args();assert a.cpu_build_authorized
 mf=D/'source-manifest-r201.json';assert sha(mf)==a.manifest_sha;m=json.loads(mf.read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert 'torch'not in sys.modules and 'transformers'not in sys.modules and not B.exists();B.mkdir()
 pins=json.loads((D/'actual-context-pins-r201.json').read_bytes());start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'GPU_GO':False,'Native_dlopen':False,'Runtime_created':False,'new_HOST_TUs':1,'new_HIP_TUs':0,'compiler_argv_exported':False,'source_manifest_SHA256':a.manifest_sha,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')}
 def run(cmd,label,timeout=180):
  z=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,('CPU_STEP_FAILED',label,z.returncode);return z.stdout
 try:
  receipts={}
  for label in ['Native179','moe','link']:
   assert sha(pins[label+'_receipt'])==pins[label+'_receipt_SHA256'];receipts[label]=json.loads(pathlib.Path(pins[label+'_receipt']).read_bytes());assert receipts[label]['completed']and receipts[label]['CPU_only']
  base=receipts['Native179'];assert not base['GPU']and not base['Native_dlopen'];assert base['native_SHA256']==pins['Native179_SHA256']and sha(base['native'])==base['native_SHA256']
  for path,h in base['all_linked_object_SHA256'].items():assert sha(path)==h,path
  assert base['all_linked_object_SHA256'][pins['moe_object']]==pins['moe_object_SHA256']and sha(pins['moe_object'])==pins['moe_object_SHA256']
  assert sha(pins['qualified_moe_source'])==pins['qualified_moe_source_raw_SHA256'];assert hashlib.sha256(pathlib.Path(pins['qualified_moe_source']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pins['qualified_moe_source_LF_SHA256']
  run([sys.executable,'-B',str(D/'CPU_source_contract_r201.py')],'source-contract',30)
  oldcc=next(c[:]for c in receipts['moe']['commands']if '-c'in c and c[c.index('-c')+1]==pins['qualified_moe_source'])
  flags=semantic(oldcc);assert pathlib.Path(flags[0]).name=='g++'and '-fPIC'in flags and '-pthread'in flags and any(x in flags for x in ['-O2','-O3']);assert not any(x.endswith('.cu')or x in ['-xhip','--hip-link']for x in flags)
  obj=B/'moe.o';cmd=[flags[0],'-I'+str(D),*flags[1:],'-MD','-MF',str(B/'moe.d'),'-c',str(D/'kimi_moe_scale_exact_r201.cpp'),'-o',str(obj)]
  run(cmd,'moe-host');deps=dependencies(B/'moe.d');assert str(D/'scale_exact_scope_r201.hpp')in deps
  run([flags[0],'-I'+str(D),*flags[1:],'-M','-MF',str(B/'moe.fresh.d'),'-MT',str(obj),str(D/'kimi_moe_scale_exact_r201.cpp')],'fresh-dependencies',60)
  assert dependencies(B/'moe.fresh.d')==deps
  md={path:sha(path)for path in deps if pathlib.Path(path).is_file()};assert md[str(D/'scale_exact_scope_r201.hpp')]==sha(D/'scale_exact_scope_r201.hpp')
  exe=B/'CPU-contract';run(['g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_contract_r201.cpp'),'-o',str(exe)],'contract-build',40);q['pure_CPU_contract']=json.loads(run([str(exe)],'contract-run',20))
  linkbase=receipts['link'];assert linkbase['native_SHA256']=='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'
  link=next(c[:]for c in linkbase['commands']if '-shared'in c and '-o'in c and c[c.index('-o')+1]==linkbase['native'])
  for label,old in BASES.items():assert link.count(old)==1;link[link.index(old)]='/refactor/candidates/Kimi-long-agent32K-r179/'+label+'.o'
  assert {x for x in link if x.endswith('.o')}==set(base['all_linked_object_SHA256'])
  assert link.count(pins['moe_object'])==1;link[link.index(pins['moe_object'])]=str(obj);native=B/'libKimi_long_scale_exact_r201.so';link[link.index('-o')+1]=str(native)
  for i,z in enumerate(link):
   if z.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
  run(link,'native-link');objects={x:sha(x)for x in link if x.endswith('.o')};unchanged={x:h for x,h in base['all_linked_object_SHA256'].items()if x!=pins['moe_object']};assert all(objects[x]==h for x,h in unchanged.items())and len(objects)==len(base['all_linked_object_SHA256'])
  oldnm=subprocess.check_output(['nm','-g','--defined-only',pins['moe_object']],text=True,timeout=30);newnm=run(['nm','-g','--defined-only',str(obj)],'moe-symbols',30)
  methods=lambda s:sorted(x.split()[-1]for x in s.splitlines()if 'KimiK3MoE'in x or 'KimiK3Experts'in x)
  assert methods(oldnm)==methods(newnm)and methods(oldnm)
  capi=lambda s:{x.split()[-1]for x in s.splitlines()if x.split()and x.split()[-1].startswith('modern_')}
  oldcapi=capi(subprocess.check_output(['nm','-D','--defined-only',base['native']],text=True,timeout=30));newcapi=capi(run(['nm','-D','--defined-only',str(native)],'native-symbols',30));assert oldcapi==newcapi
  ldd=run(['ldd','-r',str(native)],'ldd-relocations',60);assert 'not found'not in ldd and 'undefined symbol:'not in ldd
  actual={};expected=base['actual_dependency_library_SHA256']
  for line in ldd.splitlines():
   w=line.split()
   if len(w)>=3 and w[0]in expected and w[1]=='=>':assert sha(w[2])==expected[w[0]]['SHA256'];actual[w[0]]={'path':w[2],'SHA256':sha(w[2])}
  assert set(actual)==set(expected)
  q.update(completed=True,native=str(native),native_SHA256=sha(native),all_linked_object_SHA256=objects,unchanged_objects_SHA256=unchanged,actual_dependency_library_SHA256=actual,new_HOST_object_SHA256=sha(obj),all_new_MD={'moe':md},fresh_MD_matches=True,old_CAPI_exports_preserved=len(oldcapi),public_class_layout_changed=False,scale_shape_stride_alias_source_unchanged=True,logical_design_padding_removed_per_MoE=42860544,actual_device_saving='UNKNOWN',full32K_peak='UNKNOWN',math_quality='NOT_RUN',elapsed_s=time.monotonic()-start)
 except BaseException as e:q.update(error_type=type(e).__name__,error=str(e));raise
 finally:(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','new_HOST_TUs','new_HIP_TUs','native_SHA256','elapsed_s','error_type','error']}))
if __name__=='__main__':main()
