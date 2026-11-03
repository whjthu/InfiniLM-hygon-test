"""One isolated CPU Runtime caller card; never install/replace any shared DSO."""
import hashlib,json,os,pathlib,re,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parent
C=pathlib.Path('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM')
B=pathlib.Path('/refactor/candidates/Kimi-loader-row-copy-r436-Runtime-CPU-v4')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
started=time.monotonic()
q={'completed':False,'passed':False,'scope':'Actual CPU Runtime/Tensor caller only; no Native, weights or Model; isolated DSO never installed','new_GPU':0,'new_GPU_TUs':0,'commands':[],'samples':[]}
def run(cmd,name,limit=60,env=None):
 left=255-(time.monotonic()-started);assert left>0
 before=time.monotonic();p=subprocess.run(cmd,cwd=C,env=env,capture_output=True,text=True,timeout=min(limit,left))
 (B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr)
 q['commands'].append({'name':name,'argv':cmd,'rc':p.returncode,'wall_s':time.monotonic()-before,'stdout_SHA256':sha(B/(name+'.stdout')),'stderr_SHA256':sha(B/(name+'.stderr'))})
 assert p.returncode==0,{'operation':name,'rc':p.returncode,'stderr':p.stderr[-3500:]}
 return p.stdout
def symbols(p,label):
 text=run(['nm','-D','--defined-only',str(p)],label,15)
 return {x.split()[-1]:x.split()[-2]for x in text.splitlines()if len(x.split())>=3 and x.split()[-2]in ['T','D','B','R']}
def main():
 B.mkdir(exist_ok=False)
 try:
  q['actual_CPU']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
  q['actual_GPU_visibility']={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}
  assert set(q['actual_GPU_visibility'].values())=={'-1'}
  q['source_manifest_SHA256']=sha(D/'SOURCE_MANIFEST.json');assert q['source_manifest_SHA256']==os.environ['R436_RUNTIME_SOURCE_SHA256']
  for p,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/p)==h,p
  c=json.loads((D/'BUILD_CONTRACT.json').read_bytes());q['baseline_Runtime_SHA256']=sha(c['baseline_Runtime']);assert q['baseline_Runtime_SHA256']==c['baseline_SHA256']
  for p,h in c['runtime_objects_SHA256'].items():assert sha(p)==h,p
  for p,h in c['project_source_pins'].items():assert sha(p)==h,p
  assert sha(C/'csrc/runtime/tensor/copy.cc')==c['original_copy_source_SHA256']
  assert sha(D/'cpu_strided_copy_r2515.hpp')==c['original_helper_SHA256']
  affinity=sorted(os.sched_getaffinity(0));assert affinity;os.sched_setaffinity(0,{affinity[0]});q['actual_CPU_affinity']=sorted(os.sched_getaffinity(0))
  flags=c['HOST_prefix']+['-I'+str(D),'-pthread']
  # Check the actual compiler include graph both before and after each TU.
  # These are current-source snapshots, not claims about historical headers.
  q['new_HOST_compiled']=[]
  for name in ['copy.cc','caller.cpp']:
   md=B/(name+'.d');pre=B/(name+'.pre.d');obj=B/(name+'.o')
   run(flags+['-M','-MF',str(pre),'-MT',str(obj),str(D/name)],name+'-pre-MD',30)
   def deps(path):
    s=path.read_text().replace('\\\n',' ');return {str((C/p).resolve()if not p.startswith('/')else pathlib.Path(p).resolve())for p in s.split(':',1)[1].split()}
   pins={p:sha(p)for p in deps(pre)}
   run(flags+['-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],name+'-compile',95)
   assert deps(md)==set(pins)and all(sha(p)==h for p,h in pins.items())
   q['new_HOST_compiled'].append({'source':name,'object':str(obj),'object_SHA256':sha(obj),'MD_SHA256':sha(md),'dependency_SHA256':pins})
  lib=B/'libinfinilm_runtime_r436.so'
  objs=[str(B/'copy.cc.o')if p==c['old_copy_object']else p for p in c['runtime_object_order']]
  sdk='/refactor/install/hygon-kimi-5f91ad7-r3672/lib'
  libs=['-L'+sdk,'-linfiniops','-linfiniccl','-linfinirt','-Wl,-rpath,'+sdk]
  run(['/usr/bin/g++','-shared','-m64','-fPIC','-pthread','-Wl,--no-undefined','-o',str(lib),*objs,*libs],'private-Runtime-link',35)
  old=symbols(c['baseline_Runtime'],'old-runtime-exports');new=symbols(lib,'new-runtime-exports')
  assert old and all(new.get(k)==v for k,v in old.items()),'old strong ABI missing'
  assert new.get('modern_cpu_copy_row_hits_r436')=='T'
  q['preserved_old_strong_exports']=len(old);q['new_Runtime_SHA256']=sha(lib)
  q['reused_Runtime_objects_SHA256']={p:h for p,h in c['runtime_objects_SHA256'].items()if p!=c['old_copy_object']}
  for label,runtime in [('baseline',pathlib.Path(c['baseline_Runtime'])),('candidate',lib)]:
   ldd=run(['ldd','-r',str(runtime)],label+'-ldd',20);assert 'not found'not in ldd and 'undefined symbol'not in ldd
   run(['/usr/bin/g++','-m64','-pthread',str(B/'caller.cpp.o'),str(runtime),*libs,'-ldl','-o',str(B/label)],label+'-caller-link',20)
   dynamic=run(['readelf','-d',str(B/label)],label+'-caller-DT-NEEDED',10);assert str(runtime)in dynamic,'caller does not bind exact Runtime path'
  def caller(label,on,operation,width,name):
   env=dict(os.environ);env['INFINILM_CPU_ROW_COPY_R436']=str(on)
   r=json.loads(run([str(B/label),operation,str(width)],name,15,env));assert r['passed']and not r['accelerator_fd_open']
   target=c['baseline_Runtime']if label=='baseline'else str(lib);assert pathlib.Path(r['provider']).resolve()==pathlib.Path(target).resolve()
   return r
  original=caller('baseline',0,'check',0,'actual-original-semantic')
  off=caller('candidate',0,'check',0,'new-runtime-default-OFF-semantic')
  on=caller('candidate',1,'check',0,'new-runtime-optin-ON-semantic')
  assert original['cases']==off['cases']==on['cases']==36
  assert original['stats_delta']==off['stats_delta']==on['stats_delta']
  assert off['chosen']==0 and on['chosen']>0 and on['fallback']>0
  q['actual_Tensor_semantic']={'original':original,'default_OFF':off,'optin_ON':on,'all_byte_owner_overlap_fallback_stats_checks_passed':True}
  # Compare installed old04a to candidateON, not a reconstructed scalar or
  # extra-instrumented OFF path. Each sample includes allocation/destruction.
  for width in [96,6]:
   for i,mode in enumerate(['OFF','ON']+['OFF','ON','ON','OFF']*4):
    r=caller('baseline'if mode=='OFF'else'candidate',int(mode=='ON'),'bench',width,f'width{width}-sample{i}-{mode}')
    assert r['copy_calls_timed']==128 and (mode=='OFF'or r['chosen']==129)
    q['samples'].append({'width':width,'warm':i<2,'index':i,'mode':mode,**r})
  assert sha(c['baseline_Runtime'])==c['baseline_SHA256']
  assert all(sha(p)==h for p,h in c['runtime_objects_SHA256'].items())
  q.update(passed=True,Model_GO=False,shared_install_modified=False,device_copy_or_loader_speedup_qualified=False)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  q.update(completed=True,wall_s=time.monotonic()-started);(B/'result.json').write_text(json.dumps(q,indent=2)+'\n')
  print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','wall_s','new_Runtime_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
