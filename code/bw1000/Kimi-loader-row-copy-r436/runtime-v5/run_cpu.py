"""One isolated CPU Runtime caller card; never install/replace any shared DSO."""
import hashlib,json,os,pathlib,re,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parent
C=pathlib.Path('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM')
B=pathlib.Path('/refactor/candidates/Kimi-loader-row-copy-r436-Runtime-CPU-v5')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
started=time.monotonic()
q={'completed':False,'passed':False,'scope':'Actual CPU Runtime/Tensor caller only; no Native, weights or Model; isolated DSO never installed','new_GPU':0,'new_GPU_TUs':0,'HOST_compile_count':0,'Runtime_link_count':0,'commands':[],'samples':[]}
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
  prior_source=D.parent/'runtime-v4';prior_B=pathlib.Path('/refactor/candidates/Kimi-loader-row-copy-r436-Runtime-CPU-v4')
  pins=json.loads((D/'PRIOR_884.json').read_bytes())
  assert sha(prior_source/'SOURCE_MANIFEST.json')==pins['source_manifest_SHA256']
  for p,h in json.loads((prior_source/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(prior_source/p)==h,p
  receipt=prior_B/'result.json';assert sha(receipt)==pins['receipt_SHA256']
  prior=json.loads(receipt.read_bytes());assert prior['completed']and not prior['passed']and prior['actual_CPU']=='910407.884'
  assert prior['samples']==[] and prior['source_manifest_SHA256']==pins['source_manifest_SHA256']
  assert prior['new_Runtime_SHA256']==pins['library_SHA256']
  assert [v['name']for v in prior['commands']]==pins['completed_operations']
  assert all(v['rc']==0 for v in prior['commands'])
  for item in prior['new_HOST_compiled']:
   assert sha(item['object'])==item['object_SHA256']
   assert sha(prior_B/(item['source']+'.d'))==item['MD_SHA256']
   for path,h in item['dependency_SHA256'].items():assert sha(path)==h,path
  c=json.loads((prior_source/'BUILD_CONTRACT.json').read_bytes())
  assert sha(c['baseline_Runtime'])==c['baseline_SHA256']
  for p,h in c['runtime_objects_SHA256'].items():assert sha(p)==h,p
  for p,h in c['project_source_pins'].items():assert sha(p)==h,p
  lib=prior_B/'libinfinilm_runtime_r436.so';assert sha(lib)==pins['library_SHA256']
  q.update(original_884_passed=False,original_884_receipt_SHA256=sha(receipt),original_884_source_manifest_SHA256=pins['source_manifest_SHA256'],reused_HOST_objects=prior['new_HOST_compiled'],reused_Runtime_objects_SHA256=prior['reused_Runtime_objects_SHA256'],new_Runtime_SHA256=sha(lib),preserved_old_strong_exports=prior['preserved_old_strong_exports'],actual_LD_LIBRARY_PATH=os.environ['LD_LIBRARY_PATH'])
  assert '/usr/local/lib/python3.10/dist-packages/torch/lib'in os.environ['LD_LIBRARY_PATH'].split(':')
  affinity=sorted(os.sched_getaffinity(0));assert affinity;os.sched_setaffinity(0,{affinity[0]});q['actual_CPU_affinity']=sorted(os.sched_getaffinity(0))
  sdk='/refactor/install/hygon-kimi-5f91ad7-r3672/lib'
  libs=['-L'+sdk,'-linfiniops','-linfiniccl','-linfinirt','-Wl,-rpath,'+sdk]
  # Both existing DSOs must resolve completely before either new caller link.
  q['actual_dynamic_dependencies']={}
  for label,runtime in [('baseline',pathlib.Path(c['baseline_Runtime'])),('candidate',lib)]:
   ldd=run(['ldd','-r',str(runtime)],label+'-ldd',25)
   assert 'not found'not in ldd and 'undefined symbol'not in ldd
   paths=set(re.findall(r'(?:=>\s*)?(/[^\s()]+)',ldd))
   q['actual_dynamic_dependencies'][label]={p:sha(p)for p in sorted(paths)if pathlib.Path(p).is_file()}
  for label,runtime in [('baseline',pathlib.Path(c['baseline_Runtime'])),('candidate',lib)]:
   run(['/usr/bin/g++','-m64','-pthread',str(prior_B/'caller.cpp.o'),str(runtime),*libs,'-ldl','-o',str(B/label)],label+'-caller-link',20)
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
