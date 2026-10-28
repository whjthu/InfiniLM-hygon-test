import pathlib,json,hashlib,subprocess,shlex,time,os
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/GLM-C1-decode-observer-r169';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);B.mkdir(exist_ok=False)
q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'GPU_TUs_compiled':0,'commands':[]};start=time.monotonic()
def run(a,n):
 q['commands'].append(a);z=subprocess.run(a,capture_output=True,text=True,timeout=180);(B/(n+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(n,z.stderr[-4000:]);return z.stdout
def deps(text):return set(shlex.split(text.replace('\\\n',' ').split(':',1)[1]))
def replay(a):
 out=[];i=0
 while i<len(a):
  if a[i]in ['-MF','-MT','-MQ','-o','-c']:i+=2;continue
  if a[i]in ['-MD','-MMD','-MP']:i+=1;continue
  out.append(a[i]);i+=1
 return out
try:
 m=json.loads((D/'source-manifest-r169.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 cp_path=R/'candidates/GLM-critical-observer-r131/CPU-build.json';assert sha(cp_path)==m['parent_receipt_SHA256'];cp=json.loads(cp_path.read_bytes());assert cp==json.loads((D/'parent-CPU-r131.json').read_bytes())
 assert cp['completed']and cp['CPU_only']and not cp['GPU_initialized']and cp['GPU_TUs_compiled']==0 and sha(cp['library'])==cp['library_SHA256']==m['parent_native_SHA256']
 for field in ['reused_objects_SHA256','actual_binary_SHA256','dependency_SHA256']:
  for n,h in cp[field].items():assert sha(n)==h,n
 for c in cp['compiled']:assert sha(c['object'])==c['object_SHA256']and sha(c['source'])==c['source_SHA256']
 contract=B/'CPUselector';run(['/usr/bin/g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_selector_contract_r169.cpp'),'-o',str(contract)],'selector-build');q['selector_contract']=json.loads(run([str(contract)],'selector-run'));assert q['selector_contract']['passed']
 compiled=[];hashes={}
 for c in cp['compiled']:
  n=pathlib.Path(c['source']).name;assert sha(D/n)==c['source_SHA256'],n
  original=next(a for a in cp['commands']if '-c'in a and c['source']in a and '-o'in a and a[a.index('-o')+1]==c['object'])
  prefix=replay(original);prefix.insert(1,'-I'+str(D));src=D/n;obj=B/(n+'.o');md=B/(n+'.d');run(prefix+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],n+'.compile');fresh=run(prefix+['-M',str(src)],n+'.fresh-M');assert deps(fresh)==deps(md.read_text())
  assert str(D/'glm_sparse_observer_r131.hpp')in deps(fresh)and str(D/'glm_decode_selector_r169.hpp')in deps(fresh)
  for path in deps(fresh):hashes[path]=sha(path)
  compiled.append(dict(source=str(src),source_SHA256=sha(src),object=str(obj),object_SHA256=sha(obj),replaces=c['object'],MD_SHA256=sha(md),fresh_M_equal_MD=True,full_semantic_argv_replayed=True))
 link=list(next(a for a in cp['commands']if '-shared'in a and '-o'in a));replace={c['replaces']:c['object']for c in compiled};assert all(link.count(k)==1 for k in replace);link=[replace.get(x,x)for x in link];lib=B/'libGLM_C1_decode_observer_r169.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 names=lambda s:{x.split()[-1]for x in s.splitlines()if x.split()};old=names(run(['nm','-D','--defined-only',cp['library']],'parentexports'));new=names(run(['nm','-D','--defined-only',str(lib)],'exports'));q['missing_parent_symbols']=sorted(old-new);assert not q['missing_parent_symbols'];assert {'modern_glm_sparse_prepare_r131','modern_glm_sparse_policy_r131','modern_glm_sparse_read_r131'}<=new
 ldd=run(['ldd',str(lib)],'ldd');assert 'not found'not in ldd;dynamic=run(['readelf','-d',str(lib)],'dynamic')
 reused=dict(cp['reused_objects_SHA256']);reused={n:h for n,h in reused.items()if n not in replace}
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,dependency_SHA256=hashes,HOST_TUs_compiled=4,parent_native_SHA256=cp['library_SHA256'],parent_receipt_SHA256=m['parent_receipt_SHA256'],reused_objects_SHA256=reused,actual_binary_SHA256=cp['actual_binary_SHA256'],all_parent_public_symbols_retained=True,ABI=1,Context_layout_changed=False,default_OFF=True,events_per_owner=512,new_synchronization_calls=0,math_changes=0,schedule_changes=0,ldd=ldd,ELF_dynamic=dynamic,manifest_SHA256=sha(D/'source-manifest-r169.json'),normal_CPU_exit=True,Model_GPU_qualified=False,selector_revision='r169',selector_required=[0,0,0,0,3],GPU_busy_claim=False,TP_only_UNKNOWN=True)
except BaseException as e:q['error']=repr(e)
q['seconds']=time.monotonic()-start;q['producer']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');(B/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
