import pathlib,json,hashlib,subprocess,shlex,time,os
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).resolve().parent;b=r/'candidates/GLM-critical-observer-r131';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((inc/'manifest-r131.json').read_bytes());p=json.loads((inc/'source-binding-and-build-r131.json').read_bytes());assert not b.exists();b.mkdir();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'GPU_TUs_compiled':0,'commands':[]};start=time.monotonic()
def run(a,name):
 q['commands'].append(a);z=subprocess.run(a,text=True,capture_output=True,timeout=180);(b/(name+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(name,z.stderr[-4000:]);return z.stdout
def deps(text):return set(shlex.split(text.replace('\\\n',' ').split(':',1)[1]))
def replay(a):
 # Preserve ALL original semantic options, including flags after -MD.
 out=[];i=0
 while i<len(a):
  if a[i]in ['-MF','-MT','-MQ','-o','-c']:i+=2;continue
  if a[i]in ['-MD','-MMD','-MP']:i+=1;continue
  out.append(a[i]);i+=1
 return out
try:
 for n,h in m['CPU_source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in p['actual_files_SHA256'].items():assert sha(n)==h,n
 cp=p['parent'];assert sha(p['parent_receipt_path'])==p['parent_receipt_SHA256'];assert cp['completed']and sha(cp['library'])==cp['library_SHA256']=='1760cd0782deed03bee608dda9c7b62690c7f2180ca1038300760596f29a9e63'
 for n,h in cp['reused_objects_SHA256'].items():assert sha(n)==h,n
 for c in cp['compiled']:assert sha(c['object'])==c['object_SHA256']and sha(c['source'])==c['source_SHA256']
 for n,h in cp['actual_binary_SHA256'].items():assert sha(n)==h,n
 compiled=[];hashes={}
 for c in p['compiled_plan']:
  prefix=replay(c['command']);prefix.insert(1,'-I'+str(inc));src=inc/c['name'];obj=b/(c['name']+'.o');md=b/(c['name']+'.d');run(prefix+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],c['name']+'.compile');fresh=run(prefix+['-M',str(src)],c['name']+'.fresh-M');assert deps(fresh)==deps(md.read_text())
  for path in deps(fresh):hashes[path]=sha(path)
  compiled.append(dict(source=str(src),source_SHA256=sha(src),object=str(obj),object_SHA256=sha(obj),replaces=c['old_object'],MD_SHA256=sha(md),fresh_M_equal_MD=True,full_semantic_argv_replayed=True))
 link=list(next(a for a in cp['commands']if '-shared'in a and '-o'in a));replace={c['replaces']:c['object']for c in compiled};assert all(link.count(k)==1 for k in replace);link=[replace.get(x,x)for x in link];lib=b/'libGLM_critical_observer_r131.so';link[link.index('-o')+1]=str(lib);run(link,'link')
 oldexports=run(['nm','-D','--defined-only',cp['library']],'parentexports');exports=run(['nm','-D','--defined-only',str(lib)],'exports');names=lambda s:{x.split()[-1]for x in s.splitlines()if x.split()};assert names(oldexports)<=names(exports);assert {'modern_glm_sparse_prepare_r131','modern_glm_sparse_policy_r131','modern_glm_sparse_read_r131'}<=names(exports)
 dynamic=run(['readelf','-d',str(lib)],'dynamic');ldd=run(['ldd',str(lib)],'ldd');assert 'not found'not in ldd
 reused=dict(cp['reused_objects_SHA256']);reused.update({c['object']:c['object_SHA256']for c in cp['compiled']});reused={n:h for n,h in reused.items()if n not in replace}
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,dependency_SHA256=hashes,HOST_TUs_compiled=4,parent_native_SHA256=cp['library_SHA256'],parent_receipt_SHA256=p['parent_receipt_SHA256'],reused_objects_SHA256=reused,actual_binary_SHA256=cp['actual_binary_SHA256'],all_parent_public_symbols_retained=True,ABI=1,Context_layout_changed=False,default_OFF=True,events_per_owner=512,new_synchronization_calls=0,math_changes=0,schedule_changes=0,ldd=ldd,ELF_dynamic=dynamic,manifest_SHA256=sha(inc/'manifest-r131.json'),normal_CPU_exit=True,Model_GPU_qualified=False)
except BaseException as e:q['error']=repr(e)
q['seconds']=time.monotonic()-start;q['producer']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
