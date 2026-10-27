import pathlib,json,hashlib,subprocess,shlex,time,os
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).resolve().parent;b=r/'candidates/GLM-M15-oproj-Model-r125';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((inc/'manifest-r125.json').read_bytes());p=json.loads((inc/'parent-binding-r125.json').read_bytes());assert not b.exists();b.mkdir();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'GPU_TUs_compiled':0,'commands':[]};start=time.monotonic()
def run(a,name):
 q['commands'].append(a);z=subprocess.run(a,text=True,capture_output=True,timeout=180);(b/(name+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(name,z.stderr[-4000:]);return z.stdout
def deps(text):return set(shlex.split(text.replace('\\\n',' ').split(':',1)[1]))
try:
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['actual_binary_SHA256'].items():assert sha(n)==h,n
 assert sha(p['parent_native_path'])==p['parent_native_SHA256']=='c6506898f7d718e7257253b51bf720ff8fc84ed733d9b7306edd989308d37514'
 for n,h in p['parent_objects_SHA256'].items():assert sha(n)==h,n
 assert sha(p['bridge_record']['source'])==p['bridge_record']['source_SHA256'];assert sha(p['runtime_record']['source'])==p['runtime_record']['source_SHA256']
 for n,h in p['bridge_record'].get('policy_dependencies_SHA256',{}).items():assert sha(n)==h,n
 flags=['g++','-std=c++17','-O2','-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-pthread','-I'+str(inc)];exe=b/'geometryCPU';run(flags+[str(inc/'geometry_contract_r125.cpp'),'-o',str(exe)],'geometrycompile');contract=json.loads(run([str(exe)],'geometryrun'));assert contract['passed']and contract['valid_rank_offsets']==8
 compiled=[];hashes={}
 for name,oldcmd,oldobj in [('modern_glm_bridge_oproj_r125.cpp',p['bridge_compile_command'],'/refactor/candidates/glm-M16-r8861/private-3.o'),('glm_fp8_runtime_oproj_r125.cpp',p['runtime_compile_command'],'/refactor/candidates/glm-transient-layer-r5681/private-15.o'),('glm_fp8_cache_m15_r119.cpp',flags,None)]:
  prefix=oldcmd[:oldcmd.index('-MD')]if '-MD'in oldcmd else oldcmd;prefix=list(prefix);prefix.insert(1,'-I'+str(inc));src=inc/name;obj=b/(name+'.o');md=b/(name+'.d');run(prefix+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],name+'.compile');fresh=run(prefix+['-M',str(src)],name+'.fresh-M');assert deps(fresh)==deps(md.read_text())
  for path in deps(fresh):hashes[path]=sha(path)
  compiled.append(dict(source=str(src),source_SHA256=sha(src),object=str(obj),object_SHA256=sha(obj),replaces=oldobj,MD_SHA256=sha(md),fresh_M_equal_MD=True))
 old=p['link_command'];link=list(old);replace={c['replaces']:c['object']for c in compiled if c['replaces']};assert all(link.count(k)==1 for k in replace);link=[replace.get(x,x)for x in link];link.insert(link.index('-o'),compiled[2]['object']);lib=b/'libGLM_M15_oproj_Model_r125.so';link[link.index('-o')+1]=str(lib);run(link,'link');oldexports=run(['nm','-D','--defined-only',p['parent_native_path']],'parentexports');exports=run(['nm','-D','--defined-only',str(lib)],'exports');names=lambda s:{x.split()[-1]for x in s.splitlines()if x.split()};assert names(oldexports)<=names(exports);assert {'modern_glm_model_m15_oproj_policy_r125','modern_glm_m15_oproj_stats_r125','glm_fp8_cache_create_r119'}<=names(exports)
 dynamic=run(['readelf','-d',str(lib)],'dynamic');ldd=run(['ldd',str(lib)],'ldd');assert 'not found'not in ldd
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,dependency_SHA256=hashes,HOST_TUs_compiled=3,CPU_geometry_contract=contract,parent_native_SHA256=p['parent_native_SHA256'],parent_objects_SHA256=p['parent_objects_SHA256'],reused_objects_SHA256={n:h for n,h in p['parent_objects_SHA256'].items()if n not in replace},actual_binary_SHA256=m['actual_binary_SHA256'],all_parent_public_symbols_retained=True,ABI=1,Context_layout_changed=False,default_OFF=True,admission='M15N6144K2048SR48SC128ro0co=rank*2048only',stats_count=17,private_cache_source_reused_from_qualified758=True,ldd=ldd,ELF_dynamic=dynamic,manifest_SHA256=sha(inc/'manifest-r125.json'),normal_CPU_exit=True,Model_GPU_qualified=False)
except BaseException as e:q['error']=repr(e)
q['seconds']=time.monotonic()-start;q['producer']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
