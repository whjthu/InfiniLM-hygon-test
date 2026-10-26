import pathlib,json,hashlib,subprocess,time,os,shlex
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).resolve().parent;b=r/'candidates/GLM-M15-component-r121'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((inc/'manifest-r121.json').read_bytes());assert not b.exists();b.mkdir();q={'completed':False,'CPU_only':True,'GPU_initialized':False,'dlopen':False,'GPU_TUs_compiled':0,'commands':[]};start=time.monotonic()
def run(a,name):
 q['commands'].append(a);z=subprocess.run(a,text=True,capture_output=True,timeout=150);(b/(name+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(name,z.stderr[-5000:]);return z.stdout
try:
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['actual_binding_SHA256'].items():assert sha(r/n)==h,n
 flags=['-std=c++17','-O2','-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-pthread','-I'+str(inc),'-I'+str(r/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/csrc'),'-I'+str(r/'install/hygon-glm-5f91ad7-r3672/include'),'-I'+str(r/'install/hygon-glm-5f91ad7-r3672/include/infiniccl'),'-I'+str(r/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/third_party/json/include'),'-I'+str(r/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/third_party/spdlog/include'),'-I/opt/dtk/cuda/cuda/include']
 exe=b/'CPUcachecontract';run(['g++',*flags,str(inc/'glm_fp8_cache_m15_r119.cpp'),str(inc/'cache_mock_contract_r121.cpp'),'-o',str(exe)],'mockbuild');mock=json.loads(run([str(exe)],'mockrun'));assert mock['passed']and mock['CPU_mock_only']and not mock['GPU_math_proof'];q['CPU_mock_contract']=mock
 compiled=[];deps={}
 for name in ['glm_fp8_cache_m15_r119.cpp','glm_m15_component_r121.cpp']:
  src=inc/name;obj=b/(name+'.o');md=b/(name+'.d');run(['g++',*flags,'-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],name+'.compile')
  for p in shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]):deps[p]=sha(p)
  compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'MD_SHA256':sha(md)})
 common=r/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM';sdk=r/'install/hygon-glm-5f91ad7-r3672';helper=r/'candidates/glm-grouped-native-model-r5117/helpers';lib=b/'libGLM_M15_component_r121.so'
 args=['g++','-shared',*[c['object']for c in compiled],'-L'+str(helper),'-lglm_fp8_batched_r776','-Wl,-rpath,'+str(helper),'-L'+str(common/'python/infinilm/_lib'),'-linfinilm_runtime','-Wl,-rpath,'+str(common/'python/infinilm/_lib'),'-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-Wl,-rpath,'+str(sdk/'lib'),'-L/opt/dtk/cuda/cuda/lib64','-lcudart','-pthread','-Wl,-Bsymbolic','-Wl,--no-undefined','-o',str(lib)]
 run(args,'link');exports=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(n in exports for n in ['glm_m15_component_create_r121','glm_m15_component_forward_r121','glm_m15_component_destroy_r121','glm_fp8_cache_create_r119']);dynamic=run(['readelf','-d',str(lib)],'dynamic');assert 'libglm_fp8_cache_r807'not in dynamic;ldd=run(['ldd',str(lib)],'ldd');assert 'not found'not in ldd
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,dependency_SHA256=deps,HOST_TUs_compiled=2,actual_binding_SHA256=m['actual_binding_SHA256'],manifest_SHA256=sha(inc/'manifest-r121.json'),ELF_dynamic=dynamic,ldd=ldd,ABI=1,normal_CPU_exit=True)
except BaseException as e:q['error']=repr(e)
q['seconds']=time.monotonic()-start;q['producer']=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
