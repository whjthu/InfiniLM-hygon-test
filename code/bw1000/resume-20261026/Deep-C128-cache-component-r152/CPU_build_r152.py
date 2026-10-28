"""CPU HOST wrappers only; reuse exact experimental F100 dependency closure."""
import pathlib,json,hashlib,ast,os,subprocess,time,traceback,sys
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-C128-cache-component-r152';LM=R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM';SDK=R/'install/hygon-deep-5f91ad7-r3672';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'actual_step':str(os.environ.get('SLURM_JOB_ID',''))+'.'+str(os.environ.get('SLURM_STEP_ID','')),'node':os.uname().nodename,'Torch_imported':False,'GPU_initialized':False,'Native_dlopen':False,'model_constructed':False,'weights_opened':False,'new_GPU_TUs':0,'new_HOST_wrapper_TUs':2,'production_ac937_replaced':False};t=time.monotonic()
try:
 assert os.environ.get('SLURM_JOB_ID','').isdigit()and os.environ.get('SLURM_STEP_ID','').isdigit(),'Actual CPU Slurm step required'
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']),'CPU needs hidden GPUs'
 manifest=D/'manifest-r152.json';m=json.loads(manifest.read_bytes());q['source_manifest_SHA256']=sha(manifest)
 for n,h in m['source_SHA256'].items():
  assert sha(D/n)==h,n
  if n.endswith('.py'):ast.parse((D/n).read_bytes())
 cp=R/'candidates/Deep-prefix-device-scope-r135/CPU-build.json';assert sha(cp)=='4a151a66a5149ad9b423b87c2a74e03e40bcaa3a1094bdb6eddad117b4fe9bf0';c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['GPU_initialized']and not c['dlopen'];assert sha(c['library'])==c['library_SHA256']=='f100e038c6e883e0210380724d7492810cbad1c20384ea4cd162dca4ca2326b2';q['parent_actual_CPU_receipt_SHA256']=sha(cp);q['parent_experimental_F100_SHA256']=sha(c['library'])
 for p,pin in c['actual_DTK_bindings'].items():assert sha(p)==pin['SHA256'],p
 q['actual_DTK_bindings']=c['actual_DTK_bindings'];q['headers_SHA256']={}
 for n in ['execution/submit.hpp','runtime/runtime.hpp','runtime/tensor.hpp']:
  p=LM/'csrc'/n;assert p.is_file(),str(p);q['headers_SHA256'][str(p)]=sha(p)
 for name in ['deepseek_v4_compress.h','deepseek_v4_paged_cache.h']:
  found=list((SDK/'include').rglob(name));assert len(found)==1,(name,[str(p)for p in found]);q['headers_SHA256'][str(found[0])]=sha(found[0])
 lib=B/'libdeep_component_r152.so';args=['g++','-std=c++17','-O2','-shared','-fPIC','-pthread','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(LM/'csrc'),'-I'+str(SDK/'include'),'-I'+str(SDK/'include/infiniccl'),'-I'+str(LM/'third_party/json/include'),'-I'+str(LM/'third_party/spdlog/include'),'-I/opt/dtk/cuda/cuda/include',str(D/'modern_deepseek_public_compress_probe.cpp'),str(D/'modern_deepseek_public_cache_probe.cpp'),'-L'+str(LM/'python'),'-l:_infinilm.cpython-310-x86_64-linux-gnu.so','-L'+str(LM/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(SDK/'lib'),'-linfiniops','-linfiniccl','-linfinirt','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-Wl,--no-undefined','-o',str(lib)];q['compile_args']=args;z=subprocess.run(args,capture_output=True,text=True,timeout=150);q['compiler']={'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr};assert z.returncode==0,'CPU compile failed'
 z=subprocess.run(['ldd','-r',str(lib)],capture_output=True,text=True,timeout=60);q['ldd_r']={'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr};assert z.returncode==0 and'not found'not in z.stdout+z.stderr and'undefined symbol:'not in z.stdout+z.stderr,'CPU dynamic closure failed'
 names=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True,timeout=20);required=['modern_deep_compress','modern_deep_compress_error','modern_deep_compress_shutdown','modern_deep_public_cache','modern_deep_public_cache_error','modern_deep_public_cache_shutdown'];assert all(n in{line.split()[-1]for line in names.splitlines()}for n in required)
 assert 'torch'not in sys.modules;q.update(completed=True,library=str(lib),library_SHA256=sha(lib),required_exports=required)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-t;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','error','library_SHA256','elapsed_s']}));assert q['completed']
