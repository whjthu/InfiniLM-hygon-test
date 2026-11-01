"""Build only the private T1 adapter/kernel against exact afc6. No dlopen/GPU."""
import pathlib,json,hashlib,subprocess,os,time,shlex,traceback
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent
B=R/'candidates/Deep-HC-epilogue-r358-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,weights_or_Model_loaded=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
started=time.monotonic()
def run(args,label,timeout=180):
 q['commands'].append(args);z=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr)
 assert z.returncode==0,(label,z.returncode,z.stderr[-6000:]);return z.stdout
try:
 receipt=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(receipt)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5'
 parent=json.loads(receipt.read_bytes())['families']['Deep']['Native'];native=pathlib.Path(parent['new']);assert sha(native)=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 abi=['-std=c++17','-O2','-fPIC','-pthread','-D_GLIBCXX_USE_CXX11_ABI=1']
 includes=[S,R/'candidates/deep-c4-projection-model-r5920/incoming',R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/csrc',R/'candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM/csrc',R/'install/hygon-deep-5f91ad7-r3672/include',R/'install/hygon-deep-5f91ad7-r3672/include/infiniccl',R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/third_party/json/include',R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/third_party/spdlog/include',pathlib.Path('/opt/dtk/cuda/cuda/include')]
 flags=abi+['-I'+str(p)for p in includes];obj=B/'adapter.o';md=B/'adapter.d';src=S/'adapter.cpp'
 run(['g++',*flags,'-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],'adapter')
 parse=lambda text:set(shlex.split(text.replace('\\\n',' ').split(':',1)[1]));deps=parse(md.read_text());fresh=parse(run(['g++',*flags,'-M',str(src)],'fresh-M'));assert deps==fresh
 kernel=B/'epilogue.o';run(['hipcc','-O3','-fPIC','--offload-arch=gfx936','-ffp-contract=off','-c',str(S/'hc_epilogue_r358.hip'),'-o',str(kernel)],'HIP')
 libraries=[v for v in parent['link_argv']if v.startswith(('-L','-l','-Wl,'))and v!='-Wl,-Bsymbolic'];lib=B/'libDeep_HC_epilogue_r358.so'
 run(['g++',str(obj),str(kernel),str(native),'-shared','-fPIC',*libraries,'-L/opt/dtk/lib','-lamdhip64','-Wl,--no-undefined','-Wl,-rpath,'+str(native.parent),'-o',str(lib)],'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');required=['deep_hc_epilogue_'+n+'_r358'for n in ['create','forward','destroy','error','report','hit']];assert all(any(l.split()[-1]==n for l in exports.splitlines())for n in required)
 ld=run(['ldd',str(lib)],'ldd');assert 'not found'not in ld
 q.update(completed=True,source_SHA256={p.name:sha(p)for p in S.iterdir()if p.is_file()},Native_path=str(native),Native_SHA256=sha(native),library=str(lib),library_SHA256=sha(lib),object_SHA256={str(p):sha(p)for p in [obj,kernel]},MD_equals_fresh_M=True,dependencies_SHA256={p:sha(p)for p in deps},ldd=ld,exports=required,seconds=time.monotonic()-started,GPU_numeric='NOT_RUN',Model_gain='UNMEASURED')
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_step','error','seconds']}));assert q['completed']
