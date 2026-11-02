"""Build only the private T1 adapter/kernel against exact afc6. No dlopen/GPU."""
import pathlib,json,hashlib,subprocess,os,time,shlex,traceback
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent
B=R/'candidates/Deep-HC-full-pre-fusion-r372-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,weights_payload_opened=False,whole_Model_loaded=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
started=time.monotonic()
def run(args,label,timeout=180):
 q['commands'].append(args);z=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr)
 assert z.returncode==0,(label,z.returncode,z.stderr[-6000:]);return z.stdout
try:
 receipt=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/CPU-build.json';assert sha(receipt)=='e3012f3c27c58f3c2c18f237906cd0120a4232620873cbea9b90e5012fd61152'
 current=json.loads(receipt.read_bytes());assert current['passed']and len(current['all_objects_SHA256'])==49
 for p,h in current['all_objects_SHA256'].items():assert sha(p)==h
 parent=json.loads((R/'candidates/latest-family-native-r269/CPU-build.json').read_bytes())['families']['Deep']['Native'];native=pathlib.Path(current['library']);assert sha(native)=='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
 abi=['-std=c++17','-O2','-fPIC','-pthread','-D_GLIBCXX_USE_CXX11_ABI=1']
 includes=[S,R/'candidates/deep-combined-r7169/incoming',R/'candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM/csrc',R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/csrc',R/'install/hygon-deep-5f91ad7-r3672/include',R/'install/hygon-deep-5f91ad7-r3672/include/infiniccl',R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/third_party/json/include',R/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/third_party/spdlog/include',pathlib.Path('/opt/dtk/cuda/cuda/include')]
 flags=abi+['-I'+str(p)for p in includes];obj=B/'adapter.o';md=B/'adapter.d';src=S/'adapter.cpp'
 run(['g++',*flags,'-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],'adapter')
 parse=lambda text:set(shlex.split(text.replace('\\\n',' ').split(':',1)[1]));deps=parse(md.read_text());fresh=parse(run(['g++',*flags,'-M',str(src)],'fresh-M'));assert deps==fresh
 assert any(p.endswith('/Model/Deep/InfiniLM/csrc/execution/submit.hpp')for p in deps)
 assert any(p.endswith('/execution/owner_affinity_r2392.hpp')for p in deps)
 kernel=B/'staged.o';run(['hipcc','-O3','-fPIC','--offload-arch=gfx936','-ffp-contract=off','-c',str(S/'hc_staged_r372.hip'),'-o',str(kernel)],'HIP')
 libraries=[v for v in parent['link_argv']if v.startswith(('-L','-l','-Wl,'))and v!='-Wl,-Bsymbolic'];lib=B/'libDeep_HC_staged_r372.so'
 run(['g++',str(obj),str(kernel),str(native),'-shared','-fPIC',*libraries,'-L/opt/dtk/lib','-lamdhip64','-Wl,--no-undefined','-Wl,-rpath,'+str(native.parent),'-o',str(lib)],'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');required=['deep_hc_epilogue_'+n+'_r372'for n in ['create','forward','destroy','error','report','hit','state']];assert all(any(l.split()[-1]==n for l in exports.splitlines())for n in required)
 ld=run(['ldd',str(lib)],'ldd');assert 'not found'not in ld
 leaf=B/'libCPU_math_leaf_r372.so';run(['g++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-shared','-fPIC','-I'+str(S),str(S/'CPU_leaf.cpp'),'-o',str(leaf)],'CPU-leaf')
 helper=R/'candidates/models-5f91ad7-r3675/Deep/helpers/src/deepseek_mhc_sinkhorn_r89.cu';assert sha(helper)=='f254be195ce38d6399c0a55f4b4680c1eba73ac6dfe4bad392adc98d528691d4'
 run(['python3','-B',str(S/'CPU_oracle.py')],'CPU-oracle',180);fp=B/'CPU-oracle.json';fixture=json.loads(fp.read_bytes());assert fixture['passed']and fixture['cases']==172 and fixture['groups']==86 and not fixture['GPU_initialized']and not fixture['Native_dlopen'];q.update(weights_payload_opened=True,small_HC_parameter_tiles_only=True,CPU_fixture_receipt_SHA256=sha(fp),actual_fixture_groups=86,oracle_cases=172,current49_objects_SHA256=current['all_objects_SHA256'],original_Sinkhorn_source_SHA256=sha(helper),current_Native_CPU_receipt_SHA256=sha(receipt),CPU_math_leaf_SHA256=sha(leaf))
 q.update(completed=True,source_SHA256={p.name:sha(p)for p in S.iterdir()if p.is_file()},Native_path=str(native),Native_SHA256=sha(native),library=str(lib),library_SHA256=sha(lib),object_SHA256={str(p):sha(p)for p in [obj,kernel]},MD_equals_fresh_M=True,dependencies_SHA256={p:sha(p)for p in deps},ldd=ld,exports=required,seconds=time.monotonic()-started,GPU_numeric='NOT_RUN',Model_gain='UNMEASURED')
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_step','error','seconds']}));assert q['completed']
