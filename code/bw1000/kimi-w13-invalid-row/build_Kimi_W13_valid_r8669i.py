import pathlib,json,hashlib,subprocess,os,shlex,shutil
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-W13-valid-second-r8669i';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();out={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
def run(cmd,label):
 out['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:z=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT,timeout=220)
 assert z.returncode==0,(label,(b/(label+'.log')).read_text()[-5000:])
try:
 base=r/'candidates/kimi-joint-expert-5f91ad7-r4620';ops=r/'candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps';sdk=r/'install/hygon-kimi-5f91ad7-r3672'
 assert sha(r/'candidates/kimi-packed-reuse-r6136/packed.o')=='355607f22e0232b6422cef6f81224af24f62b585e5e251fbb656e7eb0da4700c'
 src=r/'incoming/kimi_W13_valid_second_r8669i.cu';proof=json.loads((r/'incoming/Kimi-W13-valid-source-r8669i.json').read_text());assert sha(src)==proof['source_SHA256'][src.name];local=b/src.name;shutil.copyfile(src,local)
 for n in ['kimi_grouped_W2_r4620.cuh','kimi_mxfp4_static_r326.cuh','kimi_tiled_w13_r1858.cuh','kimi_reduce_wave_r1044.cuh']:shutil.copyfile(base/n,b/n)
 cc='/opt/dtk/cuda/cuda/bin/nvcc';helptext=subprocess.check_output([cc,'--help'],text=True,stderr=subprocess.STDOUT)
 # Use supported compiler retention only; do not speculate on a new toolchain.
 keep=['--keep']if '--keep' in helptext else ['-save-temps']if 'save-temps' in helptext else []
 cmd=[cc,'-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(b),'-I'+str(ops/'src'),'-I'+str(ops/'generated'),'-I'+str(sdk/'include'),*keep,'-MMD','-MF',str(b/'wave.d'),'-c',str(local),'-o',str(b/'wave.o')];run(cmd,'compile')
 run(['g++','-shared','-fPIC',str(b/'wave.o'),'-Wl,-Bsymbolic','/refactor/candidates/kimi-packed-reuse-r6136/packed.o','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-lcuda','-o',str(b/'libKimi_W13_valid_r8669i.so')],'link')
 deps=sorted(set(shlex.split((b/'wave.d').read_text().replace('\\\n',' ').split(':',1)[1])));out['MD']={p:sha(p)for p in deps if pathlib.Path(p).is_file()}
 out['vendor_header_prototypes']={p:pathlib.Path(p).read_text(errors='replace')for p in deps if pathlib.Path(p).is_file()and pathlib.Path(p).name in ['cublas_api.h','cublas_v2.h']}
 files=[p for p in b.rglob('*')if p.is_file()];out['retained_files']=[{'file':p.name,'bytes':p.stat().st_size,'SHA256':sha(p)}for p in files]
 out['assembly_files']={p.name:p.read_text(errors='replace')for p in files if p.suffix in ['.s','.ll'] and p.stat().st_size<12000000}
 undefined=subprocess.check_output(['nm','-D','-C','--undefined-only',str(b/'libKimi_W13_valid_r8669i.so')],text=True)
 assert not any(n in undefined for n in ['packRoutes4531','buildJobs4531','staticW2R326U','modern_kimi_grouped_r4620']),undefined
 out['undefined']=undefined
 out.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],library=str(b/'libKimi_W13_valid_r8669i.so'),library_SHA256=sha(b/'libKimi_W13_valid_r8669i.so'),source_SHA256=sha(local),retention_flags=keep,defined=subprocess.check_output(['nm','-D','-C','--defined-only',str(b/'libKimi_W13_valid_r8669i.so')],text=True),needed=subprocess.check_output(['readelf','-d',str(b/'libKimi_W13_valid_r8669i.so')],text=True))
except BaseException as e:out['error']=repr(e);raise
finally:
 p=r/'results/Kimi-W13-wave-CPU-r8669i.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({k:out.get(k)for k in ['completed','step','error','library','library_SHA256','retention_flags']}))
