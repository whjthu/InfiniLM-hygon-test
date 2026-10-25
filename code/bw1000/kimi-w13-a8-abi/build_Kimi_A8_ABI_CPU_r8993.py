import pathlib,json,hashlib,shutil,subprocess,shlex,os,time
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-W13-A8-ABI-r8993';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();started=time.monotonic()
d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[],'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(cmd,name):
 d['commands'].append(cmd)
 with(b/(name+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,155-(time.monotonic()-started)))
 assert z.returncode==0,(name,(b/(name+'.log')).read_text()[-8000:])
try:
 b.mkdir(exist_ok=False);(b/'src').mkdir();proof=json.loads((r/'incoming/Kimi-A8-ABI-source-r8993.json').read_text())
 for name in ['kimi_A8_matrix_r8993.hip','kimi_A8_adapter_r8993.cu']:
  src=r/'incoming'/name;assert sha(src)==proof['source_SHA256'][name];shutil.copyfile(src,b/'src'/name)
 y=json.loads((r/'results/Kimi-I192-CPU-r8669y.json').read_text());assert y['completed']
 oldobj=next(p for p in y['objects']if p.endswith('/provider.o'));assert sha(oldobj)==y['objects'][oldobj]
 hardware=json.loads((r/'candidates/hardware-dumma-r8403/CPU-build.json').read_text());assert hardware['completed']and hardware['CPU_only']and not hardware['GPU_initialized']
 for name,h in hardware['header_SHA256'].items():assert sha(pathlib.Path('/opt/dtk/include')/name)==h,name
 run(['/opt/dtk/bin/hipcc','-std=c++17','-O3','--offload-arch=gfx936','-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-MD','-MF',str(b/'matrix.d'),'-c',str(b/'src/kimi_A8_matrix_r8993.hip'),'-o',str(b/'matrix.o')],'pure-HIP-matrix')
 cmd=list(y['commands'][0]);cmd[cmd.index('-c')+1]=str(b/'src/kimi_A8_adapter_r8993.cu');cmd[cmd.index('-o')+1]=str(b/'adapter.o');cmd[cmd.index('-MF')+1]=str(b/'adapter.d');run(cmd,'CUDA-adapter')
 native=b/'libKimi_A8_ABI_r8993.so'
 run(['/opt/dtk/bin/hipcc','-shared','--offload-arch=gfx936','-fPIC',str(b/'adapter.o'),str(b/'matrix.o'),oldobj,'-L/opt/dtk/cuda/cuda/lib64','-lcudart','-o',str(native)],'link')
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True)
 for n in ['modern_kimi_A8_W13_r8993','kimi_A8_HIP_matrix_r8993','modern_kimi_grouped_I192_r8669y','staticW2I192R8669y','packRoutesI192R8669y']:assert n in defs,n
 fresh={}
 for unit in ['matrix','adapter']:
  deps=shlex.split((b/(unit+'.d')).read_text().replace('\\\n',' ').split(':',1)[1]);fresh[unit]={p:sha(p)for p in sorted(set(deps))if pathlib.Path(p).is_file()}
 assert any('du_mma'in p for p in fresh['matrix'])
 assert not any('/cuda/'in p or '/InfiniOps/'in p or '/install/'in p for p in fresh['matrix'])
 assert not any('du_mma'in p or '/hip/'in p for p in fresh['adapter'])
 d.update(completed=True,native=str(native),native_SHA256=sha(native),source_SHA256=proof['source_SHA256'],objects={str(b/(n+'.o')):sha(b/(n+'.o'))for n in ['matrix','adapter']},unchanged_provider_object={oldobj:sha(oldobj)},hardware702_header_SHA256=hardware['header_SHA256'],fresh_MD=fresh,defined=defs,undefined=subprocess.check_output(['nm','-D','--undefined-only',str(native)],text=True),needed=subprocess.check_output(['readelf','-d',str(native)],text=True),new_GPU_TUs=1,new_host_TUs=1,stream_scope='default stream0 only; CUDA/HIP runtime ordering UNPROVEN before finite GPU qualification',elapsed_s=time.monotonic()-started)
except BaseException as e:d['error']=repr(e);raise
finally:
 d['elapsed_s']=time.monotonic()-started
 p=r/'results/Kimi-A8-ABI-CPU-r8993.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256','elapsed_s']}))
