import pathlib,json,hashlib,subprocess,os,traceback,time
r=pathlib.Path('/refactor');src=r/'incoming/Kimi-Delta-batch-profile-r9140';b=r/'candidates/Kimi-Delta-batch-profile-r9140';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 pure=['g++','-std=c++17','-O2','-I'+str(src),str(src/'CPU_contract.cpp'),'-o',str(b/'CPU_contract')];z=subprocess.run(pure,capture_output=True,text=True,timeout=30);assert z.returncode==0,z.stderr;z=subprocess.run([str(b/'CPU_contract')],capture_output=True,text=True,timeout=10);assert z.returncode==0,z.stderr;q['pure_contract']=json.loads(z.stdout)
 parent=json.loads((r/'results/Kimi-CCL512-CPU-r8606.json').read_text());cmd=parent['commands'][0][:];(src/'kimi_spec_math_r3676.hpp').write_bytes((r/'candidates/Kimi-CCL512-r8606/src/kimi_spec_math_r3676.hpp').read_bytes());cmd[1:1]=['-I'+str(src)];cmd[cmd.index('-c')+1]=str(src/'delta.cpp');cmd[cmd.index('-o')+1]=str(b/'delta.o');cmd[cmd.index('-MF')+1]=str(b/'delta.d')
 trace=json.loads((r/'candidates/Kimi-target-taps-r9022/CPU-build.json').read_text());link=trace['commands'][-1][:];old='/refactor/candidates/Kimi-CCL512-r8606/delta.o';assert link.count(old)==1;link[link.index(old)]=str(b/'delta.o');main=b/'libKimi_batch_Main_r9122.so';link[link.index('-o')+1]=str(main)
 for j,a in enumerate(link):
  if a.startswith('-Wl,-Map,')or a.startswith('-Wl,-Map='):link[j]='-Wl,-Map,'+str(b/'link.map')
 oldshim=json.loads((r/'results/Kimi-layer0-link-CPU-r9038e-910407.272.json').read_text());shim=oldshim['command'][:];shim[1:1]=['-I'+str(src)];native=b/'libKimi_batch_shim_r9122.so';shim[shim.index('-shared')+1]=str(src/'layer0_native.cpp');shim[shim.index(oldshim['actual678_DSO'])]=str(main);shim[shim.index('-o')+1]=str(native);q['commands']=[pure,cmd,link,shim]
 for j,c in enumerate([cmd,link,shim]):
  z=subprocess.run(c,capture_output=True,text=True,timeout=140);(b/f'command{j}.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr[-10000:]
 needed=[x.strip()for x in subprocess.check_output(['readelf','-d',str(native)],text=True).splitlines()if'NEEDED'in x];assert str(main)in needed[0];nm=subprocess.check_output(['nm','-D','--defined-only',str(main)],text=True);assert 'kimi_batch_r9122'in nm
 ldd=subprocess.check_output(['ldd',str(native)],text=True);assert 'not found'not in ldd;q['ldd']=ldd;q['Torch_GPU_uninitialized']=not __import__('torch').cuda.is_initialized();assert q['Torch_GPU_uninitialized']
 q.update(completed=True,native=str(native),native_SHA256=sha(native),Main=str(main),Main_SHA256=sha(main),source_SHA256={p.name:sha(p)for p in src.iterdir()if p.is_file()},provider_DSO=oldshim['provider_DSO'],provider_DSO_SHA256=oldshim['provider_DSO_SHA256'],DT_NEEDED=needed,elapsed_seconds=time.monotonic()-start)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
