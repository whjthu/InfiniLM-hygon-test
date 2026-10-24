import pathlib,json,hashlib,subprocess,shutil,os,shlex
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-caller-sampler-r8669q';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d={'CPU_only':True,'GPU':False,'dlopen':False,'completed':False}
try:
 old=json.loads((r/'results/Kimi-full-MoE-caller-CPU-r8669p.json').read_text());assert old['completed'];src=b/'kimi_HIP_caller_sample_r8669q.cpp';shutil.copyfile(r/'incoming'/src.name,src);lib=b/'libKimi_HIP_caller_sample_r8669q.so';cmd=[str(src)if p.endswith('kimi_HIP_caller_sample_r8669o.cpp')else str(b/'sampler.d')if p.endswith('/sampler.d')else str(lib)if p==old['sampler']else p for p in old['commands'][1]];d['command']=cmd;z=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert z.returncode==0,z.stderr[-8000:]
 deps=shlex.split((b/'sampler.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]);vendor=pathlib.Path('/opt/dtk/hip/lib/libgalaxyhip.so.5')
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],sampler=str(lib),sampler_SHA256=sha(lib),source_SHA256=sha(src),adapter=old['library'],adapter_SHA256=old['library_SHA256'],fresh_MD={p:sha(pathlib.Path(p))for p in deps if pathlib.Path(p).is_file()},vendor=str(vendor),vendor_SHA256=sha(vendor),needed=subprocess.check_output(['readelf','-d',str(lib)],text=True))
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-caller-sampler-CPU-r8669q.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','sampler','sampler_SHA256']}))
