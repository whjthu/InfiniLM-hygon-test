import pathlib,json,hashlib,subprocess,os,traceback,time
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-state-only-replay-r9113';b.mkdir(exist_ok=False);(b/'src').mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();t=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'changed_HOST_TUs':2,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 parent=json.loads((r/'results/Kimi-CCL512-CPU-r8606.json').read_text());cmd=parent['commands'][0][:];src=b/'src/delta.cpp';src.write_bytes((r/'incoming/Kimi_state_only_delta_r9113.cpp').read_bytes());(src.parent/'state_only_r9113.hpp').write_bytes((r/'incoming/Kimi_state_only_r9113.hpp').read_bytes());helper=r/'candidates/Kimi-CCL512-r8606/src/kimi_spec_math_r3676.hpp';(src.parent/helper.name).write_bytes(helper.read_bytes());cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(b/'delta.o');cmd[cmd.index('-MF')+1]=str(b/'delta.d')
 trace=json.loads((r/'candidates/Kimi-target-taps-r9022/CPU-build.json').read_text());link=trace['commands'][-1][:];old='/refactor/candidates/Kimi-CCL512-r8606/delta.o';assert link.count(old)==1;link[link.index(old)]=str(b/'delta.o');main=b/'libKimi_state_only_Main_r9113.so';link[link.index('-o')+1]=str(main)
 for j,a in enumerate(link):
  if a.startswith('-Wl,-Map,')or a.startswith('-Wl,-Map='):link[j]='-Wl,-Map,'+str(b/'link.map')
 oldshim=json.loads((r/'results/Kimi-layer0-link-CPU-r9038e-910407.272.json').read_text());shim=oldshim['command'][:];native=b/'libKimi_state_only_shim_r9113.so';bridge=b/'src/layer0_native.cpp';bridge.write_bytes((r/'incoming/Kimi_state_only_bridge_r9113.cpp').read_bytes());shim[shim.index('-shared')+1]=str(bridge);shim[shim.index(oldshim['actual678_DSO'])]=str(main);shim[shim.index('-o')+1]=str(native);q['commands']=[cmd,link,shim]
 for j,c in enumerate(q['commands']):
  p=subprocess.run(c,capture_output=True,text=True,timeout=140);(b/f'command{j}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-8000:]
 needed=[x.strip()for x in subprocess.check_output(['readelf','-d',str(native)],text=True).splitlines()if'NEEDED'in x];assert str(main)in needed[0]
 nm=subprocess.check_output(['nm','-D','--defined-only',str(main)],text=True);assert 'kimi_state_only_r9113'in nm
 q.update(completed=True,native=str(native),native_SHA256=sha(native),Main=str(main),Main_SHA256=sha(main),source_SHA256={p.name:sha(p)for p in(src,bridge,src.parent/'state_only_r9113.hpp')},provider_DSO=oldshim['provider_DSO'],provider_DSO_SHA256=oldshim['provider_DSO_SHA256'],DT_NEEDED=needed,elapsed_seconds=time.monotonic()-t)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
p=r/'results'/('Kimi-state-only-CPU-r9113-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
