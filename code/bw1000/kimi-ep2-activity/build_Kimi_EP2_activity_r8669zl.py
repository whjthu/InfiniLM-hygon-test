import pathlib,json,hashlib,subprocess,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-EP2-activity-r8669zl';b.mkdir(exist_ok=False);src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();proof=json.loads((r/'incoming/Kimi-EP2-activity-source-r8669zl.json').read_text());name='kimi_TP16_EP2_activity_r8669zl.cpp';assert sha(r/'incoming'/name)==proof['source_SHA256'][name];(src/name).write_bytes((r/'incoming'/name).read_bytes());d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
try:
 old=json.loads((r/'results/Kimi-EP2-bind-CPU-r8669zk.json').read_text());assert old['completed'];cmd=list(old['commands'][0]);oldsource=next(x for x in cmd if x.endswith('/kimi_TP16_EP2_full_MoE_r8669zj.cpp'));cmd[cmd.index(oldsource)]=str(src/name);library=b/'libKimi_EP2_activity_r8669zl.so';cmd[cmd.index('-o')+1]=str(library);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');d['commands'].append(cmd)
 assert sha(old['native'])=='342cc28314e595d28ecd7fdbd13c9d384961ec2b1943645a5438bbab3f86a01e';z=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert z.returncode==0,z.stderr[-8000:]
 md=shlex.split((b/'adapter.d').read_text().replace('\\\n',' ').split(':',1)[1]);headers={p:sha(p)for p in md if pathlib.Path(p).is_file()};assert not any('/incoming/'in p for p in md)
 needed=subprocess.check_output(['readelf','-d',str(library)],text=True);assert old['native']in needed
 defs=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True);assert all(s in defs for s in ['modern_kimi_TP16_moe_r8669zl','modern_kimi_TP16_moe_result_r8669zl'])
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],library=str(library),library_SHA256=sha(library),native=old['native'],native_SHA256=sha(old['native']),source_SHA256=proof['source_SHA256'],fresh_MD=headers,wrapper_needed=needed,scope='OnehostwrapperTU adds originalhipprofStartStop aroundtwo completeMoEforwards; noGPUmath/provider/SDK changes')
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-EP2-activity-CPU-r8669zl.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','library_SHA256']}))
