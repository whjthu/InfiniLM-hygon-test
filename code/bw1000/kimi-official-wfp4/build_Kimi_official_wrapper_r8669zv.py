import pathlib,json,hashlib,subprocess,os,shlex
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-official-WFP4-wrapper-r8669zv';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
try:
 p=r/'results/Kimi-official-WFP4-CPU-r8669zu.json';old=json.loads(p.read_text());assert not old['completed']and old['commands'][-1][0]=='patchelf';assert old['error'].startswith('FileNotFoundError')
 proof=json.loads((r/'incoming/Kimi-official-WFP4-source-r8669zu.json').read_text())
 for n,h in proof['source_SHA256'].items():assert sha(r/'candidates/Kimi-official-WFP4-r8669zu/src'/n)==h
 link=old['commands'][2];native=pathlib.Path(link[link.index('-o')+1]);assert native.exists();parent=json.loads((r/'results/Kimi-I192-smoke-static-closure-r8669zg.json').read_text());defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(n in defs for n in ['modern_kimi_official_WFP4_r8669zu','modern_kimi_official_counts_r8669zu'])
 und=lambda n:set(subprocess.check_output(['nm','-D','--undefined-only',str(n)],text=True).splitlines());assert not und(native)-und(parent['native'])
 y=json.loads((r/'results/Kimi-I192-CPU-r8669y.json').read_text());cmd=next(list(c)for c in y['commands']if any(v.endswith('/kimi_TP16_full_MoE_I192_r8669y.cpp')for v in c));assert cmd.count(y['native'])==1;cmd[cmd.index(y['native'])]=str(native);wrapper=b/'libKimi_fullMoE_official_WFP4_r8669zv.so';cmd[cmd.index('-o')+1]=str(wrapper);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');d['commands'].append(cmd)
 with(b/'adapter.log').open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(b/'adapter.log').read_text()[-9000:]
 needed=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert str(native)in needed and y['native']not in needed
 paths=shlex.split((b/'adapter.d').read_text().replace('\\\n',' ').split(':',1)[1]);d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),library=str(wrapper),library_SHA256=sha(wrapper),parent227_raw_SHA256=sha(p),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256={v:sha(v)for v in link if v.endswith('.o')},fresh_MD={v:sha(v)for v in paths if pathlib.Path(v).is_file()},wrapper_needed=needed,math_recompiled=False,wrapper_source_unchanged=True,new_undefined=[])
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-official-WFP4-wrapper-CPU-r8669zv.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256','library','library_SHA256']}))
