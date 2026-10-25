import pathlib,json,hashlib,subprocess,shlex,os,shutil
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-official-WFP4-pair-r8669zx';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]}
try:
 p=r/'results/Kimi-official-WFP4-wrapper-CPU-r8669zv.json';old=json.loads(p.read_text());assert old['completed']and sha(old['native'])==old['native_SHA256']=='6b669a1a5f3e58caaaf78d441931e7098f56bb94cb80e558911fa5b402544d1d'
 proof=json.loads((r/'incoming/Kimi-official-pair-source-r8669zx.json').read_text());src=b/'src';src.mkdir()
 for n,h in proof['source_SHA256'].items():assert sha(r/'incoming'/n)==h;shutil.copyfile(r/'incoming'/n,src/n)
 cmd=list(old['commands'][0]);source=next(x for x in cmd if x.endswith('.cpp'));cmd[cmd.index(source)]=str(src/'kimi_TP16_official_WFP4_r8669zx.cpp');wrapper=b/'libKimi_TP16_official_pair_r8669zx.so';cmd[cmd.index('-o')+1]=str(wrapper);cmd[cmd.index('-MF')+1]=str(b/'adapter.d');d['commands'].append(cmd)
 with(b/'adapter.log').open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=190)
 assert z.returncode==0,(b/'adapter.log').read_text()[-9000:]
 needed=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert old['native']in needed;defs=subprocess.check_output(['nm','-D','--defined-only',str(wrapper)],text=True);assert 'modern_kimi_TP16_moe_r8669zx'in defs
 paths=shlex.split((b/'adapter.d').read_text().replace('\\\n',' ').split(':',1)[1]);d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=old['native'],native_SHA256=old['native_SHA256'],library=str(wrapper),library_SHA256=sha(wrapper),source_SHA256=proof['source_SHA256'],all_linked_object_SHA256=old['all_linked_object_SHA256'],fresh_MD={v:sha(v)for v in paths if pathlib.Path(v).is_file()},wrapper_needed=needed,math_recompiled=False,source_TUs=['provider','MoE_host','paired_thin_adapter'],parent228_raw_SHA256=sha(p),scope='Original teacher0/currentI192 TP16 vs official mode114 W13 only, two distinct GPUcontexts/original then repacked CPUownedweightshards; 2warm+8ABBA, noGPUrun')
except BaseException as e:d['error']=repr(e);raise
finally:
 p=r/'results/Kimi-official-WFP4-pair-CPU-r8669zx.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native','native_SHA256','library','library_SHA256']}))
