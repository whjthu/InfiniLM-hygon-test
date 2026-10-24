import hashlib,json,pathlib,shutil,subprocess,time,os
r=pathlib.Path('/refactor');b=r/'candidates/glm-prefill-window-r8701';assert not b.exists();(b/'src').mkdir(parents=True);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((r/'incoming/GLM-prefill-window-CPU-r8701-manifest.json').read_text())
for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
parent=r/'candidates/glm-shared-validation-model-r5817/CPU-build.json';p=json.loads(parent.read_text());assert p['completed']and sha(p['library'])==p['library_SHA256']=='1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d'
old=r/'candidates/glm-window-model-r5797/CPU-build.json';a=json.loads(old.read_text());assert sha(old)=='611d400db8802e6ac38931db0c222527d45adaa1f56d17f0c34a45a81fc31624'
commands=[];report={'completed':False,'CPU_only':True,'dlopen':False,'GPU_initialized':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'parent_DSO_SHA256':sha(p['library']),'parent_build_SHA256':sha(parent)};start=time.monotonic()
def run(cmd,name):
 commands.append(cmd);t=subprocess.run(cmd,text=True,capture_output=True);(b/(name+'.log')).write_text(t.stdout+t.stderr)
 if t.returncode:raise RuntimeError((name,t.returncode,t.stderr[-6000:]))
 return t.stdout
try:
 link=next(list(v)for v in p['commands']if '-shared'in v);objects={v:sha(v)for v in link if v.endswith('.o')};assert len(objects)==28
 replacements={};compiled=[]
 for name,local in [('glm_moe_dsa_balanced_text_model_r126.cpp','glm_prefill_window_model_r8701.cpp'),('glm_model_window_r5797.cpp','glm_prefill_window_observer_r8701.cpp')]:
  rec=next(v for v in a['compiled']if pathlib.Path(v['source']).name==name);assert rec['object']in objects and sha(rec['source'])==rec['source_SHA256']and sha(rec['object'])==rec['object_SHA256']
  oldcmd=next(v for v in a['commands']if '-c'in v and rec['source']in v);src=b/'src'/name;shutil.copyfile(r/'incoming'/local,src);obj=b/(name+'.o');dep=b/(name+'.d');fl=[];i=1
  while i<len(oldcmd):
   if oldcmd[i]in['-c','-o','-MF','-MT','-MQ']:i+=2;continue
   if oldcmd[i]in['-MD','-MMD']:i+=1;continue
   fl.append(oldcmd[i]);i+=1
  run([oldcmd[0],*fl,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],name);replacements[rec['object']]=str(obj);compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':rec['object'],'base_source_SHA256':rec['source_SHA256']})
 link=[replacements.get(v,v)for v in link];lib=b/'libglm_prefill_window_r8701.so';link[link.index('-o')+1]=str(lib);run(link,'link');exports=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(v in exports for v in['modern_glm_window_arm_r5795','modern_glm_window_read_r5795','modern_glm_window_prepare_r5797'])
 for n,h in objects.items():assert sha(n)==h
 report.update(completed=True,compiled=compiled,library=str(lib),library_SHA256=sha(lib),reused_objects_SHA256={n:h for n,h in objects.items()if n not in replacements},scope='2TU only/noheaderABI/math/SDK changes; no dlopen/GPU/weights. Eventqueue in originalstream only; existing original CAPI drain/query. Compile is not scopehit proof; Model owner metadata required.')
except BaseException as e:report['error']=repr(e);raise
finally:report.update(seconds=time.monotonic()-start,commands=commands);(b/'CPU-build.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
