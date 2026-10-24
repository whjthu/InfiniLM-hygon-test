import pathlib,json,hashlib,subprocess,time,os,shutil
r=pathlib.Path('/refactor');b=r/'candidates/glm-decode-window-r8812';assert not b.exists();(b/'src').mkdir(parents=True);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((r/'incoming/GLM-decode-window-CPU-manifest-r8812.json').read_text())
for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h,n
parent=r/'candidates/glm-prefix-bridge-r8705/CPU-build.json';p=json.loads(parent.read_text());assert p['completed']and sha(parent)==m['parent_CPU_build_SHA256']and sha(p['library'])==p['library_SHA256']=='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
old=r/'candidates/glm-window-model-r5797/CPU-build.json';a=json.loads(old.read_text());assert sha(old)=='611d400db8802e6ac38931db0c222527d45adaa1f56d17f0c34a45a81fc31624'
q={'completed':False,'CPU_only':True,'dlopen':False,'GPU_initialized':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]};start=time.monotonic()
def run(cmd,name):
 q['commands'].append(cmd);x=subprocess.run(cmd,text=True,capture_output=True,timeout=180);(b/(name+'.log')).write_text(x.stdout+x.stderr)
 if x.returncode:raise RuntimeError((name,x.returncode,x.stderr[-6000:]))
 return x.stdout
try:
 link=next(list(x)for x in p['commands']if '-shared'in x);objects={x:sha(x)for x in link if x.endswith('.o')};assert len(objects)==28;replacements={};compiled=[]
 shutil.copyfile(r/'incoming/glm_decode_window_header_r8812.hpp',b/'src/glm_model_window_r5797.hpp')
 for canonical,local in [('glm_model_window_r5797.cpp','glm_decode_window_observer_r8812.cpp'),('modern_glm_cached_model_bridge_r809.cpp','glm_decode_window_bridge_r8812.cpp'),('glm_moe_dsa_balanced_text_model_r126.cpp','glm_decode_window_model_r8812.cpp')]:
  rec=next(x for x in a['compiled']if pathlib.Path(x['source']).name==canonical);assert rec['object']in objects and sha(rec['object'])==rec['object_SHA256']and sha(rec['source'])==rec['source_SHA256']
  cmd=next(x for x in a['commands']if '-c'in x and rec['source']in x);fl=[];i=1
  while i<len(cmd):
   if cmd[i]in['-c','-o','-MF','-MT','-MQ']:i+=2;continue
   if cmd[i]in['-MD','-MMD']:i+=1;continue
   fl.append(cmd[i]);i+=1
  src=b/'src'/canonical;shutil.copyfile(r/'incoming'/local,src);obj=b/(canonical+'.o');run([cmd[0],'-I'+str(b/'src'),*fl,'-MD','-MF',str(obj)+'.d','-c',str(src),'-o',str(obj)],canonical);replacements[rec['object']]=str(obj);compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'replaces':rec['object'],'base_source_SHA256':rec['source_SHA256']})
 link=[replacements.get(x,x)for x in link];lib=b/'libglm_decode_window_r8812.so';link[link.index('-o')+1]=str(lib);run(link,'link');nm=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(n in nm for n in ['modern_glm_window_prepare_r5797','modern_glm_window_arm_r5795','modern_glm_window_read_r5795','modern_glm_model_forward'])
 for x,h in objects.items():assert sha(x)==h
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,reused_objects_SHA256={x:h for x,h in objects.items()if x not in replacements},header_SHA256=sha(b/'src/glm_model_window_r5797.hpp'),parent_library_SHA256=p['library_SHA256'],scope='Only3hostTUs/additive-privateheaderdecls; original math/kernel/SDK/streams/sync counts unchanged. NoGPU/dlopen. CPUbuild is notGPUscope hit proof.')
except BaseException as e:q['error']=repr(e);raise
finally:q['seconds']=time.monotonic()-start;(b/'CPU-build.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:q.get(k)for k in ['completed','producer','seconds','library','library_SHA256','error']}))
