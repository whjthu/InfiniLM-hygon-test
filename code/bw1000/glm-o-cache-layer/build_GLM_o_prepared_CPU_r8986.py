import pathlib,subprocess,json,hashlib,os,time,shutil
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-o-prepared-r8986';b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'GLM-o-prepared-CPU-manifest-r8986.json').read_text());commands=[];start=time.perf_counter();d={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'GPU':False,'dlopen':False}
def run(args,label):
 p=subprocess.run(args,capture_output=True,text=True,timeout=110);commands.append(args);(b/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-7000:]);return p.stdout
try:
 for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
 p=json.loads((r/'candidates/glm-prefix-bridge-r8705/CPU-build.json').read_text());assert p['completed']and sha(p['library'])==p['library_SHA256']=='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
 args=next(list(a)for a in p['commands']if '-shared'in a);objects=[a for a in args if a.endswith('.o')];old='/refactor/candidates/glm-transient-layer-r5681/private-15.o';assert objects.count(old)==1;before={n:sha(n)for n in objects}
 vp=json.loads((r/'candidates/glm-transient-layer-r5681/CPU-build.json').read_text());rec=next(x for x in vp['compiled']if x['object']==old);assert sha(rec['source'])==rec['source_SHA256']=='7f994d97fc3b2a50173cf318b24d08d0b714a2258e4c4e183aab1d2f237a9a44'
 cmd=next(a for a in vp['commands']if '-c'in a and rec['source']in a);flags=cmd[1:cmd.index('-MD')];obj=b/'runtime.o';(b/'src').mkdir();src=b/'src/glm_fp8_runtime_o_prepared_r8986.cpp';shutil.copyfile(i/src.name,src)
 run([shutil.which('g++'),'-I/refactor/candidates/glm-transient-layer-r5681/src',*flags,'-MD','-MF',str(b/'runtime.d'),'-c',str(src),'-o',str(obj)],'runtime')
 lib=b/'libglm_o_prepared_r8986.so';args=[str(obj)if a==old else a for a in args];args[args.index('-o')+1]=str(lib);run(args,'link');nm=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(n in nm for n in ['modern_glm_o_prepared_policy_r8986','modern_glm_o_prepared_stats_r8986','modern_glm_model_create','glm_layer_forward_r5670'])
 assert all(sha(n)==h for n,h in before.items());d.update(completed=True,elapsed_seconds=time.perf_counter()-start,library=str(lib),library_SHA256=sha(lib),source_manifest_SHA256=sha(i/'GLM-o-prepared-CPU-manifest-r8986.json'),compiled_TUs=1,compiled=[{'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj)}],reused_objects_SHA256={n:h for n,h in before.items()if n!=old},scope='Private O-onlyM8 defaultOFF2040 reuse dispatch; oneHOSTTU+original28objectlink, no SDK/kernel/dlopen/GPU.')
except BaseException as exc:d['failure']=repr(exc)
d['commands']=commands;(b/'CPU-build.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items()if k not in ['commands','reused_objects_SHA256']}));assert d['completed']
