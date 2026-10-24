import pathlib,subprocess,json,hashlib,os,time,shutil,re
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-tile2-r8831';b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'GLM-tile2-CPU-manifest-r8831.json').read_text());commands=[];start=time.perf_counter();d={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'GPU':False,'dlopen':False}
def run(args,label):
 p=subprocess.run(args,capture_output=True,text=True,timeout=180);commands.append({'args':args,'returncode':p.returncode});(b/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-5000:]);return p.stdout
try:
 for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
 parent=r/'candidates/glm-prefix-bridge-r8705/CPU-build.json';p=json.loads(parent.read_text());assert p['completed']and sha(p['library'])==p['library_SHA256']=='5f3dd3d9c2a6715992be48f5de514a145c11d971cb7e07a4ec1bc10fb6a7ad52'
 args=next(list(a)for a in p['commands']if '-shared'in a);objects=[a for a in args if a.endswith('.o')]
 oldgpu='/refactor/candidates/glm-vector-r8529/vector.o';oldglue='/refactor/candidates/glm-vector-model-r8542/vector-1.o';assert objects.count(oldgpu)==objects.count(oldglue)==1
 before={n:sha(n)for n in objects};nvcc='/opt/dtk/cuda/cuda/bin/nvcc';src=i/'glm_output_tile2_r8830.cu';obj=b/'tile2.o'
 run([nvcc,'-std=c++17','-O3','-arch=gfx936','-S',str(src),'-o',str(b/'tile2.s')],'ISA')
 run([nvcc,'-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-MD','-MF',str(b/'tile2.d'),'-c',str(src),'-o',str(obj)],'kernel')
 vp=json.loads((r/'candidates/glm-vector-model-r8542/CPU-build.json').read_text());cmd=next(a for a in vp['commands']if '-c'in a and any(str(x).endswith('glm_vector_dispatch_r8540.cpp')for x in a));flags=cmd[1:cmd.index('-MD')];glue=b/'dispatch.o';run([shutil.which('g++'),'-I/refactor/candidates/glm-vector-model-r8542/src',*flags,'-MD','-MF',str(b/'dispatch.d'),'-c',str(i/'glm_tile2_dispatch_r8830.cpp'),'-o',str(glue)],'dispatch')
 lib=b/'libglm_tile2_r8831.so';args=[str(obj)if a==oldgpu else str(glue)if a==oldglue else a for a in args];args[args.index('-o')+1]=str(lib);run(args,'link');nm=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(n in nm for n in ['modern_glm_tile2_policy_r8830','modern_glm_model_create','glm_tile2_gate_r8830'])
 text=(b/'tile2.s').read_text();meta=[v.strip()for v in text.splitlines()if any(k in v for k in ['NumVgprs','NumSgprs','ScratchSize','.amdhsa_next_free_vgpr','.amdhsa_private_segment_fixed_size'])]
 assert all(sha(n)==h for n,h in before.items())
 d.update(completed=True,elapsed_seconds=time.perf_counter()-start,library=str(lib),library_SHA256=sha(lib),parent_CPU_SHA256=sha(parent),parent_library_SHA256=p['library_SHA256'],source_manifest_SHA256=sha(i/'GLM-tile2-CPU-manifest-r8831.json'),compiled=[{'source':str(src),'object':str(obj),'object_SHA256':sha(obj)},{'source':str(i/'glm_tile2_dispatch_r8830.cpp'),'object':str(glue),'object_SHA256':sha(glue)}],reused_objects_SHA256={n:h for n,h in before.items()if n not in[oldgpu,oldglue]},replaced_objects_SHA256={oldgpu:before[oldgpu],oldglue:before[oldglue]},ISA_metadata=meta,ISA_SHA256=sha(b/'tile2.s'),scope='One CUDA TU + dispatchglue only; original 8705 layer/Model image reused, no SDK/kernel math elsewhere; defaultOFF/routes64 only/down unchanged; GPU correctness and benefit UNKNOWN.')
except BaseException as exc:d['failure']=repr(exc)
d['commands']=commands;(b/'CPU-build.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items()if k not in ['commands','reused_objects_SHA256']}));assert d['completed']
