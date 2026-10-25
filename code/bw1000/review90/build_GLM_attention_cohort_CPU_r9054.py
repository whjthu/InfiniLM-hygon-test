import pathlib,json,hashlib,subprocess,time,os
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-attention-cohort-r9054';b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'GPU_TUs_compiled_on_CPU':0,'HOST_TUs_compiled':0,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]};start=time.perf_counter()
def run(a):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=100);(b/f'command-{len(q["commands"])}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[:18000];return p.stdout
try:
 old=json.loads((r/'candidates/glm-reduced-output-r9034/CPU-build.json').read_bytes());assert old['completed']and old['library_SHA256']=='94f08f7083f6fc6f57d17db3ccf2407ee8cce066a3411c100f62515cafde959f';link=next(list(a)for a in old['commands']if'-shared'in a);objects={x:sha(x)for x in link if x.endswith('.o')};assert len(objects)==28
 replacements={};compiled=[]
 for entry,objname in [('5555','caller.o'),('5096','entry5096.o')]:
  before='/refactor/candidates/glm-reduced-output-r9034/'+objname;assert before in objects
  cmd=next(list(a)for a in old['commands']if'-c'in a and'-o'in a and a[a.index('-o')+1]==before);src=i/f'glm_attention_cohort_{entry}_r9054.cpp';obj=b/objname;cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(b/(objname+'.d'));run(cmd);q['HOST_TUs_compiled']+=1;replacements[before]=str(obj);compiled.append({'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj)})
 lib=b/'libglm_attention_cohort_r9054.so';link=[replacements.get(x,x)for x in link];link[link.index('-o')+1]=str(lib);run(link);nm=run(['nm','-D','--defined-only',str(lib)]);assert all(x in nm for x in ['glm_attention_cohort_policy_r9054','glm_attention_cohort_counts_r9054','glm_attention_cohort_choose_r9054','glm_layer_forward_r5670','glm_reduced_output_choose_r9032']);run(['readelf','-d',str(lib)]);assert all(sha(p)==h for p,h in objects.items());q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,reused_objects_SHA256={p:h for p,h in objects.items()if p not in replacements},elapsed_seconds=time.perf_counter()-start,scope='2HOST only;26current624 objectsunchanged; originalQK/PV GEMM/scale392BF16round; samePscaleSoftmaxCast group;directPVdefaultOFF;noGPUkernel/SDKchange')
except BaseException as e:q['failure']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
