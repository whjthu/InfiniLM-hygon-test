import pathlib,json,hashlib,subprocess,time,os
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-M16-cohort-r9056b';b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'GPU_TUs_compiled_on_CPU':0,'HOST_TUs_compiled':0,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]};start=time.perf_counter()
def run(a):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=110);(b/f'command-{len(q["commands"])}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[:18000];return p.stdout
try:
 parent=json.loads((r/'candidates/glm-attention-cohort-r9054/CPU-build.json').read_bytes());assert parent['completed']and parent['library_SHA256']=='45a3ad06b789152c532abcaa27a45db582bf55669ec4f7a56e6ba08b9de3003a';link=next(list(a)for a in parent['commands']if'-shared'in a);objects={x:sha(x)for x in link if x.endswith('.o')};assert len(objects)==28
 old='/refactor/candidates/glm-PP2-B32-r8843/private-0.o';assert objects[old]=='22fe1940f181a7c27cd204ad38980c3017d159366cb5fd40aa11368b4dc2bb4f';doc=json.loads((r/'candidates/glm-PP2-B32-r8843/CPU-build.json').read_bytes());rec=next(x for x in doc['compiled']if x['object']==old);assert sha(rec['source'])==rec['source_SHA256']=='a89fc588f590b75968edd488c5bfeb8a7e243f38112c97a50a4d0f5fcf9b8024'
 src=i/'glm_M16_bridge_r9056.cpp';original=pathlib.Path(rec['source']).read_text();assert original.count('!B||B>8||')==1;assert src.read_text()==original.replace('!B||B>8||','!B||B>16||');original_input=src; (b/'src').mkdir();src=b/'src/bridge.cpp';src.write_bytes(original_input.read_bytes())
 cmd=next(list(a)for a in doc['commands']if'-c'in a and'-o'in a and a[a.index('-o')+1]==old);obj=b/'bridge.o';cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(b/'bridge.d');run(cmd);q['HOST_TUs_compiled']=1
 import shlex
 def dependencies(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
 actualdeps=dependencies(b/'bridge.d');olddeps=dependencies(pathlib.Path(old).with_suffix('.d'));classheads=['glm_moe_dsa_projected_decoder_r22.hpp','glm_moe_dsa_projected_cache_r22.hpp','glm_model_resident_policy_r5640.hpp']
 q['actual_header_closure']={}
 for name in classheads:
  prior=[p for p in olddeps if pathlib.Path(p).name==name];now=[p for p in actualdeps if pathlib.Path(p).name==name];assert len(prior)==len(now)==1,(name,prior,now);assert prior==now and sha(prior[0])==sha(now[0]);q['actual_header_closure'][name]={'path':now[0],'SHA256':sha(now[0])}

 lib=b/'libglm_M16_cohort_r9056b.so';link=[str(obj)if x==old else x for x in link];link[link.index('-o')+1]=str(lib);run(link);nm=run(['nm','-D','--defined-only',str(lib)]);assert all(n in nm for n in ['glm_layer_forward_r5670','glm_layer_create_r5670','glm_attention_cohort_policy_r9054','glm_attention_cohort_counts_r9054']);run(['readelf','-d',str(lib)]);assert all(sha(p)==h for p,h in objects.items())
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),source_SHA256=sha(src),object_SHA256=sha(obj),reused_objects_SHA256={p:h for p,h in objects.items()if p!=old},elapsed_seconds=time.perf_counter()-start,scope='ROOT explicit oneHOST existingCAPI B<=8to16 only,N4096unchanged;27others same639including2cohortcallers;noSDK/GPU/math/newruntime/cachebank enlargement')
except BaseException as e:q['failure']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
