"""CPU-only two-TU relink with unique Model/template; no dlopen or weights."""
import pathlib,json,hashlib,subprocess,shlex
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'incoming/Kimi-head-Model-compile-r8344.json').read_text())
b=r/'candidates/kimi-vocab-Model-CPU-r8344';b.mkdir(exist_ok=False)
out={'completed':False,'CPU_compile_only':True,'dlopen':False,'commands':[]}
def run(a,label):
 out['commands'].append(a)
 with(b/(label+'.log')).open('x')as f:p=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 if p.returncode:raise RuntimeError(label+' '+(b/(label+'.log')).read_text()[-6000:])
def deps(p):return sorted(set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])))
try:
 for n,h in m['sources'].items():assert sha(r/'incoming'/n)==h,n
 q=json.loads((r/'results/Kimi-N4-CPU-r8174.json').read_text())
 assert q['completed']and sha(pathlib.Path(q['library']))==q['library_SHA256']==m['parent_DSO_SHA256']
 for v in q['actual_deps'].values():
  for n,h in v.items():assert sha(pathlib.Path(n))==h,n
 for n,h in q['actual_dynamic_dependency_SHA256'].items():assert sha(pathlib.Path(n))==h,n
 old=json.loads((r/'results/kimi-packed-W2-Model-r5049-build.json').read_text())
 proofs={v['path']:v['SHA256']for v in q['parent_reused_host_objects']}
 proofs.update(q['parent_GPU_objects']);proofs[old['GPU_object']]=old['GPU_object_SHA256']
 for n,h in q['new_object_SHA256'].items():proofs[str(pathlib.Path(q['library']).parent/n)]=h
 link=list(q['commands'][-1]);objects=[v for v in link if v.endswith('.o')]
 assert all(n in proofs and sha(pathlib.Path(n))==proofs[n]for n in objects)
 oldmodel=next(v for v in q['parent_reused_host_objects']if v['source']==m['replace_only_source'])['path'];assert link.count(oldmodel)==1
 base=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc'
 common=['g++','-std=c++17','-O2','-fPIC','-pthread',*[v for v in link if v.startswith('-D')],'-I'+str(r/'incoming'),'-I'+str(base),*[v for v in link if v.startswith('-I')],'-I/opt/dtk/cuda/cuda/include']
 out['compiled_units']=[]
 for name in ['kimi_head_model_r8344','kimi_vocab_policy_r8341']:
  source=r/'incoming'/(name+'.cpp');md=b/(name+'.d');fresh=b/(name+'.fresh.d');obj=b/(name+'.o')
  run([*common,'-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],name)
  run([*common,'-M','-MF',str(fresh),str(source)],name+'-freshM');assert deps(md)==deps(fresh)
  out['compiled_units'].append({'source':str(source),'object':str(obj),'object_SHA256':sha(obj),'MD_equals_fresh_M':True,'deps':{n:sha(pathlib.Path(n))for n in deps(md)}})
 library=b/'libmodern_Kimi_vocab_Model_r8344.so'
 link[link.index(oldmodel)]=str(b/'kimi_head_model_r8344.o');link.insert(1,str(b/'kimi_vocab_policy_r8341.o'));link[link.index('-o')+1]=str(library)
 for i,arg in enumerate(link):
  if arg.startswith('-Wl,-Map,'):link[i]='-Wl,-Map,'+str(b/'link.map')
 run(link,'link');defined=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True);undefined=subprocess.check_output(['nm','-D','--undefined-only',str(library)],text=True);dynamic=subprocess.check_output(['readelf','-d',str(library)],text=True)
 assert 'SYMBOLIC'in dynamic and 'libmodern_Kimi'not in dynamic
 assert all(n in defined for n in ['modern_kimi_vocab_head_get_r8341','modern_kimi_vocab_head_set_r8341','modern_kimi_vocab_head_count_r8341','KimiK3ForConditionalGenerationHeadR8341','TextCausalLMHeadR8341','DualVocabHeadR8341','VocabParallelR8296'])
 assert all(n not in undefined for n in ['HeadR8341','VocabParallelR8296','modern_kimi_vocab_head_'])
 out.update(completed=True,library=str(library),library_SHA256=sha(library),all_linked_object_SHA256={n:sha(pathlib.Path(n))for n in link if n.endswith('.o')},actual_dynamic_dependency_SHA256=q['actual_dynamic_dependency_SHA256'],parent_DSO_SHA256=q['library_SHA256'],input_manifest_SHA256=sha(r/'incoming/Kimi-head-Model-compile-r8344.json'),nm_defined=defined,nm_undefined=undefined,readelf_dynamic=dynamic,unique_Model_and_template_symbols=True,source=q['parent_reused_host_objects'],no_live_Model_or_SDK_mutation=True,GPU_or_Model_qualification='NOT_RUN',memory_scope='Validationimage retainsfullhead+shard;no ModelVRAMsavingclaimed')
except BaseException as e:out['error']=repr(e);raise
finally:
 with(r/'results/Kimi-head-Model-CPU-r8344.json').open('x')as f:json.dump(out,f,indent=2)
 print(json.dumps({'completed':out['completed'],'error':out.get('error'),'library_SHA256':out.get('library_SHA256'),'CPU_only':True}))
