import pathlib,json,hashlib,subprocess,os,time,re,shlex
r=pathlib.Path('/refactor');src=r/'incoming/Kimi-loader-native-bindings-r9212d';b=r/'candidates/Kimi-loader-native-bindings-r9212d';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]};start=time.monotonic()
def call(c,tag,limit=160):
 q['commands'].append(c);z=subprocess.run(c,capture_output=True,text=True,timeout=limit);(b/(tag+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(tag,z.stderr[-8000:]);return z.stdout
try:
 parent=json.loads((r/'candidates/Kimi-shard-only-Model-r9183/CPU-build.json').read_text());assert parent['completed']and sha(parent['native'])==parent['native_SHA256']=='b4f916dcf32f3bafb7451c94d6a98746a0e6c084edf73aeb6e229275579dbb8d'
 for f,h in parent['all_linked_object_SHA256'].items():assert sha(f)==h,f
 oldworker='/refactor/candidates/Kimi-speculative-batch-Model-r9122c/worker.o';assert parent['all_linked_object_SHA256'][oldworker]=='42b891322186149b0018ddbf8044f2724080e41549ac95950e9bac8e6d176bb9'
 baseline=json.loads((r/'candidates/Kimi-speculative-batch-Model-r9122c/CPU-build.json').read_text());wc=next(c for c in baseline['commands']if '-c'in c and c[c.index('-c')+1].endswith('/worker.cpp'))
 pure=['g++','-std=c++17','-O2','-pthread','-I'+str(src),str(src/'CPU_contract.cpp'),'-o',str(b/'CPU_contract')];call(pure,'purecompile',40);q['pure_contract']=json.loads(call([str(b/'CPU_contract')],'pureexecute',10));assert q['pure_contract']['passed']
 for name,source in [('worker','worker_r9212.cpp'),('weights','weights_r9212.cpp')]:
  c=wc[:];c[1:1]=['-I'+str(src)];c[c.index('-c')+1]=str(src/source);c[c.index('-o')+1]=str(b/(name+'.raw.o'));c[c.index('-MF')+1]=str(b/(name+'.d'));call(c,'compile-'+name)
 oldnm=subprocess.check_output(['nm',str(b/'worker.raw.o')],text=True);symbol=next(x.split()[-1]for x in oldnm.splitlines()if 'load_parameters_no_sync'in x and ' U 'in x);unique='modern_kimi_load_parameters_cached_r9212'
 assert symbol in subprocess.check_output(['nm','--defined-only',str(b/'weights.raw.o')],text=True)
 for name in ['worker','weights']:call(['objcopy','--redefine-sym',symbol+'='+unique,str(b/(name+'.raw.o')),str(b/(name+'.o'))],'rename-'+name,30)
 # Private Weight helpers are copied unchanged for exact LoadBatch behavior, localized to avoid overriding pinnedSDK/publicmethods.
 raw=subprocess.check_output(['nm','--defined-only',str(b/'weights.o')],text=True);dem=subprocess.check_output(['nm','-C','--defined-only',str(b/'weights.o')],text=True);rawlines=raw.splitlines();demlines=dem.splitlines();assert len(rawlines)==len(demlines);local=[]
 for a,z in zip(rawlines,demlines):
  if ' T 'in a and a.split()[-1].startswith(('_ZN8infinilm7weights','_ZNK8infinilm7weights')):local.append(a.split()[-1])
 assert local;call(['objcopy',*[x for s in local for x in ['--localize-symbol',s]],str(b/'weights.o')],'localize-weights',30)
 nw=subprocess.check_output(['nm','-C',str(b/'worker.o')],text=True);assert unique in nw and 'ModelWeights::load_parameters_no_sync('not in nw
 link=parent['commands'][-1][:];assert link.count(oldworker)==1;link[link.index(oldworker)]=str(b/'worker.o');link.insert(link.index('-o'),str(b/'weights.o'));native=b/'libKimi_native_loading_bindings_r9212.so';link[link.index('-o')+1]=str(native)
 for j,x in enumerate(link):
  if x.startswith('-Wl,-Map'):link[j]='-Wl,-Map,'+str(b/'link.map')
 call(link,'link');exports=subprocess.check_output(['nm','-D','-C','--defined-only',str(native)],text=True);oldexports=subprocess.check_output(['nm','-D','-C','--defined-only',parent['native']],text=True);assert unique in exports and 'modern_kimi_loading_binding_counts_r9212'in exports
 # No new ModelWeights/Weight strong public implementation can interpose Runtime.
 strong=[v for v in exports.splitlines()if ' T 'in v and 'infinilm::weights::'in v];assert not strong,strong[:20]
 deps=subprocess.check_output(['ldd',str(native)],text=True);q['ldd_before_assert']=deps;assert 'not found'not in deps,('LD_closure_missing',deps)
 q.update(completed=True,native=str(native),native_SHA256=sha(native),source_SHA256={p.name:sha(p)for p in src.iterdir()if p.is_file()},renamed_exact_member_symbol=symbol,new_private_member_symbol=unique,localized_original_Weight_symbols=len(local),oldworker=oldworker,oldworker_SHA256=parent['all_linked_object_SHA256'][oldworker],all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},runtime_or_Model_execution=False,ldd=deps,elapsed_s=time.monotonic()-start)
except BaseException as e:q['error']=repr(e);q['traceback']=__import__('traceback').format_exc()
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items()if k not in ['commands','all_linked_object_SHA256','ldd']}));assert q['completed']
