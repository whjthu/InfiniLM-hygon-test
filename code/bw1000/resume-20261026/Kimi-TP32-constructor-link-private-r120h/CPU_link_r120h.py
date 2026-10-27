"""Link-only recovery. Reuse Native454 objects + actual457 construct.o once."""
import pathlib,json,hashlib,subprocess,os,shlex,time,traceback,collections
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Kimi-TP32-construct-r120h';F=R/'incoming/Kimi-TP32-Native-r120f';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def deps(p):return sorted(set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])))
def main():
 assert all(os.environ.get(n)=='-1'for n in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 m=json.loads((D/'manifest-r120h.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 cp=R/'candidates/Kimi-TP32-Native-r120f/CPU-build.json';fp=R/'candidates/Kimi-TP32-construct-r120f/CPU-build.json';obj=fp.parent/'construct.o';md=fp.parent/'construct.d';fresh=fp.parent/'construct.fresh.d'
 assert sha(cp)==m['Native_CPU_receipt_SHA256']and sha(fp)==m['failed457_receipt_SHA256'];c=json.loads(cp.read_bytes());f=json.loads(fp.read_bytes())
 assert c['completed']and c['producer']=='910407.454'and c['native_SHA256']==m['Native_provenance_SHA256']==sha(c['native'])
 assert not f['completed']and f['producer']=='910407.457'and f['HOST_TUs_compiled']==1 and f['GPU_TUs_compiled']==0 and f['parent_CPU_receipt_SHA256']==sha(cp)
 assert f['commands'][0][f['commands'][0].index('-o')+1]==str(obj)and 'undefined reference'in f['error']
 assert sha(F/'manifest-r120f.json')==c['source_manifest_SHA256']==m['Native_source_manifest_SHA256'];s=json.loads((F/'manifest-r120f.json').read_bytes())
 for n,h in s['source_SHA256'].items():assert sha(F/n)==h,n
 for p,h in c['required_DSO_SHA256'].items():assert sha(p)==h,p
 for p,h in c['all_linked_object_SHA256'].items():assert sha(p)==h,p
 assert c['all_linked_object_SHA256']==m['Native454_object_SHA256']
 paths=deps(md);assert paths==deps(fresh);current_md={p:sha(p)for p in paths}
 shared={}
 for label,old_md in c['actual_complete_freshMD'].items():
  for p,h in old_md.items():
   if p in current_md:assert h==current_md[p],(label,p,'cross-TU header mismatch');shared[p]=h
 for n in ['kimi_shard_head_r9183.hpp','kimi_shard_stats_r9183.hpp','tp32_contract_r120f.hpp']:
  p=str(F/n);assert p in current_md and p in shared
 watched={str(p):sha(p)for p in [cp,fp,obj,md,fresh,F/'manifest-r120f.json',*map(pathlib.Path,c['all_linked_object_SHA256']),*map(pathlib.Path,current_md),*map(pathlib.Path,c['required_DSO_SHA256']),pathlib.Path(c['native'])]}
 B.mkdir(exist_ok=False);start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'GPU_initialized':False,'dlopen':False,'Runtime_create':False,'HOST_TUs_compiled':0,'GPU_TUs_compiled':0,'reused_compiled_constructor_HOST_TUs':1,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[],'provider_mode':'selfcontained_single_DSO_from_Native454_objects_plus_construct457_object','load_separate_Native':False,'Native_provenance_library':c['native'],'Native_provenance_SHA256':c['native_SHA256'],'parent_CPU_receipt_SHA256':sha(cp),'preserved_failed457_receipt_SHA256':sha(fp),'source_manifest_SHA256':c['source_manifest_SHA256'],'repair_manifest_SHA256':sha(D/'manifest-r120h.json'),'construct_object':str(obj),'construct_object_SHA256_at_repair_start':sha(obj),'prior457_receipt_did_not_record_object_SHA_before_link_failure':True,'watched_before_SHA256':watched}
 def run(cmd,label,timeout=90):
  q['commands'].append(cmd);z=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-12000:]);return z.stdout
 try:
  objects=list(c['all_linked_object_SHA256']);definitions=collections.defaultdict(list);strong=collections.defaultdict(list)
  for index,p in enumerate([*objects,str(obj)]):
   object_hash=sha(p)
   text=run(['nm','-S','--extern-only','--defined-only','--format=posix',p],'object-nm-'+str(index),40)
   for line in text.splitlines():
    v=line.split()
    if len(v)<2:continue
    name,kind=v[:2];entry={'object':p,'object_SHA256':object_hash,'type':kind,'nm_fields':v[2:]};definitions[name].append(entry)
    if kind not in ['W','w','V','v','u']:strong[name].append(entry)
  conflict={n:v for n,v in strong.items()if len(v)>1};assert not conflict,('strong-definition ODR conflict',conflict)
  utext=run(['nm','--undefined-only',str(obj)],'construct-undefined',40);unames=[line.split()[-1]for line in utext.splitlines()if line.split()]
  z=subprocess.run(['c++filt'],input='\n'.join(unames)+'\n',text=True,capture_output=True,timeout=30);assert z.returncode==0;demangled=dict(zip(unames,z.stdout.splitlines()));dyntext=run(['nm','-D','--defined-only',c['native']],'old-Native-dynamic',40);dyn={line.split()[-1]for line in dyntext.splitlines()if line.split()}
  families=['ModelWeights::state_dict','BaseLinear::weight','BaseLinear::process_weights_after_loading','vtable for infinilm::nn::BaseLinear','PagedKVCacheConfig::PagedKVCacheConfig','BaseQuantization','NoneQuantization','ColumnParallelLinear::ColumnParallelLinear'];located=[]
  for name,readable in demangled.items():
   if any(term in readable for term in families):
    assert name in definitions,('missing actual object provider',readable);located.append({'symbol':name,'demangled':readable,'exported_by_old_Native_DSO':name in dyn,'actual_definition_providers':definitions[name]})
  assert all(any(term in x['demangled']for x in located)for term in families),('incomplete failed-symbol audit',located)
  assert any(not x['exported_by_old_Native_DSO']for x in located)
  critical_duplicates={n:v for n,v in definitions.items()if len(v)>1 and ('ShardOnlyVocabHeadR9183'in n or 'kimi_shard_r9183'in n)}
  q['ODR']={'strong_duplicate_definitions':conflict,'critical_weak_or_unique_duplicates':critical_duplicates,'identical_actual_shared_header_SHA256':shared,'proof':'Same checked r120f head/stats/contract headers and complete actualMD; duplicate inline definitions are weak/COMDAT/GNUunique; strong conflicts rejected. No claim that nm byte size alone proves ODR.'};q['failed_hidden_symbol_provider_audit']=located
  link=next(cmd[:]for cmd in c['commands']if '-shared'in cmd and '-o'in cmd and cmd[cmd.index('-o')+1]==c['native']);assert '-c'not in link and '-Wl,--no-undefined'in link and '-Wl,-Bsymbolic'in link
  assert [x for x in link if x.endswith('.o')]==objects;assert c['native']not in link[:link.index('-o')]
  lib=B/'libKimi_TP32_construct_self_r120h.so';link.insert(1,str(obj));link[link.index('-o')+1]=str(lib)
  for i,x in enumerate(link):
   if x.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
  assert not any(x.endswith(('.cpp','.cc','.cu'))for x in link);run(link,'link',180)
  dynamic=run(['readelf','-d',str(lib)],'dynamic',40);assert 'SYMBOLIC'in dynamic and pathlib.Path(c['native']).name not in dynamic
  newdefs=run(['nm','-D','--defined-only',str(lib)],'new-dynamic-symbols',40);newnames={line.split()[-1]for line in newdefs.splitlines()if line.split()};assert dyn<=newnames
  required=['modern_kimi_TP32_construct_r120f','modern_kimi_TP32_construct_result_r120f'];assert all(n in newnames for n in required)
  for pattern in ['KimiK3DeltaAttentionC1','KimiK3MLAAttentionC1','kimi_k3_allocate_cache_tensors','CommunicationGroupC1']:
   assert len([n for n in newnames if pattern in n])==1,(pattern,'self-contained runtime provider must be publicly auditable')
  # Verify old hidden definitions became locally resolved in this one DSO.
  allfinal=run(['nm','--defined-only',str(lib)],'new-all-symbols',40);finalnames={line.split()[-1]for line in allfinal.splitlines()if line.split()};assert all(x['symbol']in finalnames for x in located)
  env=dict(os.environ);env['LD_LIBRARY_PATH']='/usr/local/lib/python3.10/dist-packages/torch/lib:'+env.get('LD_LIBRARY_PATH','');z=subprocess.run(['ldd','-r',str(lib)],env=env,text=True,capture_output=True,timeout=60);(B/'ldd-r.log').write_text(z.stdout+z.stderr);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr and 'undefined symbol:'not in z.stdout+z.stderr and pathlib.Path(c['native']).name not in z.stdout+z.stderr
  after={p:sha(p)for p in watched};assert after==watched
  q.update(completed=True,library=str(lib),library_SHA256=sha(lib),all_reused_object_SHA256={p:sha(p)for p in [*objects,str(obj)]},Native454_object_SHA256=c['all_linked_object_SHA256'],fresh_MD_SHA256=current_md,required_DSO_SHA256=c['required_DSO_SHA256'],watched_after_SHA256=after,ELF_dynamic=dynamic,ldd=z.stdout+z.stderr,all_old_dynamic_exports_retained=True,actual_model_constructor_provider='SELFCONTAINED_DSO_NOT_ORIGINAL_0f',one_private_Native_objectset_linked=True,registry_runtime_state_proof='UNQUALIFIED_until_GPU_entry_dladdr_maps; separate_originalNative_DT_NEEDED_absent',GPU_GO=False,fullModel_budget='UNKNOWN_STOP',normal_CPU_exit=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','library_SHA256','HOST_TUs_compiled','error']}));assert q['completed']
if __name__=='__main__':main()
