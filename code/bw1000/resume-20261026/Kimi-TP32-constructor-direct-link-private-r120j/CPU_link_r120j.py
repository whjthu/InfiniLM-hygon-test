"""Minimal link-only repair: existing457 object + Native0f + explicit Common2e14."""
import pathlib,json,hashlib,subprocess,os,shlex,time,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Kimi-TP32-construct-r120j';F=R/'incoming/Kimi-TP32-Native-r120f';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def deps(p):return sorted(set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])))
def main():
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);m=json.loads((D/'manifest-r120j.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 cp=R/'candidates/Kimi-TP32-Native-r120f/CPU-build.json';fp=R/'candidates/Kimi-TP32-construct-r120f/CPU-build.json';hp=R/'candidates/Kimi-TP32-construct-r120h/CPU-build.json';obj=fp.parent/'construct.o';md=fp.parent/'construct.d';fresh=fp.parent/'construct.fresh.d';common=pathlib.Path(m['Common_provider'])
 assert sha(cp)==m['Native_CPU_receipt_SHA256']and sha(fp)==m['failed457_receipt_SHA256']and sha(hp)==m['failed460_receipt_SHA256']and sha(obj)==m['construct_object_SHA256']and sha(common)==m['Common_SHA256'];c=json.loads(cp.read_bytes());f=json.loads(fp.read_bytes());h=json.loads(hp.read_bytes())
 assert c['completed']and c['producer']=='910407.454'and sha(c['native'])==c['native_SHA256']==m['Native_SHA256']and not f['completed']and f['producer']=='910407.457'and not h['completed']and h['producer']=='910407.460'
 assert c['source_manifest_SHA256']==sha(F/'manifest-r120f.json')==m['Native_source_manifest_SHA256'];s=json.loads((F/'manifest-r120f.json').read_bytes())
 for n,x in s['source_SHA256'].items():assert sha(F/n)==x,n
 for p,x in c['required_DSO_SHA256'].items():assert sha(p)==x,p
 for p,x in c['all_linked_object_SHA256'].items():assert sha(p)==x,p
 actual=deps(md);assert actual==deps(fresh);mdmap={p:sha(p)for p in actual};shared={}
 for old in c['actual_complete_freshMD'].values():
  for p,x in old.items():
   if p in mdmap:assert mdmap[p]==x,(p,'actual shared header ABI mismatch');shared[p]=x
 assert '-D_GLIBCXX_USE_CXX11_ABI=1'in f['commands'][0]and all(x in f['commands'][0]for x in ['-O2','-fPIC','-std=c++17'])
 watched={str(p):sha(p)for p in [cp,fp,hp,obj,md,fresh,common,pathlib.Path(c['native']),F/'manifest-r120f.json',*map(pathlib.Path,c['all_linked_object_SHA256']),*map(pathlib.Path,mdmap),*map(pathlib.Path,c['required_DSO_SHA256'])]}
 B.mkdir(exist_ok=False);start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'GPU_initialized':False,'dlopen':False,'Runtime_create':False,'HOST_TUs_compiled':0,'GPU_TUs_compiled':0,'reused_compiled_constructor_HOST_TUs':1,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[],'provider_mode':'private_Native0f_plus_explicit_existing_Common2e14','parent_CPU_receipt_SHA256':sha(cp),'native':c['native'],'native_SHA256':c['native_SHA256'],'Common_provider':str(common),'Common_SHA256':sha(common),'construct_object_SHA256':sha(obj),'source_manifest_SHA256':c['source_manifest_SHA256'],'repair_manifest_SHA256':sha(D/'manifest-r120j.json'),'preserved_failed457_receipt_SHA256':sha(fp),'preserved_failed460_receipt_SHA256':sha(hp),'watched_before_SHA256':watched}
 def run(cmd,label,timeout=60):
  q['commands'].append(cmd);z=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-10000:]);return z.stdout
 try:
  undef=run(['nm','--undefined-only',str(obj)],'construct-undefined');unames=[line.split()[-1]for line in undef.splitlines()if line.split()];z=subprocess.run(['c++filt'],input='\n'.join(unames)+'\n',text=True,capture_output=True,timeout=20);assert z.returncode==0;readable=dict(zip(unames,z.stdout.splitlines()))
  common_defs=run(['nm','-D','--defined-only',str(common)],'Common-dynamic');cnames={line.split()[-1]for line in common_defs.splitlines()if line.split()};native_defs=run(['nm','-D','--defined-only',c['native']],'Native-dynamic');nnames={line.split()[-1]for line in native_defs.splitlines()if line.split()};audit=[]
  families=['ModelWeights::state_dict','BaseLinear','BaseQuantization','NoneQuantization','ColumnParallelLinear::ColumnParallelLinear','PagedKVCacheConfig::PagedKVCacheConfig']
  for name,text in readable.items():
   if any(t in text for t in families):
    assert name in cnames,('actual Common provider absent',name,text);audit.append({'undefined':name,'demangled':text,'actual_public_provider':str(common),'provider_SHA256':sha(common),'also_exported_by_Native':name in nnames})
  assert all(any(t in a['demangled']for a in audit)for t in families);q['actual_Common_provider_audit']=audit;q['shared_actual_MD_ABI_headers']=shared
  original=f['commands'][-1][:];assert '-shared'in original and '-c'not in original and '-Wl,--no-undefined'in original and [p for p in original if p.endswith('.o')]==[str(obj)]
  assert original.count(c['native'])==1 and str(common)not in original;link=original[:];link.insert(link.index('-Wl,--as-needed'),str(common));lib=B/'libKimi_TP32_construct_r120j.so';link[link.index('-o')+1]=str(lib)
  assert all(x in link for x in ['-Wl,--no-undefined',c['native'],str(common),str(obj)])and not any(p.endswith(('.cpp','.cu'))for p in link);run(link,'link',150)
  dynamic=run(['readelf','-d',str(lib)],'dynamic');assert dynamic.count(pathlib.Path(c['native']).name)==1 and dynamic.count(common.name)==1
  defs=run(['nm','-D','--defined-only',str(lib)],'wrapper-dynamic');assert all(t in defs for t in ['modern_kimi_TP32_construct_r120f','modern_kimi_TP32_construct_result_r120f'])
  env=dict(os.environ);env['LD_LIBRARY_PATH']='/usr/local/lib/python3.10/dist-packages/torch/lib:'+str(common.parent)+':'+env.get('LD_LIBRARY_PATH','');z=subprocess.run(['ldd','-r',str(lib)],env=env,text=True,capture_output=True,timeout=60);(B/'ldd-r.log').write_text(z.stdout+z.stderr);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr and 'undefined symbol:'not in z.stdout+z.stderr
  assert pathlib.Path(c['native']).name in z.stdout+z.stderr and common.name in z.stdout+z.stderr;after={p:sha(p)for p in watched};assert after==watched
  q.update(completed=True,library=str(lib),library_SHA256=sha(lib),fresh_MD_SHA256=mdmap,required_DSO_SHA256={**c['required_DSO_SHA256'],str(common):sha(common)},Native454_object_provenance_SHA256=c['all_linked_object_SHA256'],actual_linked_object_SHA256={str(obj):sha(obj)},ELF_dynamic=dynamic,ldd=z.stdout+z.stderr,watched_after_SHA256=after,explicit_direct_Common_dependency=True,private_provider_expectation='KDA/MLA/cacheallocator/CommunicationGroup:Native0f',common_provider_expectation='BaseLinear/ColumnParallelLinear/CacheConfig/ModelWeights/quantization:Common2e14',no_new_Native_objectset_or_math=True,normal_CPU_exit=True,GPU_GO=False,fullModel_budget='UNKNOWN_STOP')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','library_SHA256','HOST_TUs_compiled','error']}));assert q['completed']
if __name__=='__main__':main()
