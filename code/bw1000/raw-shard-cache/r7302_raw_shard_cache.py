"""Thin CPU/disk-only wrapper over existing InfiniLoader; dry-run by default."""
import argparse,pathlib,json,time,os,sys
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--shard',required=True);p.add_argument('--apply',action='store_true');p.add_argument('--max-bytes',type=int,default=24*1024**3);a=p.parse_args()
source=pathlib.Path(a.source).resolve();cache=pathlib.Path(a.cache_root).resolve();name=pathlib.Path(a.shard)
assert len(name.parts)==1 and name.suffix=='.safetensors','Exactly one explicit HF shard basename required'
file=source/name;assert file.is_file()and file.stat().st_size<=a.max_bytes
assert cache!=source and not cache.is_relative_to(source),'Cache must be an independent project-owned disk directory'
out={'source':str(source),'selected_shard':str(name),'bytes':file.stat().st_size,'cache_root':str(cache),'workers':1,'chunk_bytes':8*1024**2,'mode':'APPLY'if a.apply else'DRY_RUN','memory_scope':'No memfd, resident daemon or rank copies. One stream buffer; OS/filesystem cache charges are additional, not guaranteed free RAM.','cost_scope':'First copy with streaming SHA and subsequent full destination verify are separately charged. Later Model load not measured.'}
if a.apply:
 from infiniloader.local_cache import materialize_model_cache,verify_cache
 existing=next((v for v in [cache,*cache.parents]if v.exists()),None);assert existing is not None
 st=os.statvfs(existing);assert st.f_bavail*st.f_frsize>=out['bytes']+1024**3,'Insufficient filesystem space; this is not a user quota proof'
 tick=time.perf_counter();result=materialize_model_cache(source,cache_root=cache,include_patterns=(str(name),),exclude_patterns=(),workers=1,chunk_size=8*1024**2)
 out['materialize_seconds']=time.perf_counter()-tick;out['result']=result.to_dict();assert result.file_count==1 and result.total_bytes==out['bytes']
 tick=time.perf_counter();out['verify']=verify_cache(result.cache_path,full=True);out['verify_seconds']=time.perf_counter()-tick;assert out['verify']['valid']
 out['cached_shard']=str(result.cache_path/name);out['reuse']='Use this shard path only for the identical original filename. Other shard reads remain original source paths; do not point a complete HF index at an incomplete cache directory.'
print(json.dumps(out,indent=2))
