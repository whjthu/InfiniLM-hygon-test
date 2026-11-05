from pathlib import Path
import argparse,gzip,hashlib,json,shlex
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def read(archive):
 raw=Path(archive).read_bytes();assert len(raw)==385045 and hashlib.sha256(raw).hexdigest()=='c368d09361bbc46874618616a464d32c24529b6fd226eea9a5680044c257adaf'
 q=json.loads(gzip.decompress(raw));files=q['files'];art=q['artifacts']
 assert len(files)==118 and len(art)==6
 for p,v in files.items():
  b=v['rawtext'].encode();assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['SHA256'],p
 def physical(p):
  assert p.startswith('/refactor/');return R+p[len('/refactor'):]
 def meta(p):return files[physical(p)]
 results={}
 for version,cpu,phase in [('v1','1070','CPU_contract-compile'),('v2','1071',None),('v3','1072','CPU_contract-compile'),('v4','1073','fa_backend-compile')]:
  v=json.loads(files[R+'/candidates/Kimi-split-V128-CPU-'+version+'/CPU-result.json']['rawtext']);assert v['actual_CPU']=='910407.'+cpu and v['completed']and not v['passed']and v.get('phase')==phase
  row=next(t for t in q['accounting_original']if '|Kimi-split-V128-CPU-'+version+'|'in t);assert '|FAILED|1:0|'in row
  results[cpu]={'accounting':row,'phase':phase,'error':v['error'],'elapsed_s':v['elapsed_s']}
 v=json.loads(files[R+'/candidates/Kimi-split-V128-CPU-v4/CPU-result.json']['rawtext']);assert set(v['compiled'])=={'allocate','mla','pair'}
 compiled={}
 for name,t in v['compiled'].items():
  assert meta(t['source'])['SHA256']==t['source_SHA256']and art[physical(t['object'])]['SHA256']==t['object_SHA256']
  md=meta(t['object'][:-2]+'.d');assert md['SHA256']==t['MD_SHA256'];paths=set(shlex.split(md['rawtext'].replace('\\\n',' ').split(':',1)[1]));assert paths==set(t['all_MD'])
  compiled[name]={'source_SHA256':t['source_SHA256'],'object_SHA256':t['object_SHA256'],'MD_SHA256':t['MD_SHA256'],'MD_records':len(paths)}
 assert v['CPU_contract']['passed']and v['CPU_contract']['actual_shared_tile_cases']==23 and v['CPU_contract']['actual_queries_covered']==2295
 assert v['CPU_contract']['stable_F32_LSE_extreme_cases']==5 and not v['CPU_contract']['SDK_planning_supports_executed']
 assert v['CPU_cache']['passed']and v['CPU_cache']['failure_drain_while_alive']and v['binding_recording']['passed']
 planning=v['SDK_planning_compile_only'];assert planning['executed']is False and planning['linked']is False
 assert art[R+'/candidates/Kimi-split-V128-CPU-v4/SDK-planning-compile-only.o']['SHA256']==planning['object_SHA256']
 assert 'no SDK runtime linked/executed'in planning['supports_dynamic_result']
 err=files[R+'/candidates/Kimi-split-V128-CPU-v4/fa_backend-compile.log'];errors=[x for x in err['rawtext'].splitlines()if 'error:'in x];assert len(errors)==1 and 'const void*'in errors[0]and 'void*'in errors[0]
 assert not any(p.endswith('.so')for p in art)
 return dict(audit_passed=True,execution_passed=False,raw_files_verified=len(files),artifact_SHA_metadata_records=len(art),artifact_bytes_recomputed_locally=False,original_CPU_results=results,actual_successful_production_HOST=compiled,contracts=v['CPU_contract'],cache_contract=v['CPU_cache'],binding_contract=v['binding_recording'],SDK_planning='actual header/template compile-only; supports_dynamic UNKNOWN',first_FA_error_log_SHA256=err['SHA256'],new_GPU_TUs_compiled=0,Native_linked=False,GPU_GO=False,Model_GO=False,logical_payload_saving=24*(2*132*256*3*192*2-132*256*3*320*2),qualification='source/pureCPU contracts and 3 actual HOST compiles only; SDKdevice/Runtime/weights/FA execution/fit/speed unknown')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args();print(json.dumps(read(a.archive),indent=2,sort_keys=True))
