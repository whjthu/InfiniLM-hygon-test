"""Standalone stdlib audit of original CPU961 archive. No candidate imports."""
from pathlib import Path
import base64,gzip,hashlib,json,shlex,sys
SHA=lambda b:hashlib.sha256(b).hexdigest()
SOURCE='f37bef41ef98b216c1c3b5a1a4a3015a0a161a64bd5d5842f96be5a96e0827cd'
ARCHIVE='d0527ba873ca9c63d020aa48d989b00096b492a4a2da416832dbb89936bea30a'
NATIVE='fc3def9a539f3e9c9e1fc4c4ef2c2f0bbee03ddbe32300c689a8113310068056'
B='/refactor/candidates/Kimi-long-geometry-optin-r453-CPU-v1/'
D='/refactor/incoming/Kimi-long-geometry-optin-r453/'
def symbols(s):return{w[-1]:w[-2]for line in s.splitlines()if len(w:=line.split())>=2}
def body(s,marker):
 i=s.index(marker);i=s.index('{',i);depth=1;j=i+1
 while depth:depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[i:j]
def audit(path):
 transport=json.loads(Path(path).read_bytes());compressed=base64.b64decode(transport['gzip_base64']);assert SHA(compressed)==transport['archive_SHA256']==ARCHIVE
 raw=gzip.decompress(compressed);assert len(raw)==transport['uncompressed_bytes']and SHA(raw)==transport['uncompressed_SHA256']
 a=json.loads(raw);assert a['CPU']=='910407.961'and a['source_manifest_SHA256']==SOURCE and a['GPU']==0 and not a['Native_execution']
 files=a['files'];assert len(files)==transport['rawfiles']==76
 for p,v in files.items():
  b=v['rawtext'].encode();assert len(b)==v['bytes']and SHA(b)==v['SHA256'],p
 get=lambda p:files[p]['rawtext']
 q=json.loads(get(B+'CPU-result.json'));assert q['completed']and q['passed']and q['CPU_only']and q['actual_CPU']=='910407.961'
 assert not q['Native_dlopen']and not q['Runtime_created']and q['GPU']==q['Model']==0
 assert q['source_manifest_SHA256']==SOURCE and q['native_SHA256']==NATIVE
 assert files[D+'CPU_SOURCE_MANIFEST.json']['SHA256']==SOURCE
 m=json.loads(get(D+'CPU_SOURCE_MANIFEST.json'))
 for p,h in m['files_SHA256'].items():assert files[D+p]['SHA256']==h,p
 deriv=json.loads(get(D+'DERIVATION.json'))
 for p,h in deriv['original_raw_source_SHA256'].items():assert files[p]['SHA256']==h,p
 for p,h in deriv['private_source_SHA256'].items():assert files[D+'native/'+p]['SHA256']==h,p
 assert q['new_HOST_TUs']==5 and q['new_GPU_TUs']==0 and q['pure_contract_TUs']==1
 assert len(q['all_linked_object_SHA256'])==33 and len(q['unchanged_objects_SHA256'])==28
 assert len(a['artifacts'])==35 and a['artifacts'][q['native']]['SHA256']==NATIVE
 for p,h in q['all_linked_object_SHA256'].items():assert a['artifacts'][p]['SHA256']==h,p
 base=json.loads(get('/refactor/candidates/Kimi-append512-r247/CPU-build.json'))
 for p,h in q['unchanged_objects_SHA256'].items():assert base['all_linked_object_SHA256'][p]==h
 assert {k:v['SHA256']for k,v in q['actual_dependency_library_SHA256'].items()}=={k:v['SHA256']for k,v in base['actual_dependency_library_SHA256'].items()}
 assert all(v['rc']==0 for v in q['logs'].values())
 for label,v in q['logs'].items():assert files[B+label+'.log']['SHA256']==v['SHA256']
 c=q['actual_header_CPU_contract'];assert c['passed']and c['checks']==837 and c['profiles']==[[16,2],[8,4],[32,1]]and c['all_world_ranks']==32 and c['layers_each_profile']==93 and not c['Native_dlopen']and not c['Runtime_created']
 assert json.loads(get(B+'pure-header-contract-run.log'))==c
 prod=[v for v in q['commands']if '-c'in v];assert len(prod)==5 and {Path(v[v.index('-c')+1]).name for v in prod}=={'pair_probe.cpp','kimi_compact_allocate_r4974.cpp','kimi_moe_scale_exact_r201.cpp','kimi_shard_model_r9183.cpp','kimi_shard_policy_r9183.cpp'}
 assert all(Path(v[0]).name=='g++'and '-O2'in v and '-fPIC'in v for v in prod)
 link=next(v for v in q['commands']if '-shared'in v);assert '-Wl,-Bsymbolic'in link and '-Wl,--no-undefined'in link
 assert {v for v in link if v.endswith('.o')}==set(q['all_linked_object_SHA256'])
 for unit,v in q['HOST'].items():
  def dep(p):return set(shlex.split(get(p).replace('\\\n',' ').split(':',1)[1]))
  assert dep(B+unit+'.d')==dep(B+unit+'.fresh.d')==set(v['all_new_MD'])
  for p,h in v['all_new_MD'].items():
   if p.startswith(D):assert files[p]['SHA256']==h,p
 assert q['actual_known_MD_changed_header_includers']=={'/refactor/incoming/Kimi-shard-only-Model-r9183/kimi_shard_stats_r9183.hpp':['head','policy'],'/refactor/incoming/Kimi-shard-only-Model-r9183/kimi_shard_causal_lm_r9183.hpp':['head'],'/refactor/incoming/Kimi-shard-only-Model-r9183/kimi_shard_head_r9183.hpp':['head'],'/refactor/incoming/Kimi-long-scale-exact-r201/scale_exact_scope_r201.hpp':['moe']}
 reuse=get(B+'reused-object-symbols.log');assert 'kimi_shard_r9183::'not in reuse and 'kimi_scale_exact_r201::'not in reuse
 old=symbols(get(B+'old-defined.log'));new=symbols(get(B+'new-defined.log'))
 strong={p for p,t in old.items()if t.upper()in {'T','D','B','R','G','S','A'}};assert strong<=set(new)and len(strong)==386
 oldc={p for p in old if p.startswith('modern_')};newc={p for p in new if p.startswith('modern_')}
 added={'modern_kimi_engine_create_long_geometry_r453','modern_kimi_engine_forward_long_geometry_r453','modern_kimi_long_geometry_capabilities_r453'}
 assert oldc<=newc and newc-oldc==added and len(oldc)==130
 assert 'not found'not in get(B+'ldd-r.log')and 'undefined symbol:'not in get(B+'ldd-r.log')
 old=get('/refactor/incoming/Kimi-append512-r247/native/pair_probe.cpp');new=get(D+'native/pair_probe.cpp')
 source_same={}
 for marker in ['struct Context','int input_forward_locked_r247(','modern_kimi_engine_create_long_agent_r179(','modern_kimi_engine_create_batch_r9122c(','modern_kimi_pair_append_true_input_r247(','modern_kimi_engine_reset_r2475(','modern_kimi_engine_destroy_r2475(']:
  assert body(old,marker)==body(new,marker);source_same[marker]=SHA(body(old,marker).encode())
 assert 'geometry_r453!=authorized_long_geometry_r453'in new and 'INFINILM_KIMI_LONG_GEOMETRY_R453'in new
 assert 'if(!c->engine->append_ready_r247())'in new
 old=get('/refactor/incoming/Kimi-shard-only-Model-r9183/kimi_shard_head_r9183.hpp');new=get(D+'native/kimi_shard_head_r9183.hpp')
 assert old[old.rindex('private:'):]==new[new.rindex('private:'):]
 assert body(old,'infinilm::runtime::Tensor forward(')==body(new,'infinilm::runtime::Tensor forward(')
 assert 'reference_pipeline_partition.hpp'in get(D+'CPU_contract.cpp')and 'kimi_k3_pipeline_layer_range(93,pp,stage)'in get(D+'CPU_contract.cpp')
 acct=get('accounting961');assert '910407.961|Kimi453-long-geometry-CPU-v1|COMPLETED|0:0|00:01:19|a14r2n20'in acct
 assert not a['owned_srun_PID4107776_cmdline']or 'Kimi453-long-geometry-CPU-v1'not in a['owned_srun_PID4107776_cmdline']
 return dict(schema=1,passed=True,scope='Native HOST/static link + pure header and profile CPU closure; no device qualification',actual_CPU='910407.961',CPU_state='COMPLETED',CPU_exit='0:0',elapsed_s=q['elapsed_s'],source_manifest_SHA256=SOURCE,CPU_receipt_SHA256=files[B+'CPU-result.json']['SHA256'],Native=q['native'],Native_SHA256=NATIVE,rawfiles=76,artifact_metadata=35,production_HOST_TUs=5,pure_header_contract_TUs=1,reused_objects=28,geometry_checks=837,old_strong_exports=386,old_CAPI=130,new_CAPI=sorted(added),source_same=source_same,archive_SHA256=ARCHIVE,archive_path=transport['sealed_path'],archive_uncompressed_SHA256=transport['uncompressed_SHA256'],Native_dlopen=False,Runtime_created=False,GPU=0,Model=0,GPU_GO=False,Model_GO=False,device_fit_numeric_quality_performance='UNKNOWN',artifact_bytes_scope='Object/library bytes hashed on original CPU host at seal; archive contains metadata, source/MD/logs, not ELF object payloads')
if __name__=='__main__':
 out=audit(sys.argv[1]);b=(json.dumps(out,indent=2)+'\n').encode()
 if len(sys.argv)>2:
  p=Path(sys.argv[2]);assert not p.exists();p.write_bytes(b)
 else:print(b.decode())
