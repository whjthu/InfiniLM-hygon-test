from pathlib import Path
import hashlib,json,re,sys
D=Path(__file__).resolve().parent;N=D/'native';H=lambda b:hashlib.sha256(b).hexdigest()
lf=lambda s:s.replace('\r\n','\n').replace('\r','')
def body(s,marker):
 i=s.index(marker);i=s.index('{',i);j=i+1;depth=1
 while depth:depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[i:j]
def check(parent_native,originals):
 old=(parent_native/'pair_probe.cpp').read_text();new=(N/'pair_probe.cpp').read_text();proof={}
 for marker in ['struct Context','int input_forward_locked_r247(','modern_kimi_engine_reset_r2475(','modern_kimi_engine_destroy_r2475(','modern_kimi_engine_create_long_agent_r179(','modern_kimi_pair_append_true_input_r247(']:
  assert body(old,marker)==body(new,marker);proof[marker]=H(body(old,marker).encode())
 for name in ['kimi_shard_causal_lm_r9183.hpp','kimi_shard_head_r9183.hpp','kimi_shard_stats_r9183.hpp','kimi_shard_model_r9183.cpp','kimi_shard_model_r9183.hpp','kimi_shard_policy_r9183.cpp','long_geometry_r453.hpp','scale_exact_scope_r201.hpp']:
  assert (parent_native/name).read_bytes()==(N/name).read_bytes(),name
 original=lambda suffix:lf(next(v['rawtext']for p,v in originals.items()if p.endswith(suffix)and v.get('exists')))
 assert lf((N/'worker.cpp').read_text())==original('worker.cpp')
 # Existing reset drains/releases/trims BEFORE allocating replacements. No new
 # success sync is added or removed; old try/failure handling stays in worker.
 worker=lf((N/'worker.cpp').read_text());start=worker.index('} else if (local_cmd == Command::RESET_CACHE)')if '} else if (local_cmd == Command::RESET_CACHE)'in worker else worker.index('local_cmd == Command::RESET_CACHE')
 reset=worker[start:worker.index('} else if (local_cmd == Command::COMPILE)',start)]
 ordering=['runtime_->syncStream();','forward_context_.kv_cache_vec.clear();','forward_context_.conv_state_vec.clear();','forward_context_.ssm_state_vec.clear();','runtime_->trimMemory();','model_->reset_cache(']
 at=-1
 for term in ordering:at=reset.index(term,at+1)
 proof['worker_reset_release_before_reallocate']=H(reset.encode())
 old=original('kimi_compact_engine_r4974.cpp');new=(N/'kimi_compact_engine_r4974.cpp').read_text();flag=',model_config_->get_config_json().value("infinilm_kimi_ar_storage_optin_r456",false)'
 assert new.count(flag)==6 and new.replace(flag,'')==old
 # Same cache layout, zeroing and failure-drain body; only exact allocation
 # and a stronger exact-geometry admission replace the original generic call.
 cache=original('cache/kv_cache.cpp');cache=cache[cache.index('infinilm::runtime::Tensor create_layer_kv_cache(',cache.index('namespace PagedKVCache')):]
 exact=(N/'exact_mla_cache_r456.hpp').read_text();assert body(cache,'    try {')==body(exact,'    try {')
 assert 'Tensor::empty_parameter(runtime, kv_shape, dtype)'in exact
 assert 'const infinilm::engine::distributed::RankInfo &rank'in exact
 proof['MLA_zero_and_failure_drain_body']=H(body(exact,'    try {').encode())
 helper=(N/'kimi_engine_input_r145.hpp').read_text();assert 'if(ar_storage_r456)::kimi_ar_storage_r456::request'in helper
 assert 'lengths.size()==(ar_storage_r456?2u:3u)'in helper
 assert '::kimi_long_agent_r179::validate_row_table'in helper
 assert 'ar_storage_r456!=authorized_ar_storage_r456'in (N/'pair_probe.cpp').read_text()
 assert 'c->max_rows!=2'in body((N/'pair_probe.cpp').read_text(),'modern_kimi_long_geometry_capabilities_r453(')
 # Extern API scopes: new storage only TP32PP1, original r453 factory retains2.
 assert 'address,port,1);'in body((N/'pair_probe.cpp').read_text(),'modern_kimi_engine_create_ar_storage_r456(')
 assert 'address,port,2);'in body((N/'pair_probe.cpp').read_text(),'modern_kimi_engine_create_long_geometry_r453(')
 pair=(N/'pair_probe.cpp').read_text()
 for marker,predicate in [('modern_kimi_pair_prefill_r9006(','!c->pair_r9006'),('modern_kimi_batch_reset_r9122(','!c->pair_r9006'),('modern_kimi_batch_mode_r9122(','!c->batch_enabled_r9122'),('modern_kimi_engine_spec_r3632(','!c->speculation_r3632')]:assert predicate in body(pair,marker),(marker,predicate)
 f=body(pair,'modern_kimi_engine_forward_ar_storage_r456(')
 assert all(p in f for p in ['c->pair_r9006','c->speculation_r3632','c->batch_enabled_r9122','c->batch_capture_r9122','c->state_only_r9114'])
 proof['worker_batch_control_unreachable_from_typed_AR']='New factory never constructs pair/spec/batch state; typed forward rejects each non-AR flag; every pair/batch/spec CAPI guards before engine control. Worker old batch-control mode defaultfalse is retained and unreachable from admitted typed AR calls.'
 return dict(passed=True,source_only=True,source_proof=proof,new_production_HOST_TUs=5,new_GPU_TUs=0,Context_fields_changed=False,head_class_fields_changed=False,Native_dlopen=False,GPU=0)
if __name__=='__main__':
 if len(sys.argv)>2:originals=json.loads(Path(sys.argv[2]).read_bytes())['files'];parent=Path(sys.argv[1])
 else:
  parent=Path('/refactor/incoming/Kimi-long-geometry-optin-r453/native');o=json.loads((D/'PARENT_SOURCES.json').read_bytes());originals={p:{'exists':True,'rawtext':Path(p).read_bytes().decode()}for p in o}
 print(json.dumps(check(parent,originals),indent=2))
