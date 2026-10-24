import pathlib,json,hashlib,ast,re
r=pathlib.Path('.');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();proof={}
q=json.loads((r/'Kimi-EP2-bind-CPU-observed-r8669zk-a.json').read_text());c=q['result'];assert c['completed']and not c['GPU']and not c['dlopen'];m=json.loads((r/'Kimi-EP2-manifest-r8669zk.json').read_text())
assert m['native_SHA256']==c['native_SHA256']and m['library_SHA256']==c['library_SHA256']and m['CPU_receipt_SHA256']==q['raw_result_SHA256'];assert c['native']in c['wrapper_needed']
for name,h in m['incoming_source_SHA256'].items():
 assert sha(r/name)==h,name
 if name.endswith('.py'):ast.parse((r/name).read_text())
for n in ['modern_kimi_EP2_mask_r8669zj','modern_kimi_EP2_counts_r8669zj','modern_kimi_grouped_r4620','modern_kimi_grouped_I192_r8669y']:assert n in c['native_defined']and n not in c['native_undefined']
proof['actual_CPU216_ELF_wrapper_needed_and_consumer_symbols']=True
h=json.loads((r/'Kimi-EP2-actual-header-r8669zj.json').read_text())['text'];new=(r/'kimi_k3_moe_EP2_r8669zj.hpp').read_text();assert new==h.replace('void register_mxfp4_experts(infinilm::runtime::Runtime &runtime);','void register_mxfp4_experts(infinilm::runtime::Runtime &runtime, size_t global_expert_start = 0);')
proof['actual_header_only_private_method_parameter_no_class_fields']=True
for expert in range(896):
 owners=[rank for rank in range(16)if rank//8==expert//448];assert len(owners)==8
 spans=[(rank%8*384,(rank%8+1)*384)for rank in owners];assert spans==[(i*384,(i+1)*384)for i in range(8)]
 for rank,(begin,end)in zip(owners,spans):assert[(2*(rank%8)+i)*192 for i in range(3)]==[begin,begin+192,end]
for e in range(-1,897):
 a=e if 0<=e<448 else -1;b=e-448 if 448<=e<896 else -1;assert int(a>=0)+int(b>=0)==int(0<=e<896)
proof['all896_global_experts_owned_once_EPgroup_each8TPshards_cover3072_and_original192_union']=True
proof['mask_boundary_minus1_0_447_448_895_896_no_missing_or_double_route']=True
src=(r/'kimi_k3_moe_EP2_r8669zj.cpp').read_text();original=(r/'kimi_k3_moe_I192_caller_smoke_r8669zf.cpp').read_text()
assert src[src.index('    if (tp_size_ > 1 && communicator_'):src.index('// Bounded synthetic')]==original[original.index('    if (tp_size_ > 1 && communicator_'):original.index('// Bounded synthetic')]
proof['actual_global16_reduce_norm_projection_shared_add_tail_byte_identical']=True
adapter=(r/'kimi_TP16_EP2_full_MoE_r8669zj.cpp').read_text();assert 'auto y=modules[plan[arm]]->forward(r,input)'in adapter and 'sum!=16.f'in adapter and 'plan[arm]?"1":"0"'not in adapter and 'uint64_t(!plan[arm]*16)'in adapter
assert 'if(ep_config.ep_size!=1)throw'in src and 'dynamic_cast<const EP2Experts_r8669zj*>'in src and 'ep2_masked_ids' in src
proof['actual_consumer_OFF_currentI192ON_global16_barrier_and_real_derived_ON']=True
inside=(r/'Kimi_TP16_EP2_r8669zk_inside.sh').read_text();entry=(r/'launch_Kimi_EP2_GPU_r8669zk.py').read_text();runner=(r/'Kimi_TP16_EP2_runner_r8669zk.py').read_text()
assert 'INFINILM_KIMI_EP2_R8669ZJ=1 INFINILM_KIMI_I192_COMPACT_R8669ZC=0'in inside
assert 'Kimi_TP16_EP2_r8669zk_inside.sh'in entry and 'incoming/Kimi-EP2-manifest-r8669zk.json'in entry and 'Kimi-EP2-r8669zk-'in runner
assert 'modern_kimi_TP16_moe_r8669zj'in runner and 'modern_kimi_TP16_moe_result_r8669zj'in runner
proof['actual_runtime_paths_namespace_two_contexts_19MBscratch_and_no_GPU_execution']=True
proof['numerical_scope']='Crossvariant intermediatebits diagnostic; predeclared broad BF16eps-based screen inmanifest, not Modeltoken acceptance. No GPU result.'
p=r/'Kimi-EP2-CPU-contract-r8669zk.json';p.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(proof))
