import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={'scope':'namedCPUsource/binarymetadata only; no GPU/dlopen/weights'}
for rel in ['candidates/kimi-helper-bounded-bf16-r6054/build.json','candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so','candidates/models-5f91ad7-r3675/Kimi/helpers/libkimi_grouped_prefill.so','incoming/kimi_helper_bounded_bf16_r6054.cpp','candidates/models-5f91ad7-r3675/Kimi/helpers/src/kimi_grouped_fused_fp32_runtime_r21.cpp','candidates/Kimi-I192-batch-Model-r9129/kimi_moe_batch_r9129.cpp','candidates/Kimi-ctx2048-r123e/CPU-build.json','candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/csrc/runtime/runtime.cc','candidates/upstream-5f91ad7-r3667/InfiniTensor-InfiniRT-cd1d48258f0a/src/reusable_memory_pool.cc']:
 p=r/rel
 if not p.exists():out[rel]={'exists':False};continue
 v={'exists':True,'SHA256':sha(p),'bytes':p.stat().st_size}
 if p.suffix=='.json':
  d=json.loads(p.read_bytes());v['selected_metadata']={k:d.get(k)for k in ['completed','helper_sha256','source_sha256','native_SHA256','new_GPU_TUs','actual_step']}
 if p.suffix in ['.cpp','.cc']:
  lines=p.read_text().splitlines();wanted=['struct Work','contributions=','modern_grouped_mxfp4_owned','trimMemory','ReusableMemoryPool::trim','pin_count == 0','!(*it)->in_use'];v['source_anchors']=[{'line':i+1,'text':s[:1600]}for i,s in enumerate(lines)if any(t in s for t in wanted)][:16]
 out[rel]=v
out['actual478_helper_path']='UNRECORDED: completedtarget maps saved onlyfour corelibraries; LD precedence alone is not dladdr proof'
print(json.dumps(out))
