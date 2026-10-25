import pathlib,json,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={'reports':[],'sources':{}};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run='batch-r9122b2-222b154b669f9165'
for node in range(4):
 p=r/'results'/f'kimi-batch-r9122b2-{run}-node{node}.json'
 if p.exists():
  d=json.loads(p.read_text());keys=['node','hostname','pid','phase','error','create_seconds','loading_seconds','loading_elapsed_seconds','loaded_keys','parameter_keys','memory_after_create','memory_after_finish','memory_after_finish_domains','CPU_copy_stats_after_loading','loader_flushes','source_parameter_keys_loaded','loader_transaction_policy','cgroup_probe','load_cgroup_probe','loading_cgroup_probe','preload_memory_gate'];z={k:d[k]for k in keys if k in d};z['memory_cgroup_fields']={k:v for k,v in d.items()if 'cgroup'in k.lower()};z['producer_path']=str(p);z['producer_SHA256']=sha(p);q['reports'].append(z)
 p=r/'logs'/f'Kimi-batch-TP16PP2-{run}-910380.407-{node}.log'
 if p.exists():q.setdefault('load_log_lines',{})[str(node)]={'path':str(p),'lines':[x for x in p.read_text(errors='replace').splitlines()if 'LOAD'in x or 'WEIGHTS_READY'in x][-20:]}
base=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/csrc'
for rel in ['engine/infer_engine.cpp','weights/weights.cpp','runtime/tensor.cpp','runtime/runtime.cpp']:
 p=base/rel
 if p.exists():q['sources'][str(p)]={'SHA256':sha(p),'text':p.read_text()}
print(json.dumps(q))
