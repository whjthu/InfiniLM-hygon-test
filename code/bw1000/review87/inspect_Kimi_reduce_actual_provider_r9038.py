import pathlib,json,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={}
rels=['candidates/Kimi-target-taps-r9022/libKimi_target_taps_r9022.so','candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/_infinilm.cpython-310-x86_64-linux-gnu.so','candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so','candidates/Kimi-layer0-TP16-replay-r9038b/libKimi_layer0_r9038b.so']
for rel in rels:
 p=r/rel;nm=subprocess.check_output(['nm','-D',str(p)],text=True);q[rel]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'reduce_symbols':[x for x in nm.splitlines()if 'modern_kimi_reduce_'in x],'needed':[x.strip()for x in subprocess.check_output(['readelf','-d',str(p)],text=True).splitlines()if 'NEEDED'in x]}
for rel in ['candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine/distributed/kimi_reduce_precision_r3226.cpp','candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/csrc/engine/distributed/kimi_reduce_precision_r3226.cpp']:
 p=r/rel
 if p.exists():q[rel]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(q))
