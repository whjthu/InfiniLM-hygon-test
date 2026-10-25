import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');raw=r/'results/Kimi-layer0-raw-r9042-910375.693';q={'GPU':False,'new_capture':False,'points':{},'compiled_source':{},'driver_env_lines':[]}
for arm in ['A','B','C']:
 p=raw/arm/'stage0-rank0-layer0-kda-call25.jsonl';q['points'][arm]=[{'point':z['point'],'shape':z['shape'],'dtype':z['dtype']}for z in[json.loads(x)for x in p.read_text().splitlines()]]
p=r/'candidates/Kimi-rowreduce-group-r9042b/src/delta.cpp';s=p.read_text();q['compiled_source']={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'selection_trace_lines':[x for x in s.splitlines()if any(k in x for k in ['ROWGROUP','trace_kda_projection(runtime,output','GroupStart','GroupEnd'])]};p=r/'incoming/Kimi_layer0_driver_r9042.py';q['driver_env_lines']=[x for x in p.read_text().splitlines()if any(k in x for k in ['ROWGROUP','OPROJ','o_proj_local'])];print(json.dumps(q))
