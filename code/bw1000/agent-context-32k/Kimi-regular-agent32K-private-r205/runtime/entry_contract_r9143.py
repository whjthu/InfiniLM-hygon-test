import json,hashlib,pathlib
GIB=1024**3

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def validate(m):
 assert isinstance(m['frontend_directory'],str) and m['frontend_directory'].startswith('/')
 assert m['schema']==1 and m['served_layers']==93 and m['topology']['tp_nodes']==2
 t=m['topology'];assert t['pp']in [2,4] and t['logical_capacity']==1 and len(m['nodes'])==2*t['pp'] and len(set(m['nodes']))==len(m['nodes'])
 b=m['static_memory'];assert b['served_layers']==93 and b['config_SHA256']==m['model_view']['config_SHA256'] and b['index_SHA256']==m['model_view']['index_SHA256'] and len(b['stages'])==t['pp']
 assert [i for a,z in b['PP_ranges']for i in range(a,z)]==list(range(93))
 assert b['PP_ranges']==([[0,46],[46,93]]if t['pp']==2 else[[0,24],[24,47],[47,71],[71,93]]),'actual native964 private ServingMoE084ab partition'
 assert all(v['parameter_plus_reserve_fits']and v['planning_reserve_bytes']>=(12 if t['pp']==4 else 6)*GIB for v in b['stages'])
 return t

def geometry(cap):
 assert cap==1;return {'max_rows':2,'pages':260,'page_size':256,'KDA_pool_rows':3,'pages_per_request':128,'max_total_tokens':32768}
def memory_gate(packets):
 vals=[pair[0]for packet in packets for pair in packet['memory']];assert len(vals)==8*len(packets) and min(vals)>4*GIB,'4GiB allowner reserve unchanged';return min(vals)

def verify_inputs(m):
 for path,h in m['source_SHA256'].items():assert sha(path)==h,path
 model=pathlib.Path(m['model_view']['path']);assert sha(model/'config.json')==m['model_view']['config_SHA256'] and sha(model/'model.safetensors.index.json')==m['model_view']['index_SHA256'];assert sha(m['native_receipt'])==m['native_receipt_SHA256'];b=json.loads(pathlib.Path(m['native_receipt']).read_text());assert b['completed']and b['CPU_only']and not b['GPU']and not b['Native_dlopen'];assert b['native_SHA256']==m['native_DSO_SHA256']and sha(b['native'])==m['native_DSO_SHA256']
 for path,h in b['all_linked_object_SHA256'].items():assert sha(path)==h,path
 return model,b
