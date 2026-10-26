import json,hashlib,pathlib
GIB=1024**3

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def validate(m):
 assert isinstance(m['frontend_directory'],str) and m['frontend_directory'].startswith('/')
 assert m['schema']==1 and m['served_layers']==93 and m['topology']['tp_nodes']==2
 t=m['topology'];assert t['pp']in [2,4] and t['logical_capacity']in [8,16,32] and len(m['nodes'])==2*t['pp'] and len(set(m['nodes']))==len(m['nodes'])
 b=m['static_memory'];assert b['served_layers']==93 and b['config_SHA256']==m['model_view']['config_SHA256'] and b['index_SHA256']==m['model_view']['index_SHA256'] and len(b['stages'])==t['pp']
 assert [i for a,z in b['PP_ranges']for i in range(a,z)]==list(range(93))
 assert b['PP_ranges']==([[0,46],[46,93]]if t['pp']==2 else[[0,22],[22,46],[46,70],[70,93]]),'actual93virtual95endpoint1 partition'
 assert all(v['parameter_plus_reserve_fits']and v['planning_reserve_bytes']>=(12 if t['pp']==4 else 6)*GIB for v in b['stages'])
 return t

def geometry(cap):
 assert cap in [8,16,32];physical=2*cap;return {'max_rows':physical,'pages':4*(physical+1),'page_size':256,'KDA_pool_rows':physical+1,'pages_per_request':4,'max_total_tokens':1024}
def memory_gate(packets):
 vals=[pair[0]for packet in packets for pair in packet['memory']];assert len(vals)==8*len(packets) and min(vals)>4*GIB,'4GiB allowner reserve unchanged';return min(vals)

def verify_inputs(m):
 for path,h in m['source_SHA256'].items():assert sha(path)==h,path
 model=pathlib.Path(m['model_view']['path']);assert sha(model/'config.json')==m['model_view']['config_SHA256'] and sha(model/'model.safetensors.index.json')==m['model_view']['index_SHA256'];b=json.loads(pathlib.Path(m['native_receipt']).read_text());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen'];assert b['native_SHA256']==m['native_DSO_SHA256']and sha(b['native'])==m['native_DSO_SHA256']
 for path,h in b['all_linked_object_SHA256'].items():assert sha(path)==h,path
 return model,b
