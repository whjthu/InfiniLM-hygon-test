"""Private TP32 entry. Rejects unknown full working set before importing Torch."""
import json,hashlib,pathlib
GIB=1024**3
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def validate(m):
 assert m['schema']==1 and m['served_layers']==93 and m['topology']=={'tp_nodes':4,'pp':1,'logical_capacity':8}
 assert m['nodes']==['a02r4n08','a02r4n09','a02r4n10','a02r4n11']
 b=m['static_memory'];assert b['PP_ranges']==[[0,93]]and b['WORKSPACE_OFF']is True and b['static_persistent_bytes']==63250309692
 # Neither a logical static fit nor a no-weight subset constructor authorizes a
 # full93 parameter constructor. Every bound must have independent evidence.
 g=m['full_working_set_gate'];assert g['qualified']is True and g['unknown_terms']==[] and g['source_published']is True and g['root_full_native_GO']is True
 assert g['actual_total_memory_bytes']>0 and g['driver_CCL_allocator_peak_bound_bytes']>0 and g['simultaneously_live_transient_bound_bytes']>0
 assert g['actual_total_memory_bytes']-b['static_persistent_bytes']-g['driver_CCL_allocator_peak_bound_bytes']-g['simultaneously_live_transient_bound_bytes']>4*GIB
 assert {'target_model.py','entry_contract_r9143.py','shard_head_contract_r9183.py','parameter_loader_r9143.py','weight_key_guard_r9156.py','batch_native_r9128.py'}<={pathlib.Path(p).name for p in m['source_SHA256']}
 return m['topology']
def geometry(cap):
 assert cap==8;return {'max_rows':16,'pages':68,'page_size':256,'KDA_pool_rows':17,'pages_per_request':4,'max_total_tokens':1024}
def memory_gate(packets):
 vals=[x[0]for p in packets for x in p['memory']];assert len(vals)==8*len(packets)and min(vals)>4*GIB;return min(vals)
def verify_inputs(m):
 for p,h in m['source_SHA256'].items():assert sha(p)==h,p
 model=pathlib.Path(m['model_view']['path']);assert sha(model/'config.json')==m['model_view']['config_SHA256']and sha(model/'model.safetensors.index.json')==m['model_view']['index_SHA256']
 b=json.loads(pathlib.Path(m['native_receipt']).read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen']and b['HOST_TUs_compiled']==5 and b['GPU_TUs_compiled']==0
 assert b['native_SHA256']==m['native_DSO_SHA256']==sha(b['native'])
 for p,h in b['all_linked_object_SHA256'].items():assert sha(p)==h,p
 return model,b
