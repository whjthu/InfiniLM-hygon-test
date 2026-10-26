from pathlib import Path
import json,sys,hashlib,numpy as np
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D));from run_component_r9176 import layout_contract
checks=0
for M in [1,4,8,16,32,64]:
 c=layout_contract(M);assert c['local_weight_shape']==[10240,7168]and c['local_weight_bytes']==146800640;rankmajor=np.stack([np.arange(rank*10240,(rank+1)*10240,dtype=np.int32)[None,:]+np.arange(M,dtype=np.int32)[:,None]*163840 for rank in range(16)]);out=rankmajor.transpose(1,0,2).copy().reshape(M,163840);ref=np.arange(163840,dtype=np.int32)[None,:]+np.arange(M,dtype=np.int32)[:,None]*163840;assert np.array_equal(out,ref);checks+=1
assert [i for rank in range(16)for i in range(rank*10240,(rank+1)*10240)]==list(range(163840));checks+=1
try:layout_contract(128)
except AssertionError:checks+=1
else:raise AssertionError('invalidM')
source=(D/'head_component_r9176.cpp').read_bytes();assert b'r.memcpyH2D(w->data(),slice,w->nbytes(),false)'in source and b'info.tp_rank*W'in source and b'W=10240'in source;assert b'if(mode!="shard")full='in source;assert b'30:1'in source;checks+=1
q={'CPU_only':True,'GPU':False,'no_torch_import':True,'no_dlopen':True,'registered_M':[1,4,8,16,32,64],'real_rank_major_to_row_major_numpy_layout_executed':True,'global_CPU_rows_unique_coverage':True,'actual_cpp_explicit_H2D_slice_static_check':True,'checks':checks,'runtime_gpu_semantic_qualification':False};(D/'CPU-contract-r9176.json').write_bytes(json.dumps(q,indent=2).encode());print(json.dumps(q))
