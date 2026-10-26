import pathlib,json,sys,runpy,os,socket,hashlib
D=pathlib.Path(__file__).resolve().parent.parent;runtime=D/'runtime';sys.path.insert(0,str(runtime))
runpy.run_path(str(runtime/'preflight_r9183.py'),run_name='__main__')
from entry_contract_r9143 import validate,geometry
m=json.loads((runtime/'model-manifest-r9183.json').read_bytes());assert validate(m)=={'tp_nodes':2,'pp':2,'logical_capacity':8}
expected={'max_rows':16,'pages':68,'page_size':256,'KDA_pool_rows':17,'pages_per_request':4,'max_total_tokens':1024};assert geometry(8)==expected
assert m['prefill_micro_request_cap']==1 and m['memory_geometry_explicit']['shadow_bank_offset']==8 and m['memory_geometry_explicit']['hard_minimum_free_bytes']==4294967296
source=(runtime/'target_model.py').read_bytes().decode();assert "'logical_capacity': 8"in source and 'ctx=lib.modern_kimi_engine_create_batch_r9122c(json.dumps(cfg).encode(),node,tp_nodes,pp,head.encode(),port,capacity)'in source
assert 'torch'not in sys.modules
print(json.dumps({'r9293_entry_closure_passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'native_dlopen':False,'node':int(os.environ['SLURM_PROCID']),'host':socket.gethostname(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'geometry_source_validated':expected,'logical_capacity_argument':8,'prefill_micro_request_cap':1,'actual_GPU_geometry_qualification':False}),flush=True)
