"""Local stdlib AST and real64 schedule contract; no model/GPU imports."""
import pathlib,json,ast,hashlib,importlib.util,sys
D=pathlib.Path(__file__).resolve().parent;OLD=pathlib.Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/Deep-C4-independent64-private-r9278');spec=importlib.util.spec_from_file_location('frontier158CPU',D/'bootstrap_r158.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);raw=(OLD/'modern_Deep_independent64_r9278.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==mod.BASE_SHA;tree=mod.transform(raw)
def node(src,name):return ast.dump(next(n for n in ast.walk(ast.parse(src))if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
assert node((OLD/'Deep_independent64_r9278.py').read_bytes(),'generate')==node((D/'Deep_batch_frontier_r158.py').read_bytes(),'generate')
ids=[v['id']for v in json.loads((D/'dataset64.json').read_bytes())['cases']];assert len(ids)==len(set(ids))==64;grid={}
for batch in [32,8,16,4]:
 groups=[ids[i:i+batch]for i in range(0,len(ids),batch)];assert all(len(x)==batch for x in groups)and sum(groups,[])==ids;grid[str(batch)]={'groups':len(groups),'executed_ids_if_run':64,'no_duplicate_id_within_batch_arm':True}
assert 'torch'not in sys.modules and 'numpy'not in sys.modules
q=json.loads((D/'CPU-local-contract-r158.json').read_bytes());q.update(local_static_execution_passed=True,original_driver_AST_transform_executed_without_import=True,real64_schedule=grid,Torch_imported=False,Native_dlopen=False,actual_GPU_executed=False);(D/'CPU-local-contract-r158.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'passed':True,'matrix':grid,'Torch_imported':False,'GPU':False}))
