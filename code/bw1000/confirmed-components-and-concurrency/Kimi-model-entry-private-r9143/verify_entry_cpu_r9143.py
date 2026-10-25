import json,pathlib,types,copy,math
from entry_contract_r9143 import validate,geometry,memory_gate,GIB
from parameter_loader_r9143 import load_parameters
cases=[]
for pp in [2,4]:
 for cap in [8,16,32]:
  ranges=[[0,46],[46,93]]if pp==2 else[[0,22],[22,46],[46,70],[70,93]];m={'schema':1,'frontend_directory':'/CPU_ONLY_FAKE_FRONTEND','served_layers':93,'topology':{'tp_nodes':2,'pp':pp,'logical_capacity':cap},'nodes':['MOCKNODE%d'%i for i in range(2*pp)],'model_view':{'config_SHA256':'cfg','index_SHA256':'idx'},'static_memory':{'served_layers':93,'config_SHA256':'cfg','index_SHA256':'idx','PP_ranges':ranges,'stages':[{'parameter_plus_reserve_fits':True,'planning_reserve_bytes':6*GIB}for _ in range(pp)]}};assert validate(m)['logical_capacity']==cap;g=geometry(cap);assert g['max_rows']==2*cap and g['KDA_pool_rows']==2*cap+1 and g['pages']==4*(2*cap+1);packets=[{'memory':[[5*GIB,64*GIB]]*8}for _ in range(2*pp)];assert memory_gate(packets)==5*GIB;packets[-1]['memory']=packets[-1]['memory'][:-1]+[[4*GIB,64*GIB]]
  try:memory_gate(packets);raise RuntimeError('4GiBmustreject')
  except AssertionError:pass
  bad=copy.deepcopy(m);bad['static_memory']['served_layers']=94
  try:validate(bad);raise RuntimeError('94layerbudgetmustreject')
  except AssertionError:pass
  cases.append({'PP':pp,'capacity':cap,'geometry':g,'allowner4GiBstrict_and94view_reject':True})
class Tensor:
 def __init__(self,shape,dtype='BF16'):self.shape=shape;self.dtype=dtype
 def __getitem__(self,slice):return Tensor((96,),self.dtype)
 def contiguous(self):return self
 def clone(self):raise RuntimeError('clone forbidden')
 def data_ptr(self):return id(self)
 def numel(self):return math.prod(self.shape)
 def element_size(self):return 2
values={'layer.self_attn.A_log':Tensor((128,),'F32'),'x.weight':Tensor((2,3)),'y.weight':Tensor((4,5),'U8')};mapping={k:'a.safetensors'if k!='y.weight'else'b.safetensors'for k in values}
class Model:
 def __init__(self,name=''):self.name=name
 def __truediv__(self,name):return Model(name)
 def read_text(self):return json.dumps({'weight_map':mapping})
class Shard:
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def get_tensor(self,k):return values[k]
seen=[]
class Lib:
 def modern_kimi_engine_load_r2475(self,ctx,raw):seen.extend(json.loads(raw));return 0
 def modern_kimi_engine_finish_r2475(self,ctx):return 0
z=load_parameters(Lib(),1,Model(),{'linear_attn_config':{'num_heads':96}},set(values),lambda *a,**k:Shard(),types.SimpleNamespace(float32='F32',bfloat16='BF16',uint8='U8'),lambda code:None,lambda x:None);assert z['loaded_keys']==3 and z['loader_flushes']==2 and next(x for x in seen if x['name'].endswith('A_log'))['shape']==[96]
q={'CPU_only':True,'GPU':False,'dlopen':False,'Model_engine_created':False,'topology_capacity_cases':cases,'load_parameters_mock':'3keys2shards flush/no_clone/A_log trim96/typeddescriptors PASS','Model_runtime_qualified':False};pathlib.Path(__file__).with_name('CPU-entry-contract-r9143.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
