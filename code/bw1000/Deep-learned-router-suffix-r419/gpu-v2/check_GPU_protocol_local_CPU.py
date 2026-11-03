"""Execute the actual nested proof decoder, with fake device leaves only."""
from pathlib import Path
import ast,json,types
import numpy as np

def main():
 source=ast.parse((Path(__file__).parent/'GPU_driver.py').read_text())
 parent=next(x for x in source.body if isinstance(x,ast.FunctionDef)and x.name=='main')
 names={'decode','nr','snapshot','output_compare'};functions=[x for x in parent.body if isinstance(x,ast.FunctionDef)and x.name in names];assert len(functions)==4
 env={'np':np,'dtypes':{'F32':1,'I64':2,'BF16':3},'f':None,'check':None}
 fields={};env['api']=types.SimpleNamespace(proof=lambda f,check,ctx,k:fields[k])
 exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'actual_GPU_driver_functions','exec'),env)
 def fixture(tokens):
  def field(a,kind):return dict(shape=a.shape,dtype=env['dtypes'][kind],raw=np.frombuffer(a.tobytes(),np.uint8).copy())
  return {0:field(np.full((tokens,4096),0x3f80,np.uint16),'BF16'),1:field(np.ones((tokens,6),np.float32)/6,'F32'),2:field(np.tile(np.arange(6,dtype=np.int64),(tokens,1)),'I64')}
 for tokens in (1,2):
  fields.clear();fields.update(fixture(tokens));s=env['snapshot'](None);assert s['value'].shape==(tokens,4096)and np.all(s['value']==1)and env['output_compare'](s,s)==0
 rejected=[]
 for kind in ['dtype','shape','duplicate_ID','nonfinite_value','nonfinite_weights']:
  fields.clear();fields.update(fixture(1))
  if kind=='dtype':fields[1]['dtype']=999
  if kind=='shape':fields[2]['shape']=(6,)
  if kind=='duplicate_ID':fields[2]['raw'].view(np.int64)[0]=1
  if kind=='nonfinite_value':fields[0]['raw'].view(np.uint16)[0]=0x7fc0
  if kind=='nonfinite_weights':fields[1]['raw'].view(np.float32)[0]=np.nan
  try:env['snapshot'](None)
  except AssertionError:rejected.append(kind)
  else:raise AssertionError('Bad protocol accepted: '+kind)
 assert len(rejected)==5
 return dict(passed=True,actual_source_decoder=True,T1_and_untimed_T2_proof_shapes=True,BF16_bits_expanded_correctly=True,negative_cases_rejected=rejected,device_leaves_fake=True,whole_GPU_main_executed=False,GPU_qualification=False,jobs=0)

if __name__=='__main__':print(json.dumps(main()))
