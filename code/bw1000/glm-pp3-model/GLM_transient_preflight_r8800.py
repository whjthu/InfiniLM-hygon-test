"""Closed CPU preflight; no Torch/dlopen/tensor weight IO. Header metadata only."""
import pathlib,json,hashlib,ast,struct
from glm_crossrequest_contract_r8800 import validate_fixture,I512_SHA256
r=pathlib.Path('/refactor')
def validate():
 p=r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r8800.json';m=json.loads(p.read_text())
 assert m['model_SHA256']=='bbda28394742232a977556f246c38e96a69d3eb365ab7845220d745960575819'
 receipt=r/'candidates/glm-PP3-Model-r8801/CPU-build.json';proof=json.loads(receipt.read_text());assert proof['completed']and proof['library_SHA256']==m['model_SHA256']and len(proof['reused_objects_SHA256'])==27
 assert hashlib.sha256(pathlib.Path(proof['library']).read_bytes()).hexdigest()==m['model_SHA256']
 assert (m['TP'],m['PP'],m['capacity'],m['active'],m['constructor_rows'],m['performance_micro'])==(8,3,640,64,64,8)
 assert (m['resident_physical_bytes_per_layer'],m['reserve_bytes'],m['allowance_bytes'],m['runtime_floor_bytes'])==(512*1024**2,6*1024**3,64*1024**2,6*1024**3)
 for name,h in m['source_SHA256'].items():
  source=r/'incoming'/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==h,('source',name)
  if name.endswith('.py'):ast.parse(source.read_text(encoding='utf-8-sig'))
 for name,h in m['DSO_SHA256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,('DSO',name)
 probe=ast.parse((r/'incoming/glm_prefill_micro_probe_r8800.py').read_text())
 assignment=next(n for n in ast.walk(probe)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.slice,ast.Constant)and t.slice.value=='probe_source_sha256'for t in n.targets))
 refs=[n.value for n in ast.walk(assignment.value)if isinstance(n,ast.Constant)and isinstance(n.value,str)and n.value.endswith('.py')]
 assert refs and all(n in m['source_SHA256']and(r/'incoming'/n).is_file()for n in refs),refs
 fixture=r/'results/glm-I512-real-inputs-r5713.json';assert hashlib.sha256(fixture.read_bytes()).hexdigest()==I512_SHA256
 f=json.loads(fixture.read_text());validate_fixture(f);assert f['provenance']==json.loads((r/'results/glm-chat-template-r20.json').read_text())['provenance']
 # Existing loader's exact supported dtype path: F8->U8 view, F32/BF16 untouched.
 model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];dtypes={};keys=set()
 for name in sorted(set(index.values())):
  with(model/name).open('rb')as stream:
   size=struct.unpack('<Q',stream.read(8))[0];assert 0<size<64*1024**2;h=json.loads(stream.read(size))
  for key,v in h.items():
   if key=='__metadata__':continue
   assert v['dtype']in('F32','BF16','F8_E4M3'),(key,v['dtype']);dtypes[v['dtype']]=dtypes.get(v['dtype'],0)+1;keys.add(key)
 assert keys==set(index),'checkpoint header/index keys mismatch'
 return {'passed':True,'manifest_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_SHA256':m['source_SHA256'],'DSO_SHA256':m['DSO_SHA256'],'I512_fixture_SHA256':I512_SHA256,'weight_metadata_dtypes':dtypes,'no_Torch_dlopen_or_weight_tensor_IO':True}
if __name__=='__main__':print(json.dumps(validate()))
