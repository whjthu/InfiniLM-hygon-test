"""Pure CPU closed-source/DSO fingerprint gate; no Torch/dlopen/weight IO."""
import pathlib,json,hashlib,ast
r=pathlib.Path('/refactor')
def validate():
 p=r/'incoming/GLM-transient-Model-source-manifest-r5694.json';m=json.loads(p.read_text())
 assert m['model_SHA256']=='a66a1f79737e43e8293d2d4e1a597b9f4dfd01ea38c70d2856b408b9c09ef4e6'
 for name,h in m['source_SHA256'].items():
  source=r/'incoming'/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==h,('source',name)
  if name.endswith('.py'):ast.parse(source.read_text(encoding='utf-8-sig'))
 for name,h in m['DSO_SHA256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,('DSO',name)
 assert m['TP']==8 and m['PP']==2 and m['capacity']==640 and m['active']==16 and m['performance_micro']==8
 assert m['scratch_physical_budget_bytes']==67108864 and m['reserve_bytes']==6*1024**3 and m['runtime_floor_bytes']==512*1024**2
 return {'passed':True,'manifest_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_SHA256':m['source_SHA256'],'DSO_SHA256':m['DSO_SHA256']}
if __name__=='__main__':print(json.dumps(validate()))
