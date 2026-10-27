import pathlib,json,hashlib
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');logical='/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM/csrc/'
names=[logical+'engine/rank_worker.hpp',logical+'engine/rank_worker.cpp',logical+'engine/infer_engine.hpp',logical+'models/infinilm_model.hpp','/refactor/candidates/deep-c4-projection-model-r5920/incoming/modern_native_engine_probe_r22.cpp']
out={'read_only':True,'files':{}}
for n in names:
 p=pathlib.Path(n.replace('/refactor',str(R),1));b=p.read_bytes();assert len(b)<150000
 out['files'][n]={'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
print(json.dumps(out))
