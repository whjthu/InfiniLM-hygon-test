import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');logical='/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM/csrc/engine/infer_engine.cpp'
p=pathlib.Path(logical.replace('/refactor',str(r),1));b=p.read_bytes();assert len(b)<200000
print(json.dumps({'read_only':True,'files':{logical:{'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()}}}))
