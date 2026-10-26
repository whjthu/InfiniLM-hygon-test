import pathlib,json,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
def host(p):
 p=pathlib.PurePosixPath(p);return r.joinpath(*p.relative_to('/refactor').parts)
targets=['/refactor/candidates/deep-owner-device-r8918/src/deepseek_v4_native_routed_decoder_r20.cpp','/refactor/candidates/deep-owner-device-r8918/src/deepseek_v4_ffn_routed_block_r20.cpp','/refactor/candidates/deep-owner-device-r8918/src/deep_model_owner_r8900.cpp','/refactor/candidates/deep-combined-r7169/compile.sh','/refactor/candidates/deep-c4-projection-model-r5920/compile.sh'];o={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU':False,'files':[],'nearby_sources':[]}
for n in targets:
 p=host(n)
 if p.exists():
  b=p.read_bytes();assert len(b)<250000;o['files'].append({'container_path':n,'host_path':str(p),'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()})
 else:o['files'].append({'container_path':n,'exists':False})
for base in ['deep-owner-device-r8918','deep-combined-r7169','deep-c4-projection-model-r5920']:
 for folder in [r/'candidates'/base,r/'candidates'/base/'src',r/'candidates'/base/'incoming']:
  if folder.is_dir():o['nearby_sources'] += [str(p.relative_to(r))for p in folder.iterdir()if p.is_file()and p.suffix in ['.cpp','.hpp','.h','.sh']]
print(json.dumps(o))
