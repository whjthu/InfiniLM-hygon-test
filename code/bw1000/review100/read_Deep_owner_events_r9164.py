import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';ps=[r/'candidates/deep-owner-device-r8918/src/deep_owner_events_r8900.hpp',r/'candidates/deep-owner-events-r8900/src/deep_owner_events_r8900.hpp',r/'candidates/deep-c4-projection-model-r5920/incoming/deep_owner_events_r8900.hpp'];o=[]
for p in ps:
 if p.exists():b=p.read_bytes();o.append({'path':str(p),'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()})
assert o,'profileheader notatknowncompiledpaths';print(json.dumps({'CPU_only':True,'GPU':False,'files':o}))
