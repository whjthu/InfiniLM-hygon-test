import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent;old=r/'GLM-DP-Model-private-r8864';d=r/'GLM-DP-Mmatrix-private-r8868';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mh=sha(d/'GLM-CROSSREQUEST-Model-source-manifest-r8868.json');eh=sha(d/'dispatch_GLM_resident_exact_Model_r8868.py');oldmh=sha(old/'GLM-CROSSREQUEST-Model-source-manifest-r8864.json');oldeh=sha(old/'dispatch_GLM_resident_exact_Model_r8864.py')
for stem in('stage_GLM_DP_Model','invoke_GLM_DP_Model','upload_GLM_DP_Model','observe_GLM_DP_Model'):
 raw=(r/(stem+'_r8864.py')).read_bytes().replace(b'r8864',b'r8868').replace(b'GLM-DP-Model-private-r8868',b'GLM-DP-Mmatrix-private-r8868').replace(oldmh.encode(),mh.encode()).replace(oldeh.encode(),eh.encode())
 if stem=='stage_GLM_DP_Model':raw=raw.replace(b"len(c['compiled'])==1 and len(c['reused_objects_SHA256'])==27",b"len(c['compiled'])==7 and len(c['reused_objects_SHA256'])==21").replace(b"m['decode_micro_fixed']==8",b"m['decode_micro_candidates']==[8,16]")
 (r/(stem+'_r8868.py')).write_bytes(raw)
print(json.dumps({'GPU':False,'manifest_SHA256':mh,'entry_SHA256':eh,'shared_helpers_upload':False}))
