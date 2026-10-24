import ast,hashlib,json,pathlib
r=pathlib.Path('.');s=(r/'build_GLM_comm_bridge_r8784.py').read_text()
s=s.replace("b=r/'candidates/glm-comm-only-r8784'","b=r/'candidates/glm-constructor-only-r8791'")
s=s.replace("r/'candidates/glm-PP3-B64-r8780/CPU-build.json'","r/'candidates/glm-comm-only-r8784/CPU-build.json'")
s=s.replace('99b99b3ab56de735cbff177df532af0a7ce4836d6af25ce5109dfc5c997d4862','0a5c8f885932b210ce2cad0a227a74e564219d92dbedfeb4c44f78e7357c34d8')
s=s.replace('modern_glm_comm_bridge_r8783.cpp','modern_glm_constructor_bridge_r8790.cpp').replace('GLM-comm-bridge-source-r8783.json','GLM-constructor-bridge-source-r8790.json')
s=s.replace('candidates/glm-PP3-B64-r8780/tree/candidates/glm-window-model-r5797/src/modern_glm_cached_model_bridge_r809.cpp','candidates/glm-comm-only-r8784/modern_glm_comm_bridge_r8783.cpp')
s=s.replace('candidates/glm-PP3-B64-r8780/private-2.o','candidates/glm-comm-only-r8784/bridge.o').replace('libglm_comm_only_r8784.so','libglm_constructor_only_r8791.so')
s=s.replace("'modern_glm_comm_only_destroy_r8783']","'modern_glm_comm_only_destroy_r8783','modern_glm_constructor_only_create_r8790','modern_glm_constructor_only_destroy_r8790']")
(r/'build_GLM_constructor_bridge_r8791.py').write_bytes(s.encode())
launcher=(r/'launch_GLM_comm_CPU_r8784.py').read_text().replace('r8784','r8791').replace('GLM-comm-only-CPU','GLM-constructor-only-CPU').replace('GLM_comm_CPU','GLM_constructor_CPU')
(r/'launch_GLM_constructor_CPU_r8791.py').write_bytes(launcher.encode())
inside=(r/'GLM_comm_CPU_r8784_inside.sh').read_text().replace('build_GLM_comm_bridge_r8784.py','build_GLM_constructor_bridge_r8791.py');(r/'GLM_constructor_CPU_r8791_inside.sh').write_bytes(inside.encode())
names=['modern_glm_constructor_bridge_r8790.cpp','GLM-constructor-bridge-source-r8790.json','build_GLM_constructor_bridge_r8791.py','GLM_constructor_CPU_r8791_inside.sh','launch_GLM_constructor_CPU_r8791.py']
for n in names:
 p=r/n;p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 if p.suffix=='.py':ast.parse(p.read_text(),filename=n)
p=r/'GLM-constructor-only-CPU-manifest-r8791.json';p.write_bytes((json.dumps({'scope':'53conditional onebridgeTU+27reused; originalModelcreate and531comm-only preserved; noGPU/dlopen/weights/forward','sources_SHA256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}},indent=2)+'\n').encode())
print(json.dumps({'files':names+[p.name],'manifest_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
