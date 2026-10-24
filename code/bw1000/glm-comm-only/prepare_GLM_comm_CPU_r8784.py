import ast,hashlib,json,pathlib
r=pathlib.Path('.');launcher=(r/'launch_GLM_PP3_CPU_r8780.py').read_text().replace('r8780','r8784').replace('GLM-PP3-B64-CPU','GLM-comm-only-CPU').replace('GLM_PP3_CPU','GLM_comm_CPU')
(r/'launch_GLM_comm_CPU_r8784.py').write_bytes(launcher.replace('\r\n','\n').encode())
inside=(r/'GLM_PP3_CPU_r8780_inside.sh').read_text().replace('build_GLM_PP3_B64_r8780.py','build_GLM_comm_bridge_r8784.py')
(r/'GLM_comm_CPU_r8784_inside.sh').write_bytes(inside.replace('\r\n','\n').encode())
names=['modern_glm_comm_bridge_r8783.cpp','GLM-comm-bridge-source-r8783.json','build_GLM_comm_bridge_r8784.py','GLM_comm_CPU_r8784_inside.sh','launch_GLM_comm_CPU_r8784.py']
for n in names:
 p=r/n;p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 if p.suffix=='.py':ast.parse(p.read_text(),filename=n)
m={'scope':'GLM53 onebridge hostTU comm-only; noGPU/SDKcompile/dlopen/params/weights/forward','sources_SHA256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}}
p=r/'GLM-comm-only-CPU-manifest-r8784.json';p.write_bytes((json.dumps(m,indent=2)+'\n').encode());print(json.dumps({'CPU_manifest_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':names+[p.name]}))
