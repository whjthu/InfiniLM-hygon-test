import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=r/'incoming/Kimi-W2-INT8-register-v2-r113c';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=D/'GPU-manifest-source-r113v2b.json';launcher=D/'launch_GPU_r113v2.py'
assert sha(m)=='7c7c16fe53b8cab4c27e2a9c18a121fddf0d2c5f70261d63d572bd12186d34f5';assert sha(launcher)=='a4ca3dfbff01cd842ed03e5fa8c4f32b5a97fb6c89a52b0e247a7f9b7f934564';assert not(r/'results/Kimi-W2-INT8-register-v2-r113c-keeper-launch.json').exists()
go={'authorized_by':'parent-root','manifest_SHA256':sha(m),'Git_publication_SHA':'4af2ba28b1331286fc3a268d9ea9e7f5fbc33ed8','scope':'one16card v2registerscaleI192W2 fullMoEcomponent samehistoricalT8/T64+constructedlowreuse, originalwarmCPUintegeroracle; noModel/compile/shape/profile/grid'};p=D/'GPU-GO-after-source-publication-r113v2b.json';raw=(json.dumps(go,indent=2)+'\n').encode()
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
z=subprocess.run(['python3',str(launcher)],text=True,capture_output=True,timeout=40);assert z.returncode==0,(z.stdout,z.stderr);print(z.stdout)
