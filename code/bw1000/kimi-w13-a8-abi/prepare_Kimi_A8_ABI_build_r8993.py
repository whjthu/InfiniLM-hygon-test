import pathlib,hashlib,json,ast
r=pathlib.Path.cwd();private=r/'Kimi-W13-A8-ABI-private-r8993'
s=(r/'Kimi_W13_A8_CPU_r8991_inside.sh').read_text().replace('build_Kimi_W13_A8_CPU_r8991.py','build_Kimi_A8_ABI_CPU_r8993.py');(r/'Kimi_A8_ABI_CPU_r8993_inside.sh').write_bytes(s.encode())
s=(r/'launch_Kimi_W13_A8_CPU_r8991.py').read_text().replace('Kimi-W13-A8-CPU-r8991','Kimi-A8-ABI-CPU-r8993').replace('Kimi_W13_A8_CPU_r8991_inside.sh','Kimi_A8_ABI_CPU_r8993_inside.sh');ast.parse(s);(r/'launch_Kimi_A8_ABI_CPU_r8993.py').write_bytes(s.encode())
names=[private/n for n in ['kimi_A8_matrix_r8993.hip','kimi_A8_adapter_r8993.cu','Kimi_A8_ABI_real_r8993.py']]+[r/n for n in ['build_Kimi_A8_ABI_CPU_r8993.py','Kimi_A8_ABI_CPU_r8993_inside.sh','launch_Kimi_A8_ABI_CPU_r8993.py']]
proof={'source_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in names},'scope':'Explicit root ABI-isolation fix after235 STOP; one pureHIP+oneCUDAadapter; noGPU/noDLopen; defaultstream0 only; no runtime-ordering claim'}
(r/'Kimi-A8-ABI-source-r8993.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
s=(r/'stage_Kimi_W13_A8_CPU_r8991.py').read_text().replace('Kimi-W13-A8-source-r8991.json','Kimi-A8-ABI-source-r8993.json').replace('launch_Kimi_W13_A8_CPU_r8991.py','launch_Kimi_A8_ABI_CPU_r8993.py').replace('Kimi-W13-A8-private-r8991','Kimi-W13-A8-ABI-private-r8993');ast.parse(s);(r/'stage_Kimi_A8_ABI_CPU_r8993.py').write_bytes(s.encode())
s=(r/'read_Kimi_W13_A8_HIP_CPU_r8992.py').read_text().replace('Kimi-W13-A8-HIP-CPU-r8992','Kimi-A8-ABI-CPU-r8993');(r/'read_Kimi_A8_ABI_CPU_r8993.py').write_bytes(s.encode())
for p in names:
 if p.suffix=='.py':ast.parse(p.read_text())
print(json.dumps(proof,indent=2))
