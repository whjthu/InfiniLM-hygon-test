import pathlib,json,hashlib,ast,zlib,base64,shutil
r=pathlib.Path(__file__).resolve().parent;b=r/'Kimi-KDA-only-serial-packed-private-r9025';old=r/'Kimi-KDA-only-serial-private-r9020';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=json.loads((b/'CPU-observed-a.json').read_text(encoding='utf8'));cpu=next(iter(q['receipts'].values()));assert cpu['completed']and cpu['metadata_result']['only_packed_out_supports']and not cpu['metadata_result']['original_narrow_supports'];(b/'CPU-build.json').write_bytes((json.dumps(cpu,indent=2)+'\n').encode());shutil.copyfile(r/'Kimi-KDA-only-serial-packed-private-r9020c/kda_conv_native.cpp',b/'kda_conv_native.cpp');assert sha(b/'kda_conv_native.cpp')==cpu['source_SHA256']
for n in ['controller_gpu.py','launch_gpu.py']:
 s=(old/n).read_text();fields={"tag='Kimi-KDA-only-serial-r9020'":"tag='Kimi-KDA-only-serial-packed-r9025'","'Kimi-serial-manifest-r9020.json'":"'Kimi-serial-packed-manifest-r9025.json'","'Kimi-serial-published-r9020.json'":"'Kimi-serial-rootGO-r9025.json'","public['ordinary_push_success']":"public['root_continuous_GO']","wave='Kimi_serial_GPU_r9020_inside.sh'":"wave='Kimi_serial_packed_GPU_r9025_inside.sh'","'controller_Kimi_serial_r9020.py'":"'controller_Kimi_serial_packed_r9025.py'","'--job-name=Kimi-KDA-only-serial-post-r9020'":"'--job-name=Kimi-KDA-packed-post-r9025'","'--job-name=Kimi-KDA-only-serial-idle-r9020'":"'--job-name=Kimi-KDA-packed-idle-r9025'"}
 for x,y in fields.items():s=s.replace(x,y)
 ast.parse(s);(b/n).write_bytes(s.encode())
baseinside=(old/'GPU_inside.sh').read_text();checks=baseinside[baseinside.index("python - <<'PY'"):baseinside.index('timeout --kill-after')];checks=checks.replace("'incoming/Kimi-serial-manifest-r9020.json'","'incoming/Kimi-serial-packed-manifest-r9025.json'")
prefix=(r/'Kimi-ordinary-pair-private-r9007/Kimi_pair_Model_r9007_inside.sh').read_text().split('unset NCCL_DMABUF_ENABLE',1)[0]
inside=prefix+'export OMP_NUM_THREADS=1 INFINILM_OWNER_CPU_WIDTH_R2392=1\n'+checks+'timeout --kill-after=10 210 python -u /refactor/incoming/check_Kimi_serial_r9020.py --gpu-lib '+cpu['library']+' --root-go --out "/refactor/results/Kimi-KDA-only-serial-packed-r9025-${SLURM_JOB_ID}.${SLURM_STEP_ID}.json"\n';(b/'GPU_inside.sh').write_bytes(inside.encode())
m=json.loads((old/'Kimi-serial-manifest-r9020.json').read_text());m.update(status='ROOT_K82_A_CONTINUOUS_GO_CPU252_SUPPORT_CLOSED',native=cpu['library'].removeprefix('/refactor/'),native_SHA256=cpu['library_SHA256'],WAVE_INSIDE='Kimi_serial_packed_GPU_r9025_inside.sh',CPU_step=cpu['producer'],metadata_check=cpu['metadata_result'],source_SHA256=cpu['source_SHA256'],source_base_commit='cd2440c40059db6b34354ff86b09521c7b36ecd9',publication='private per root explicit continuous GO; root publishes after result',previous_negative='982 original narrowout rejected; 242/243/244/247 CPU environments preserved')
files={'check_Kimi_serial_r9020.py':old/'check_serial_component.py','lifecycle_base_r9020.py':old/'lifecycle_base_r9020.py','controller_Kimi_serial_packed_r9025.py':b/'controller_gpu.py','launch_Kimi_serial_packed_r9025.py':b/'launch_gpu.py','Kimi_serial_packed_GPU_r9025_inside.sh':b/'GPU_inside.sh'}
m['sources_SHA256']={n:sha(p)for n,p in files.items()};assert m['WAVE_INSIDE']in files and "wave='"+m['WAVE_INSIDE']+"'"in(b/'launch_gpu.py').read_text();assert cpu['library']in inside
(b/'Kimi-serial-packed-manifest-r9025.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());files['Kimi-serial-packed-manifest-r9025.json']=b/'Kimi-serial-packed-manifest-r9025.json'
payload=base64.b64encode(zlib.compress(json.dumps({n:p.read_text()for n,p in files.items()}).encode())).decode()
remote="""import pathlib,json,zlib,base64,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n
 if p.exists():assert p.read_bytes()==s.encode(),n
 else:p.write_bytes(s.encode())
m=json.loads((i/'Kimi-serial-packed-manifest-r9025.json').read_text())
for n,h in m['sources_SHA256'].items():assert hashlib.sha256((i/n).read_bytes()).hexdigest()==h,n
assert hashlib.sha256((r/m['native']).read_bytes()).hexdigest()==m['native_SHA256']
p=i/'Kimi-serial-rootGO-r9025.json';assert not p.exists();p.write_text(json.dumps({'root_continuous_GO':True,'commit':'cd2440c40059db6b34354ff86b09521c7b36ecd9','private_source_SHA256':m['source_SHA256'],'authorization':'Astra82 K82-A root NEW_TASK, no new Model GO'}))
p=subprocess.run(['python3',str(i/'launch_Kimi_serial_packed_r9025.py')],text=True,capture_output=True,timeout=180);assert p.returncode==0,(p.stdout,p.stderr);print(p.stdout)
""".replace('PAYLOAD',repr(payload));compile(remote,'invoke','exec');(r/'invoke_Kimi_serial_packed_GPU_r9025.py').write_bytes(remote.encode());print(json.dumps({'CPU':cpu['producer'],'metadata':cpu['metadata_result'],'WAVE_chain':'PASS','native':cpu['library_SHA256']}))
