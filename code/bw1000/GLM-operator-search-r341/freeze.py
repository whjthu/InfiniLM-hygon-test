import pathlib,hashlib,json,base64
p=pathlib.Path(__file__).parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
(p/'GPU-manifest.json').write_text(json.dumps({'parent_library_SHA256':'26298a61aa072a32071b00906f11b61ede9878e2c9dd9761495c28e1695f7793','native_dependency_SHA256':'f35956f532833ce964bf766c8a61b44b56fe79042921d86710ccd075101705d2','fusion_library_SHA256':'5a831e6a758ba55bdb0987b1a37aaef90d2719b0dd54aca11650bab381a217ea','CPU_compile_source_manifest_SHA256':'18dd55a537333c4d8a869af5547db83626446f24368a83d3fed39d45a6fa4987'},indent=2))
names=['fused.cu','fusion.py','GPU_run.py','GPU_inside.sh','CPU_build.py','CPU_inside.sh','CPU_contract.py','GPU-manifest.json','controller.py']
m={'schema':1,'scope':'GLM component activation/quant fusion; not Model qualification','files_SHA256':{n:sha(p/n) for n in names},'variants':[{'id':i,'lanes':[16,32,64,128][i//4],'groups':[1,2,4,8][i%4],'parent':'910371' if i<8 else '910374','node':'a02r3n16' if i<8 else 'a02r4n07','local_rank':i%8} for i in range(16)],'parent_GPU_harness_SHA256':'6f059b806a3cc0765cc51e98c78c8459114d03d018461cfc714df07789f3f673','budget':{'per_rank_seconds':600,'batch_hard_minutes':30,'GPU_world':False}}
(p/'SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2))
files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names+['SOURCE_MANIFEST.json']}
remote="""import pathlib,base64,json,subprocess,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-operator-search-r341';assert not inc.exists();inc.mkdir()
files=FILES
for n,b in files.items(): (inc/n).write_bytes(base64.b64decode(b))
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_'))}
a=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:10:00','--job-name=GLM341-CPU','--output='+str(r/'logs/GLM341-CPU-%J.log'),'env','WAVE_INSIDE_R2323=GLM-operator-search-r341/CPU_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with (r/'logs/GLM341-CPU-client.log').open('x') as f:z=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'uploaded':list(files),'CPU_submit_PID':z.pid,'argv':a,'GPU_submitted':False}))
""".replace('FILES',repr(files))
(p/'remote_upload_CPU.py').write_text(remote)
print(json.dumps({'source_manifest_SHA256':sha(p/'SOURCE_MANIFEST.json'),'files':len(files)}))
