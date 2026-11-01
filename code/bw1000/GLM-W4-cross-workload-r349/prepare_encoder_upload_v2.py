import pathlib,json,base64
p=pathlib.Path(__file__).parent;files={n:base64.b64encode((p/n).read_bytes()).decode() for n in ['CPU_encode_fixtures_v2.py','CPU_encode_inside_v2.sh']};files['FIXTURE_CARD.json']=base64.b64encode((p/'fixtures/FIXTURE_CARD.json').read_bytes()).decode()
s="""import pathlib,json,base64,subprocess,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'incoming/GLM-W4-cross-workload-r349/encoder_cpu_v2';assert not d.exists();d.mkdir(parents=True)
files=FILES
for n,b in files.items():(d/n).write_bytes(base64.b64decode(b))
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_'))}
a=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:06:00','--job-name=GLM349-official-encoder-v2','--output='+str(r/'logs/GLM349-encoder-v2-%J.log'),'env','WAVE_INSIDE_R2323=GLM-W4-cross-workload-r349/encoder_cpu_v2/CPU_encode_inside_v2.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with (r/'logs/GLM349-encoder-v2-client.log').open('x') as f:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_submit_PID':p.pid,'argv':a,'GPU_submitted':False}))
""".replace('FILES',repr(files));(p/'remote_upload_encoder_v2.py').write_text(s)
