import pathlib,json,os,subprocess,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
record=R/'results/GLM-context-count-only-r9392-launch.json';assert not record.exists()and not(R/'results/GLM-context-count-only-r9392.json').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))}
a=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','00:03:00','--job-name=GLM-context-count-only-r9392','--output='+str(R/'logs/GLM-context-count-only-r9392-%J.log'),'env','WAVE_INSIDE_R2323=GLM-context-count-only-r9392/CPU_inside.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'model_weights_loaded':False,'args':a}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/GLM-context-count-only-r9392-client.log').open('x')as f:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
