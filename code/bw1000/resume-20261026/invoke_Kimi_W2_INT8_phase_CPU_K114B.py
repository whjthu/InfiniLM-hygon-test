import pathlib,json,subprocess,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=r/'incoming/Kimi-W2-INT8-phase-K114B'
prior=subprocess.check_output(['squeue','--steps','-j','910375','-h','-o','%i|%j|%N'],text=True,timeout=20).splitlines();numeric=[x for x in prior if x.split('|')[0].split('.')[-1].isdigit()];assert not any('a02r4n08'in x for x in numeric),numeric
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_VISIBLE','HIP_VISIBLE','ROCR_VISIBLE'))};env['WAVE_INSIDE_R2323']='Kimi-W2-INT8-phase-K114B/CPU_inside.sh'
a=['srun','--jobid=910375','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a02r4n08','-t','00:15:00','--job-name=Kimi-INT8-phase-CPU-K114B','--output='+str(r/'logs/Kimi-INT8-phase-CPU-K114B-%J.log'),'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
f=(r/'logs/Kimi-INT8-phase-CPU-K114B-client.log').open('x');p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'PID':p.pid,'args':a,'GPU':False,'prior_numeric':numeric}))
