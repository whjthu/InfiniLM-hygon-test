import json,pathlib,subprocess,time
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
args=['srun','--jobid=910406','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a02r3n19','--gres=none','--mem=4096M','-t','00:05:00','--job-name=Deep-B64-M64-CPU-r8644','env','WAVE_INSIDE_R2323=Deep_B64_M64_CPU_r8644_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
record=r/'results/Deep-B64-M64-CPU-r8644-launch.json'
assert not record.exists()
log=r/'logs/Deep-B64-M64-CPU-r8644-controller.log'
with log.open('x')as f:
 p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q={'CPU_only':True,'GPU_initialized':False,'dlopen':False,'native_compile':False,'pid':p.pid,'args':args,'samehandle_only':True}
with record.open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
