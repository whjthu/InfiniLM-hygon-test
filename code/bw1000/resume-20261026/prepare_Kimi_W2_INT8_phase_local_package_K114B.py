"""LOCAL ONLY K114B source packaging. No network, Slurm, compiler, GPU or subprocess."""
import pathlib,json,hashlib,ast,base64
r=pathlib.Path(__file__).resolve().parent;old=r/'Kimi-W2-INT8-register-v2-private-r113c';d=r/'Kimi-W2-INT8-phase-private-K114B';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();server='Kimi-W2-INT8-phase-K114B'
def put(n,s):
 b=s if isinstance(s,bytes)else s.encode();b=b.replace(b'\r\n',b'\n');assert b'\r'not in b
 if n.endswith('.py'):ast.parse(b)
 p=d/n;assert not p.exists();p.write_bytes(b)
s=(old/'CPU_build_r113v2.py').read_text();s=s.replace("b=r/'candidates/Kimi-W2-INT8-register-v2-r113c'","b=r/'candidates/Kimi-W2-INT8-phase-K114B'")
s=s.replace("'HOST_wrapper_source_changed':False","'HOST_wrapper_source_changed':True").replace("'root_claim_publication'","'historical_v2_claim_publication'")
line=next(x for x in s.splitlines()if"proof=json.loads"in x)
s=s.replace(line," proof=json.loads((D/'sourceproof-K114B.json').read_bytes());source=D/'kimi_W2_INT8_provider_K114B.cu';runner=D/'W2_INT8_runner_K114B.cpp';assert sha(source)==proof['source_SHA256'][source.name];assert sha(runner)==proof['source_SHA256'][runner.name]")
s=s.replace('libKimi_W2_INT8_register_v2_r113c.so','libKimi_W2_INT8_phase_K114B.so').replace('libKimi_W2_INT8_register_v2_runner_r113c.so','libKimi_W2_INT8_phase_runner_K114B.so')
s=s.replace("(b/'matrix-ISA.s').write_text(ms);", "(b/'matrix-ISA.s').write_text(ms);assert ms==(r/'candidates/Kimi-W2-INT8-register-v2-r113c/matrix-ISA.s').read_text(),'matrix ISA changed';")
s=s.replace("matrix_controlflow_review='PENDING_CPU_SOURCE_REVIEW_NO_GPU_GO'", "matrix_controlflow_review='BYTE_IDENTICAL_TO_QUALIFIED755_NO_GPU_GO'")
s=s.replace("['modern_kimi_i8_bind_r113'", "['modern_kimi_k114b_init','modern_kimi_k114b_enable','modern_kimi_k114b_read','modern_kimi_k114b_close','modern_kimi_i8_bind_r113'")
# All listed qualified SDK headers / linked objects / wrapper ABI kept by inherited750/751 commands.
put('CPU_build_K114B.py',s)
sh='#!/bin/bash\nset -euo pipefail\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\nexport PYTHONPYCACHEPREFIX=/local/K114B-CPU\ntimeout --kill-after=5s 840 python -B -u /refactor/incoming/Kimi-W2-INT8-phase-K114B/CPU_build_K114B.py\n';put('CPU_inside.sh',sh)
# This launcher source is delivered for root review only, NOT executed by this producer.
launch='''import pathlib,json,subprocess,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=r/'incoming/Kimi-W2-INT8-phase-K114B'
prior=subprocess.check_output(['squeue','--steps','-j','910375','-h','-o','%i|%j|%N'],text=True,timeout=20).splitlines();numeric=[x for x in prior if x.split('|')[0].split('.')[-1].isdigit()];assert not any('a02r4n08'in x for x in numeric),numeric
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_VISIBLE','HIP_VISIBLE','ROCR_VISIBLE'))};env['WAVE_INSIDE_R2323']='Kimi-W2-INT8-phase-K114B/CPU_inside.sh'
a=['srun','--jobid=910375','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a02r4n08','-t','00:15:00','--job-name=Kimi-INT8-phase-CPU-K114B','--output='+str(r/'logs/Kimi-INT8-phase-CPU-K114B-%J.log'),'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
f=(r/'logs/Kimi-INT8-phase-CPU-K114B-client.log').open('x');p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'PID':p.pid,'args':a,'GPU':False,'prior_numeric':numeric}))
''';p=r/'invoke_Kimi_W2_INT8_phase_CPU_K114B.py';assert not p.exists();p.write_bytes(launch.encode());ast.parse(launch)
# Human readable exact narrowstage list; upload script is NOT executed here.
files=[d/n for n in ['kimi_W2_INT8_provider_K114B.cu','W2_INT8_runner_K114B.cpp','run_GPU_K114B.py','expected-headers-r9125.json','sourceproof-K114B.json','CPU_build_K114B.py','CPU_inside.sh']]
plan={'schema':1,'status':'LOCAL_SOURCE_READY_CPU_NOT_RUN_AUTO_REVIEW_REJECTED_COMBINED_STAGE_COMPILE','remote_incoming_directory':'/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/'+server,'CPU_resource':{'parent':'910375','node':'a02r4n08','cpus':2,'memMiB':8192,'gres':'none','hard_s':900,'inner_s':840},'GPU_GO':False,'source_and_CPU_launch_not_executed':True,'exact7_stage_files_SHA256':{p.relative_to(r).as_posix():sha(p)for p in files},'files_SHA256':{p.relative_to(r).as_posix():sha(p)for p in files+[r/'prepare_Kimi_W2_INT8_phase_K114B.py',pathlib.Path(__file__),p]},'event_API':'init(ownerdevice,originalstream),enable(threadlocal),read(float[5])afterr.syncStream,closeafterdrainbeforeRuntime; original r113 CAPIunchanged','uncertainty':'CPU compilation/ABI/actual ISA resources not yet run; CUDAevent runtime validity and perturbation UNKNOWN until separateGPU GO; matrix launch640x64 unchanged; static8averagewaves/CU is assumption not occupancy','next':'root review narrow7files stage and invokeCPU once; agent will follow sameactualhandle only'}
put('source-producer-index-K114B.json',json.dumps(plan,indent=2)+'\n');print(json.dumps({'private':str(d),'index_SHA256':sha(d/'source-producer-index-K114B.json'),'builder_SHA256':sha(d/'CPU_build_K114B.py'),'launcher_SHA256':sha(p),'executed':False}))
