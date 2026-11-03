"""Actual Bash exports into the same pure GPUentry policy, without device imports."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys
def check(d,m):
 from controller import gpu_command,NAME,LABEL
 from GPU_run import validate_policy
 config=dict(SLURM_JOB_ID='910371',SLURM_STEP_ID='recording-only')
 prefix=(Path(d)/'GPU_inside.sh').read_text().split('LEFT=')[0]
 prefix=prefix.replace('$(hostname -s)','${KIMI_R437_PREFLIGHT_NODE:?}')
 prefix=prefix.replace('source /opt/dtk/env.sh',': # container SDK setup recorded on HOST')
 prefix=prefix.replace('test -d "$TORCH_LIB_DIR"',': # container Torch directory existence recorded on HOST')
 prefix=prefix.replace('test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"',': # container rocBLAS metadata existence recorded on HOST')
 assert '$(python'not in prefix and 'python -'not in prefix and 'python3 -'not in prefix and '$('not in prefix
 keys=['SLURM_JOB_ID','SLURM_PROCID','INFINILM_OWNER_CPU_WIDTH_R2392','INFINILM_KIMI_SPEC_MATH_R3676','INFINILM_KIMI_MOE_WORKSPACE_R4665','INFINILM_KIMI_MOE_EXECUTION_R4665','INFINILM_KIMI_I192_GROUPED_R8669Y','INFINILM_KIMI_I192_COMPACT_R8669ZC','INFINILM_KIMI_BATCH_I192_R9125','INFINILM_KIMI_PACKED_W2_R6122','INFINILM_KIMI_PACKED_AUTO_R6136','INFINILM_KIMI_PROFILE_R9125','INFINILM_KIMI_MOE_AUX_M1_R4430','INFINILM_KIMI_ROUTER_M1_R4635']
 append='\n'+shlex.quote(sys.executable)+' -B -c '+shlex.quote('import os,json;print(json.dumps({k:os.environ.get(k)for k in '+repr(keys)+'}))')+'\n'
 rows=[]
 for node,hostname in enumerate(['a02r3n03','a02r3n07']):
  env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','INFINILM_','KIMI_','CUDA_','HIP_','ROCR_'))and k not in ['LD_PRELOAD','BASH_ENV','ENV']};env.update(config,SLURM_PROCID=str(node),KIMI_R437_PREFLIGHT_NODE=hostname)
  z=subprocess.run(['bash','-c',prefix+append],env=env,capture_output=True,text=True,timeout=25);assert z.returncode==0,(z.stdout,z.stderr)
  values=json.loads(z.stdout.strip().splitlines()[-1]);rows.append(dict(actual_Bash_exports=values,actual_policy=validate_policy(values,hostname)))
 args=gpu_command(Path(d).parents[2]);assert '-N2'in args and '-n2'in args and '-c64'in args and '--gres=dcu:8'in args
 assert args[args.index('-w')+1]==','.join(m['resource_claim']['nodes'])and m['resource_claim']['cards']==16
 assert str(Path(d)/'resource_wave_host_alias_outer.sh')in args
 assert LABEL=='Kimi-r437-full-MoE-lowrank-v3'and NAME=='Kimi-full-MoE-lowrank-M1-r437/gpu-v3'
 assert not any(n in sys.modules for n in ['torch','_infinilm','infinilm','transformers'])
 return dict(passed=True,actual_Bash_and_same_entry_policy=rows,requested_topology=dict(TP=16,PP=1,nodes=m['resource_claim']['nodes'],cards=16),GPU=0,Slurm=0,Native_dlopen=False,recording_leaves=['model-node hostname in original Bash identity test','container SDK env setup and container file-existence checks only'],actual_owner_or_device_qualification=False)
