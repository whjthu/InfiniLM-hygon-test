"""Final actual Bash env -> unchanged typed public CLI, no device or Slurm."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from unittest.mock import patch


def check(deploy,provider,m):
 deploy,provider=Path(deploy),Path(provider)
 sys.path.insert(0,str(provider))
 from public_kimi_native.cli import main
 from controller import gpu_environment,scope_contract
 from workload import body_and_gold,host
 config=json.loads((deploy/'config.json').read_bytes())
 assert config['execute_authorized']and config['nodes']==['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
 assert (config['Native_seconds'],config['service_seconds'])==(5100,4500)
 model=json.loads(host(config['model_manifest']).read_bytes())
 # Run every actual export/unset/if from final shell. Only container setup and
 # metadata existence tests are recorded on the login HOST; no SDK/Torch load.
 inside=(deploy/'inside.sh').read_text();prefix=inside.split('# Actual whole typed CLI')[0]
 prefix=prefix.replace('source /opt/dtk/env.sh', ': # recorded container SDK env source')
 prefix=prefix.replace('test -f "$ROCBLAS_TENSILE_LIBPATH/TensileLibrary_lazy_gfx936.dat"', ': # container-only metadata existence')
 prefix=prefix.replace('test -d "$TORCH_LIB_DIR"', ': # container-only Torch library directory existence')
 assert '$(python'not in prefix and 'python -'not in prefix
 selected=set(model['required_environment'])|{'SLURM_PROCID','SLURM_JOB_ID','SLURM_STEP_ID','GLOBAL_OFFSET_R2354',
  'KIMI_ENTRY_R9143','KIMI_CAPACITY_ONLY_R9145','KIMI_DRAFT_CARD_COUNT_R9142','INFINILM_DENSE_PREFILL_F32_R2475',
  'INFINILM_MXFP4_TILE_TP_R2559','INFINILM_KIMI_W2_PAIR_R9280','INFINILM_KIMI_I192_GROUPED_R8669Y',
  'INFINILM_KIMI_MOE_EXECUTION_R4665','INFINILM_KIMI_BATCH_I192_R9125'}
 append='\n'+sys.executable+' -B -c '+__import__('shlex').quote('import os,json;print(json.dumps({k:os.environ.get(k)for k in '+repr(sorted(selected))+'}))')+'\n'
 original_read=Path.read_bytes
 def mapped_read(path):
  if str(path)==config['model_manifest']:return original_read(host(config['model_manifest']))
  return original_read(path)
 proof=[]
 for node in range(4):
  env=gpu_environment(os.environ,0,m,'source-only-recording-nonce-not-auth')
  env.update(SLURM_PROCID=str(node),SLURM_JOB_ID='CPU',SLURM_STEP_ID='source-only')
  z=subprocess.run(['bash','-c',prefix+append],env=env,capture_output=True,text=True,timeout=25)
  assert z.returncode==0,(z.stdout,z.stderr)
  values=json.loads(z.stdout.strip().splitlines()[-1])
  for k,v in model['required_environment'].items():assert values[k]==v,(node,k,values[k],v)
  assert values['KIMI_ENTRY_R9143']=='1'and values['KIMI_CAPACITY_ONLY_R9145']=='0'
  assert values['INFINILM_DENSE_PREFILL_F32_R2475']=='1'and values['KIMI_DRAFT_CARD_COUNT_R9142']=='0'
  out=io.StringIO()
  with patch.dict(os.environ,values,clear=True),patch.object(Path,'read_bytes',mapped_read),patch('socket.gethostname',return_value=config['nodes'][node]),contextlib.redirect_stdout(out):
   main(['--config',str(deploy/'config.json'),'--config-sha',m['config_SHA256'],'--source-sha',m['provider_source_SHA256'],
    '--node',str(node),'--run','source-only-validation','--nonce','source-only-recording-nonce-not-auth',
    '--actual-step','CPU.source-only','--gpu-start-epoch','0'])
  q=json.loads(out.getvalue());assert q['source_and_config_valid']and not q['GPU']and not q['Native_dlopen']
  proof.append(dict(node=node,actual_Bash_exports=values,actual_typed_CLI=q))
 fixtures=[]
 for fixture in json.loads((deploy/'FIXTURE_CARD.json').read_bytes())['plan']:
  raw,body,gold=body_and_gold(fixture)
  fixtures.append(dict(index=fixture['index'],actual_request_SHA256=hashlib.sha256(raw).hexdigest(),
   full_history_messages=len(body['messages']),prompt_tokens=fixture['current_prompt_tokens'],
   budget=fixture['output_budget'],stream=fixture['stream'],independent_gold=gold))
 assert not any(n in sys.modules for n in ['torch','transformers','_infinilm','infinilm'])
 return dict(passed=True,whole_controller_scope=scope_contract(),actual_final_Bash_to_typed_CLI_four_roles=proof,fixtures=fixtures,
  recording_leaves=['container SDK environment source/file existence','HOST-only POSIX manifest path map','HOST-only hostname and Slurm identity'],
  actual_node_hardware=False,Native_imported=False,GPU=0,Slurm=0,
  qualification='New config/time/source handoff only; actual node inventory/final CLI/device remains in once claim, no Model qualification.')
