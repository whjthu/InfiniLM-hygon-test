"""Actual Bash policy exports -> same typed purelayout as production preengine.

DTK/Torch library discovery and device/CABI work remain outside this CPU. The
unchanged LD line is evaluated with an empty Torch-library placeholder solely
to validate its SDK component; no child is spawned after exporting LD_PRELOAD.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(root,config_path,controller_name='controller/controller.py'):
 root=Path(root);owner=root/'owner';raw=(owner/'GPU_inside_r290.sh').read_text();tree=ast.parse((root/controller_name).read_text())
 updates=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='update'and any(k.arg=='WAVE_INSIDE_R2323'for k in n.keywords)];assert len(updates)==1
 initial={k.arg:str(ast.literal_eval(k.value))for k in updates[0].keywords if k.arg and isinstance(k.value,ast.Constant)}
 prior=root.parent/'GLM-V-on-demand-Model-r404/cpu-layout-v4/check_layout_CPU.py';assert sha(prior)=='1fba84eb877ec2be2bda68a96a3a96321f92831a51c45b6d69ba40dd4517efd2'
 base_node=next(n for n in ast.walk(ast.parse(prior.read_text()))if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='base'for t in n.targets))
 expected={k.arg:ast.literal_eval(k.value)for k in base_node.value.keywords};assert len(expected)==11
 source_lines=raw.splitlines();prefix=[]
 for lead in ['export GLM_SHARED_PREFIX_VALIDATE_R5814=','export GLM_V_ON_DEMAND_R403=','export GLM_CROSSREQUEST_PREFILL_R7328=','export GLM_PP_SIZE_R3603=','export GLM_STAGE_R5040=','LM=','SDK=']:
  lines=[x for x in source_lines if x.startswith(lead)];assert len(lines)==1,lead;prefix+=lines
 ld=next(x for x in source_lines if x.startswith('export LD_LIBRARY_PATH='));preload=next(x for x in source_lines if x.startswith('export LD_PRELOAD='))
 fragment=raw[raw.index('test "${GLM_EXACT_BANKS_R4188:?}"'):raw.index('python3 /refactor/incoming/GLM_transient_preflight_r9059.py')]
 keys=sorted(set(expected)|{'SLURM_PROCID','GLM_STAGE_R5040'});suffix="printf 'BEGIN_LAYOUT_ENV\\0'\n"+''.join("printf '%s\\0%s\\0' "+k+' "${'+k+'}"\n'for k in keys)
 shell='set -euo pipefail\n'+'\n'.join(prefix)+'\nTORCH_LIB_DIR=\n'+ld+'\n'+preload+'\n'+fragment+'\n'+suffix
 spec=importlib.util.spec_from_file_location('actual_private_topology',owner/'topology_r404.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T);rows=[]
 for stage,node in enumerate(['a02r4n12','a02r4n13']):
  env=dict(PATH=os.environ.get('PATH','/usr/bin:/bin'),**initial,SLURM_PROCID=str(stage));p=subprocess.run(['bash','--noprofile','--norc','-c',shell],env=env,capture_output=True,timeout=15);assert p.returncode==0,p.stderr.decode(errors='replace')
  fields=p.stdout.split(b'BEGIN_LAYOUT_ENV\0',1)[1].split(b'\0');assert fields[-1]==b'';actual={fields[i].decode():fields[i+1].decode()for i in range(0,len(fields)-1,2)}
  for k,v in expected.items():
   if k=='LD_LIBRARY_PATH':assert '/install/hygon-glm-5f91ad7-r3672/lib:'in actual[k]and '/install/hygon-glm-5f91ad7-r3672/lib:'in v
   else:assert actual[k]==v,(k,actual[k],v)
  layout=T.read_layout(actual,config_path,node);rows.append(dict(stage=stage,actual_source_exports=actual,actual_same_typed_layout=layout,all_CPU820_fixture_policy_fields_match=True))
 return dict(completed=True,passed=True,actual_Bash=True,actual_production_typed_purelayout=True,controller_path=controller_name,controller_SHA256=sha(root/controller_name),GPU_inside_SHA256=sha(owner/'GPU_inside_r290.sh'),topology_SHA256=sha(owner/'topology_r404.py'),CPU820_fixture_source_SHA256=sha(root.parent/'GLM-V-on-demand-Model-r404/cpu-layout-v4/check_layout_CPU.py'),shell_fragment_SHA256=hashlib.sha256(shell.encode()).hexdigest(),roles=rows,GPU=0,Native_dlopen=False,Model=False,Slurm=0,not_executed='DTK source, actual Torch import/libdirectory resolution, provider/dlopen/create/device/weights. These unchanged leaves remain actual GPU qualification; empty Torch path only evaluates SDK substring and does not qualify full LD path.')
