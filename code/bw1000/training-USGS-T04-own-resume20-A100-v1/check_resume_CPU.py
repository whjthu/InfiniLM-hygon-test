"""Only new own-lineage config boundaries; no oldRSS/native/state/GPU tests."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
from pathlib import PurePosixPath
import json
from pathlib import Path
from resume_prerequisites import config_contract
D=Path(__file__).resolve().parent


def run():
    c=json.loads((D/'CONFIG.template.json').read_bytes());p=json.loads((D/'PREREQUISITES.json').read_bytes());assert config_contract(c,p) is True
    # Exercise the actual published common planner without Torch/model imports.
    root=D.parents[2]
    rp=root/'code/bw1000/training-vlm-cuda-resume20-v1/deltas/tools/hygon_vlm_recipe.py'
    raw=rp.read_bytes();assert hashlib.sha256(raw).hexdigest()=='7f83842037ecfe217d92948cc89caa1ac36ca71e24c8c4f20645e3d5e027641a'
    module=importlib.util.module_from_spec(importlib.util.spec_from_file_location('own_resume_actual_common_vlm',rp));exec(compile(raw,str(rp),'exec'),module.__dict__)
    lp=root/'code/bw1000/training-USGS-T04-fresh-domain-A100-v1/overlay/tools/usgs_domain_launch.py';text=lp.read_text();tree=ast.parse(text)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['need','options','make_launch']]
    namespace=dict(vlm=module,PurePosixPath=PurePosixPath,KEY='native_usgs_domain_witness')
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual shared400 make_launch>','exec'),namespace)
    launch=namespace['make_launch'](c,(D/'T04-original-launch.json').read_bytes(),swift=PurePosixPath('/synthetic/bin/swift'),models=PurePosixPath('/synthetic/models'),plugins=PurePosixPath('/synthetic/plugins'),framework=PurePosixPath('/synthetic/framework'),master_port=29917)
    command=launch['command'];assert command[command.index('--resume_from_checkpoint')+1]==c['checkpoint']
    assert all(command[command.index(k)+1]=='false' for k in ['--resume_only_model','--ignore_data_skip','--load_args'])
    assert command[command.index('--dataset')+1]==c['training800']['path'] and command[command.index('--max_steps')+1]=='25'
    assert launch['environment_overrides']['NPROC_PER_NODE']=='8' and launch['common_entry']=='tools/hygon_training_recipe.py'
    rejected=[]
    def reject(name,v):
        try:config_contract(v,p)
        except ValueError:rejected.append(name);return
        raise AssertionError(name)
    for name,edit in [('fresh25_replay',lambda v:v.update(mode='fresh25')),('generic_arm',lambda v:v.update(arm='generic')),('original64_input',lambda v:v['training800'].update(SHA256='0'*64)),('missing_rank_ref',lambda v:v['continuous_reference'].pop('7')),('wrong_checkpoint_pin',lambda v:v['checkpoint_files']['optimizer.pt'].update(SHA256='0'*64)),('old_output_overlap',lambda v:v.update(train_output=v['checkpoint']))]:
        v=deepcopy(c);edit(v);reject(name,v)
    return dict(passed=True,normal=2,rejected=len(rejected),rejected_cases=rejected,Torch_imports=0,actualstate_reads=0,new_native_GPU_training=0,old_tests_rerun=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
