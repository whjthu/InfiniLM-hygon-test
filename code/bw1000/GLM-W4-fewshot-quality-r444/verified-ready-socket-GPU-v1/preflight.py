"""Final real source CLI/argv/256 frozen shard metadata; no serving/device."""
from pathlib import Path
import contextlib
import copy
import hashlib
import io
import json
import sys


def host_path(root,container_path):
    # Only this exact mount mapping; no general string/path substitution.
    return Path(root)/Path(container_path).relative_to('/refactor')


def check(deploy,provider,m,slot,gpu_command):
    from client_dispatch import quality_args
    deploy,provider=Path(deploy),Path(provider);root=deploy.parents[2]
    sys.path.insert(0,str(provider))
    from public_glm_native.cli import main,validate_deployment_paths
    from public_glm_native.topology_r404 import validated_nodes_r444
    path=deploy/('config-slot%d.json'%slot['slot']);config=json.loads(path.read_bytes())
    validate_deployment_paths(config);assert list(validated_nodes_r444(config['nodes']))==slot['nodes']
    from public_glm_native.ready_socket import addresses,PORT
    assert addresses(config)==['12.1.2.74','12.1.2.75'] and config['ready_port']==PORT==43844
    assert config['execute_authorized']and config['nodes']==slot['nodes']
    assert(config['Native_seconds'],config['service_seconds'])==(28800,28200)
    results=[]
    for stage in[0,1]:
        out=io.StringIO()
        with contextlib.redirect_stdout(out):main(['--config',str(path),'--config-sha',m['config_SHA256'],
            '--source-sha',m['provider_source_SHA256'],'--stage',str(stage),'--run',slot['run'],
            '--actual-step','CPU.source-only','--gpu-start-epoch','0'])
        q=json.loads(out.getvalue());assert q['source_and_schema_valid']and not q['GPU']and not q['Native_dlopen'];results.append(q)
    client_root=root/'incoming/GLM-W4-fewshot-quality-r444-client-v1/client-v1';sys.path.insert(0,str(client_root))
    import quality_client as client
    argv=quality_args(slot,m,str(slot['parent'])+'.0',1.)
    args=client.parser().parse_args(argv)
    assert args.shards==8 and args.shard_index==slot['slot'] and args.deadline_unix==28201.
    assert args.expected_step==str(slot['parent'])+'.0' and args.provider_source_sha==m['provider_source_SHA256']
    assert host_path(root,args.provider_root)==provider
    assert host_path(root,args.config)==path and host_path(root,args.service_report)==root/'results/GLM-W4-fewshot-quality-r444'/slot['run']/'owner-1-0.json'
    mapped=copy.copy(args);mapped.fixtures=str(host_path(root,args.fixtures));mapped.encoding_receipt=str(host_path(root,args.encoding_receipt))
    chosen,plan,enc,schedule=client.fixtures(mapped)
    assert len(chosen)==32 and len(schedule)==64 and len({c['id']for c in chosen})==32
    assert sum(schedule[i]['window']==2 for i in range(0,64,2))==16
    assert sum(schedule[i]['window']==4 for i in range(0,64,2))==16
    command=list(gpu_command)
    assert '--jobid='+str(slot['parent'])in command and command[command.index('-w')+1]==','.join(slot['nodes'])
    assert command[command.index('-N2')+1]=='-n2'and '--gres=dcu:8'in command and command[command.index('-c64')+1]=='--gres=dcu:8'
    inside=(deploy/'inside.sh').read_bytes();assert b'\r'not in inside and b'\x00'not in inside
    assert b'GLM_R444_ENV_ONLY'in inside and b'+28800-time.time()'in inside
    assert not any(n in sys.modules for n in ['torch','_infinilm','transformers'])
    return dict(slot=slot['slot'],actual_public_CLI_two_roles=results,HOST_client_parser_and_strict_container_paths=True,
        planned_case_ids=[c['id']for c in chosen],questions=32,HTTP=64,balanced_pair_order=True,
        control_exposed=sum(enc[c['id']]['planned_pair_control_exposed']for c in chosen),
        Native_imported=False,GPU=0,Slurm=0,Model_READY=False)
