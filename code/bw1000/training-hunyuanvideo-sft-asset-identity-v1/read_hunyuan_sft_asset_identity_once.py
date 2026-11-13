"""Read the pinned SFT assets in place and report bounded target availability."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
REPO = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
A100 = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
proof = REPO / 'results/bw1000/2026-11-13/training-hunyuanvideo-three-a100-case-references.json'
config = json.loads(proof.read_bytes())['cases']['T09.sft']['launch']
source_bindings = dict(cache=config['config']['training_cache'],
                       cache_sha=config['config']['training_cache_sha'],
                       weights=config['weights'], source=config['source'])
remote = r'''
import datetime, hashlib, json, re, stat
from pathlib import Path

def status(path):
    try: st=path.stat()
    except FileNotFoundError:return None
    return dict(bytes=st.st_size,regular=stat.S_ISREG(st.st_mode),directory=stat.S_ISDIR(st.st_mode),
                device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)

def digest(path, limit):
    before=status(path)
    if before is None:return None
    if not before['regular'] or not 0<before['bytes']<=limit:
        return dict(stat=before,SHA256=None,reason='size or file type outside read scope')
    h=hashlib.sha256()
    with path.open('rb') as handle:
        left=before['bytes']
        while left:
            raw=handle.read(min(left,4*1024**2))
            if not raw:raise EOFError('asset truncated')
            left-=len(raw)
            h.update(raw)
        if handle.read(1):raise ValueError('asset grew during read')
    if status(path)!=before:raise ValueError('asset stat changed during read')
    return dict(stat=before,SHA256=h.hexdigest())

def sft_assets(weights, source, cache):
    transformer=weights/'transformer'/'480p_t2v'
    expected_sources=['hyvideo/commons/parallel_states.py','hyvideo/optim/muon.py',
        'hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py']
    files={n:digest(source/n,2*1024**2) for n in expected_sources}
    indices=['diffusion_pytorch_model.safetensors.index.json','model.safetensors.index.json']
    selected={}
    for n in indices:
        found=digest(transformer/n,4*1024**2)
        if found is None:continue
        raw=(transformer/n).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=found['SHA256']:raise ValueError('index changed')
        mapping=json.loads(raw)['weight_map']
        names=sorted(set(mapping.values()))
        if not names or len(names)>32 or any(not re.fullmatch(r'[A-Za-z0-9._-]+\.safetensors',n) for n in names):
            raise ValueError('unexpected shard paths')
        selected[n]=dict(**found,tensor_keys=len(mapping),
                         shards={n:status(transformer/n) for n in names})
    if len(selected)>1:raise ValueError('multiple transformer indices')
    singles={n:status(transformer/n) for n in ('diffusion_pytorch_model.safetensors','model.safetensors')} if not selected else {}
    return dict(weights=weights.as_posix(),source=source.as_posix(),cache=cache.as_posix(),
                weights_present=weights.is_dir(),source_present=source.is_dir(),
                cache_stat=status(cache),transformer_config=digest(transformer/'config.json',1024**2),
                transformer_indices=selected,single_file_candidates=singles,entry_source_files=files)

now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
if ROLE=='a100':
    c=Path(PINS['cache'])
    result=sft_assets(Path(PINS['weights']),Path(PINS['source']),c)
    cached=digest(c,1024**3)
    result['cache_identity']=cached
    result['cache_sha_matches']=bool(cached and cached['SHA256']==PINS['cache_sha'])
    print(json.dumps(dict(role=ROLE,observed_UTC=now(),sft=result,GPU_calls=0,weight_payload_bytes_read=0)))
else:
    root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
    home=root.parent
    results=root/'results'
    weights=[home/'shared/hunyuanvideo-1.5',home/'shared/HunyuanVideo-1.5',
             results/'training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5']
    sources=[root/'sources/hunyuanvideo-1.5',results/'training-hunyuanvideo-original-source-v1/source']
    caches=[results/'training-hunyuanvideo-original-inputs-v1/sample.pt']
    print(json.dumps(dict(role=ROLE,observed_UTC=now(),candidates={
        str(p):status(p) for p in weights+sources+caches},
        readable_transformer_configs={str(p):digest(p/'transformer/480p_t2v/config.json',1024**2)
            for p in weights if p.is_dir()},
        GPU_calls=0,weight_payload_bytes_read=0,cache_payload_bytes_read=0)))
'''
def observe(argv, role):
    program=f'ROLE={role!r}\nPINS={source_bindings!r}\n'+remote
    run=subprocess.run(argv,input=program.encode(),capture_output=True,timeout=65)
    receipt=dict(rc=run.returncode,stderr_SHA256=hashlib.sha256(run.stderr).hexdigest())
    if run.returncode==0:
        receipt['value']=json.loads(run.stdout)
    else:
        receipt['failure_types']=[name for name in ('Connection closed','Permission denied','ValueError',
             'FileNotFoundError','KeyError','Timeout') if name.encode() in run.stderr]
    return receipt

tree=ast.parse((BASE/'read_extra40_approval_original_handles_once.py').read_bytes())
index=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign)
           and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='ssh')
env={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:index+1],type_ignores=[]),'known-readonly-transport','exec'),env)
source_ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A100/'ssh_config'),
            'a100-26','timeout 50 python3 -B -u -']
checks={'A100':observe(source_ssh,'a100'),'Hygon':observe(env['ssh'],'hygon')}
checks['reference_SHA256']=hashlib.sha256(proof.read_bytes()).hexdigest()
with (BASE/'hunyuan-sft-original-and-target-asset-identity-v1.json').open('x',encoding='utf8') as f:
    json.dump(checks,f,indent=2)
origin=checks['A100'].get('value',{}).get('sft',{})
print(json.dumps(dict(source_rc=checks['A100']['rc'],target_rc=checks['Hygon']['rc'],
    source_failures=checks['A100'].get('failure_types'),target_failures=checks['Hygon'].get('failure_types'),
    cache_bytes=(origin.get('cache_stat') or {}).get('bytes'),cache_sha_matches=origin.get('cache_sha_matches'),
    source_files_found=sum(v is not None for v in origin.get('entry_source_files',{}).values()),
    transformer_indices=list(origin.get('transformer_indices',{})),
    target_paths_present=sum(v is not None for v in checks['Hygon'].get('value',{}).get('candidates',{}).values()))))
