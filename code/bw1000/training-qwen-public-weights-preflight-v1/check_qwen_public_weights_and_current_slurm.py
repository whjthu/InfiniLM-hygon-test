"""Check exact public Qwen weight availability from the target and both Slurm parents."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

base = Path(__file__).resolve().parent
repo = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
identity = repo / 'coordination/training/qwen-official-public-original-model-identity-v1.json'
pins = json.loads(identity.read_bytes())
revision = pins['revision']
assert revision == '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
shards = pins['official_weight_pointers']['files']
assert len(shards) == 4 and all(p['matches_original_complete_SHA'] for p in shards.values())
ledger_file = repo / 'coordination/training/training80-active-node-task-map-v1.json'
ledger = json.loads(ledger_file.read_bytes())
allocations = {job: sorted({node for group in ledger['groups'].values()
                            for node in group['allocations'][job]})
               for job in ('1087656','1103606')}
transport = ast.parse((base / 'read_extra40_approval_original_handles_once.py').read_bytes())
index = next(i for i, n in enumerate(transport.body) if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'ssh')
scope = {'__file__': __file__}
exec(compile(ast.Module(body=transport.body[:index+1],type_ignores=[]),'known-readonly-transport','exec'), scope)
remote = r'''
import concurrent.futures,datetime,json,shutil,subprocess,urllib.request
from pathlib import Path

def query(*args):
    return subprocess.check_output(args, text=True, timeout=10).splitlines()

parents={}
for job,expected in ALLOCATIONS.items():
    raw=query('squeue','-h','-j',job,'-o','%i|%T|%D|%b|%l|%e|%N')
    assert len(raw)==1
    job_id,state,count,gres,limit,end,nodelist=raw[0].split('|')
    nodes=query('scontrol','show','hostnames',nodelist)
    live_steps=query('squeue','--steps','-h','-j',job,'-o','%i|%j|%N')
    parents[job]=dict(state=state,nodes=len(nodes),unique_nodes=len(set(nodes)),time_limit=limit,
                      end_cluster_time=end,gres=gres,ledger_nodes_match=sorted(nodes)==expected,
                      visible_step_ids=[row.split('|')[0] for row in live_steps])
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
target=root/'training-qwen-original-public-model-v1'
def head(entry):
    name,pin=entry
    url='https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/'+REVISION+'/'+name
    try:
        request=urllib.request.Request(url,method='HEAD',headers={'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=12) as response:
            size=response.headers.get('Content-Length')
            return name,dict(status=response.status,content_length=size,
                             length_matches=size==str(pin['weight_bytes']))
    except (OSError, ValueError) as exc:
        return name,dict(status=getattr(exc,'code',None),error_type=type(exc).__name__,
                         reason_type=type(getattr(exc,'reason',None)).__name__)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    result=dict(pool.map(head,SHARDS.items()))
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    parents=parents,target_namespace_absent=not target.exists(),free_bytes=shutil.disk_usage(root).free,
    public_HEAD=result,public_payload_bytes=0,GPU_cards_used=0,device_busy_sampled=False)))
'''
prefix = f'ALLOCATIONS={allocations!r}\nREVISION={revision!r}\nSHARDS={shards!r}\n'
run = subprocess.run(scope['ssh'],input=(prefix+remote).encode(),capture_output=True,timeout=60)
receipt = dict(transport_rc=run.returncode, stderr_SHA256=hashlib.sha256(run.stderr).hexdigest(),
               identity_SHA256=hashlib.sha256(identity.read_bytes()).hexdigest(),
               ledger_SHA256=hashlib.sha256(ledger_file.read_bytes()).hexdigest())
if run.returncode == 0:
    receipt['value'] = json.loads(run.stdout)
else:
    receipt['failure_types'] = [x for x in ('Connection closed','Permission denied','AssertionError','TimeoutExpired') if x.encode() in run.stderr]
with (base/'qwen-public-weights-and-current-slurm-readonly-v1.json').open('x',encoding='utf8') as f:
    json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
