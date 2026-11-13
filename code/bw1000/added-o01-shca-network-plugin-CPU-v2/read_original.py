from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
directory = root / 'added-o01-shca-network-plugin-CPU-v2'
intent = json.loads((directory / 'launch-intent.json').read_bytes())
original = json.loads((directory / 'transport-outcome.json').read_bytes())
publication = json.loads((directory / 'source-publication-receipt.json').read_bytes())
task = json.loads((directory / 'publication/coordination/tasks/added-o01-shca-network-plugin-CPU-v2.json').read_bytes())
assert intent['claim'] == original['claim'] == publication['commit'] == publication['remote_head']
assert intent['request_id'] == original['request_id'] and original['remote_outcome'] == 'ORIGINAL_RETURNED'

tree = ast.parse((root / 'read_extra40_approval_original_handles_once.py').read_bytes())
ssh_start = [node for node in tree.body if isinstance(node, ast.Assign) and
             isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'ssh'][0]
prelude = ast.Module(body=tree.body[:tree.body.index(ssh_start) + 1], type_ignores=[])
transport = {'__file__': __file__}
exec(compile(prelude, 'existing-readonly-transport', 'exec'), transport)
remote = r'''
from pathlib import Path
import hashlib,json,subprocess
w=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/added-o01-shca-network-plugin-CPU-v2')
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
raw=(w/'outcome.json').read_bytes()
out=json.loads(raw)
assert out==EXPECTED and out['claim']==CLAIM and out['request_id']==REQUEST
assert out['GPU_cards']==0 and out['owned_srun_reaped'] and not out['forced_cleanup']
assert out['returncode']==0 and out['terminal']['terminal_visible'] and out['deadline']-out['epoch']==180
row,=out['accounting']
assert row==['1103606.22',w.name,'COMPLETED','0:0','1','f12r2n[10-11]']
accounting=subprocess.check_output(['sacct','-n','-P','-j',row[0],'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12)
assert row in [line.split('|') for line in accounting.splitlines()]
live=subprocess.check_output(['squeue','--steps','-h','-j','1103606','-o','%i'],text=True,timeout=12)
assert row[0] not in live.split()
assert digest((w/'CPU.log').read_bytes())==out['log_SHA256']
for target, name in REMOTE_FILES.items():
    assert digest((w/'src'/target).read_bytes())==SOURCES[name]
records=out['records']
assert set(records)=={'f12r2n10','f12r2n11'} and {r['value']['rank'] for r in records.values()}=={0,1}
nodes={}
for node, record in records.items():
    data=(w/'nodes'/(node+'.json')).read_bytes()
    assert record['pin']==dict(bytes=len(data),SHA256=digest(data))
    assert json.loads(data)==record['value']
    value=record['value']
    assert value['node']==node and value['step']==row[0]
    plugin=value['plugin']
    assert plugin['plugin_v8_loaded'] is True and plugin['ldd_rc']==0 and plugin['loader_errors']==[]
    assert plugin['exports']=={f'ncclNetPlugin_v{n}':f'NCCL RDMA Plugin v{n}' for n in range(5,9)}
    mapped=plugin['mapped_dependencies']
    assert len(mapped)==8 and mapped['/rdma-provider-stack/libibverbs.so.1']['bytes']>0
    assert plugin['plugin_pin']==mapped['/candidate-opt/rccl-shca-net/lib/librccl-net-shca.so.0.0.0']
    assert plugin['plugin_pin']['SHA256']=='20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0'
    nodes[node]={key:plugin[key] for key in ('plugin_pin','rccl_pin','exports','mapped_dependencies')}
    nodes[node]['record_pin']=record['pin']
pins=lambda entry:{k:v for k,v in entry.items() if k!='record_pin'}
assert pins(nodes['f12r2n10'])==pins(nodes['f12r2n11'])
print(json.dumps(dict(verified=True,claim=CLAIM,request_id=REQUEST,accounting=row,
    outcome_SHA256=digest(raw),nodes=nodes,GPU_cards=0,network_initialized=False)))
'''
bindings = dict(CLAIM=intent['claim'], REQUEST=intent['request_id'], EXPECTED=original['value'],
                REMOTE_FILES=task['remote_files'], SOURCES=task['source_SHA256'])
payload = (''.join(f'{key}={value!r}\n' for key, value in bindings.items()) + remote).encode()
process = subprocess.run(transport['ssh'], input=payload, capture_output=True, timeout=55)
result = dict(transport_rc=process.returncode, stdout_SHA256=hashlib.sha256(process.stdout).hexdigest(),
              stderr_SHA256=hashlib.sha256(process.stderr).hexdigest())
if process.returncode == 0:
    result['value'] = json.loads(process.stdout)
with (directory / (sys.argv[1] + '.json')).open('x', encoding='utf-8') as out:
    json.dump(result, out, indent=2)
if process.returncode != 0:
    print(json.dumps({key: key in process.stderr.decode(errors='replace') for key in ('Connection closed', 'Connection reset', 'Permission denied', 'Host key verification failed', 'timed out', 'AssertionError', 'KeyError', 'SyntaxError')}))
value = result.get('value', {})
print(json.dumps(dict(transport_rc=process.returncode, verified=value.get('verified'),
                      accounting=value.get('accounting'), nodes=sorted(value.get('nodes', {})))))
