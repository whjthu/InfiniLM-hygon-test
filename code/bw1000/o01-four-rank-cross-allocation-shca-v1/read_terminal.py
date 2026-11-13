from pathlib import Path
import ast,hashlib,json,subprocess,sys

D=Path(__file__).resolve().parent
B=D.parent
intent=json.loads((D/'launch-intent.json').read_bytes())
manifest=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
expected={n:p['SHA256'] for n,p in manifest.items()}
expected['SOURCE_FILES.json']=hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest()
tree=ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
idx=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and
    isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
config={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:idx+1],type_ignores=[]),'known-readonly-transport','exec'),config)
remote=r'''
import hashlib,json,re,statistics,subprocess
from pathlib import Path
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/o01-four-rank-cross-allocation-shca-v1')
PLACEMENTS=[('1087656','f12r3n17'),('1087656','f12r3n18'),('1103606','f12r2n10'),('1103606','f12r2n11')]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads((W/p).read_bytes())
def query(argv):
    return subprocess.check_output(argv,text=True,timeout=10)
actual={}
active=[]
for parent in dict.fromkeys(parent for parent,_ in PLACEMENTS):
    raw=query(['sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'])
    for line in raw.splitlines():
        row=line.split('|')
        if row[1].startswith(W.name+'-'):
            rank=int(row[1].rsplit("-r",1)[1])
            assert PLACEMENTS[rank]==(parent,row[5])
            actual[row[1]]=row
    active.extend(line.split('|')[0] for line in query(['squeue','--steps','-h','-j',parent,'-o','%i|%j']).splitlines()
                  if W.name in line)
if not (W/'outcome.json').exists():
    print(json.dumps(dict(state='ORIGINAL_OUTCOME_UNAVAILABLE',accounting=list(actual.values()),active=active)))
    raise SystemExit(0)
i=read('intent.json')
out=read('outcome.json')
assert i['claim']==out['claim']==CLAIM and i['request_id']==out['request_id']==REQUEST
assert i['source_files']==EXPECTED
for n,h in EXPECTED.items():
    assert sha(W/'src'/n)==h
assert sorted(out['accounting'])==sorted(actual.values()) and not active
for phase in ('pre','gpu','post'):
    p=out.get(phase)
    if p is None:
        continue
    assert read(phase+'-receipt.json')==p
    for step in p['steps']:
        assert sha(W/(step['name']+'.log'))==step['log_SHA256']
        if step['row'] is not None:
            assert step['row']==actual[step['name']] and step['terminal_visible']
numeric=[]
observed_logs=[]
trace_pins={}
for step in (out.get('gpu') or {}).get('steps',[]):
    raw=(W/(step['name']+'.log')).read_bytes()
    observed_logs.append(dict(node=step['node'],network=re.findall(r'Using network\s+([\w.-]+)',raw.decode(errors='replace')),
        plugin_channels=len(re.findall(rb'via NET/IBext(?:_v8)?(?:/|\b)',raw)),
        socket_channels=len(re.findall(rb'via NET/Socket(?:/|\b)',raw))))
if out['passed']:
    assert len(out['accounting'])==12 and all(r[2:4]==['COMPLETED','0:0'] for r in out['accounting'])
    assert len(observed_logs)==4 and all(t['network']==['IBext_v8'] and t['plugin_channels']>0 and t['socket_channels']==0 for t in observed_logs)
    ranks=out['ranks']
    pins=read('src/PROVIDERS.json')['files']
    for rank,(parent,node) in enumerate(PLACEMENTS):
        r=ranks[rank]
        assert r==read(f'ranks/{rank}.json') and r['rank']==rank and r['node']==node and r['step'].startswith(parent+'.')
        assert r['step']==out['gpu']['steps'][rank]['row'][0]
        for phase in ('pre','gpu-gate','post'):
            snap=read(f'{phase}/{node}.json')
            assert snap['boot_id']==r['boot_id'] and len(set(snap['pci']))==8 and r['pci'] in snap['pci'] and snap['kfd_holder_count']==0
        trace=f'traces/{rank}-first-allreduce-returned.json'
        t=read(trace)
        assert t['step']==r['step'] and t['node']==node and all(t['providers'].get(k)==v for k,v in pins.items())
        trace_pins[trace]=sha(W/trace)
    for size, *rank_rows in zip((8192,131072,4194304),*(r['rows'] for r in ranks),strict=True):
        assert all(r['bytes']==size and r['checked']==25 and len(r['latency_ns'])==20
                   for r in rank_rows)
        samples=zip(*(r['latency_ns'] for r in rank_rows),strict=True)
        numeric.append(dict(bytes=size,median_slowest_rank_us=statistics.median(map(max,samples))/1000))
print(json.dumps(dict(state='ORIGINAL_TERMINAL_VERIFIED',passed=out['passed'],accounting=out['accounting'],
    numeric=numeric,network=observed_logs,trace_pins=trace_pins,planned_ranks=4,
    completed_ranks=len(out['ranks']),outcome_SHA256=sha(W/'outcome.json'),claim=CLAIM,request_id=REQUEST)))
'''
payload=(f"CLAIM={intent['claim']!r}\nREQUEST={intent['request_id']!r}\nEXPECTED={expected!r}\n"+remote).encode()
try:
    reply=subprocess.run(config['ssh'],input=payload,capture_output=True,timeout=45)
except subprocess.TimeoutExpired:
    print(json.dumps(dict(transport_rc=None,error='TimeoutExpired')))
    raise SystemExit(1)
result=dict(transport_rc=reply.returncode,stderr_SHA256=hashlib.sha256(reply.stderr).hexdigest())
if reply.returncode==0:result['value']=json.loads(reply.stdout)
else:result['errors']=[x for x in ('AssertionError','Connection closed','Connection reset','Permission denied','SyntaxError','NameError') if x.encode() in reply.stderr]
tag=sys.argv[1]
assert tag in ('first','second','third','fourth','fifth')
with (D/f'{tag}-independent.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
