from pathlib import Path
import ast,hashlib,json,subprocess
D=Path(__file__).resolve().parent
tree=ast.parse((D.parent/'read_extra40_approval_original_handles_once.py').read_bytes())
i=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
env={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'known-readonly-transport','exec'),env)
intent=json.loads((D/'launch-intent.json').read_bytes())
remote=r'''
from pathlib import Path
import hashlib,json,re,statistics,subprocess
w=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM474-TP16-native-collective-v1')
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
i=read(w/'intent.json')
o=read(w/'outcome.json')
assert i['request_id']==o['request_id']==REQUEST and i['claim']==o['claim']==CLAIM
assert o['accounting_columns']==['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
phases=['pre','gpu','post']
labels={w.name+'-'+phase for phase in phases}
query=lambda args:subprocess.check_output(args,text=True,timeout=10).splitlines()
rows=[r.split('|') for r in query(['sacct','-n','-P','-j','1103606','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']) if r.split('|')[1] in labels]
assert sorted(rows)==sorted(o['accounting'])
assert not any(r.split('|')[1] in labels for r in query(['squeue','--steps','-h','-j','1103606','-o','%i|%j']))
assert len(rows)==3 and all(set(query(['scontrol','show','hostnames',row[-1]]))==set(i['nodes']) for row in rows)
for name,digest in i['source_files'].items():
    assert Path(name).name==name and sha(w/'src'/name)==digest
for phase in phases:
    receipt=read(w/(phase+'-receipt.json'))
    log=w/(phase+'.log')
    assert receipt==o[phase] and receipt['terminal_visible'] and receipt['row'] in rows
    assert sha(log)==receipt['log_SHA256'] and log.stat().st_size==receipt['log_bytes']
device=o['device']
files={}
for node in i['nodes']:
    pre,post=[read(w/phase/(node+'.json')) for phase in ('pre','post')]
    assert pre['boot_id']==post['boot_id'] and pre['pci']==post['pci'] and len(set(pre['pci']))==8
    assert pre['step']==o['pre']['row'][0] and post['step']==o['post']['row'][0]
    assert pre['kfd_holder_count']==post['kfd_holder_count']==0 and pre['monotonic_ns']<=post['monotonic_ns']
    if o['passed']:
        local=[r for r in device['ranks'] if r['node']==node]
        assert sorted(r['pci'] for r in local)==pre['pci'] and len({r['kfd_id'] for r in local})==8
failure=[]
for line in (w/'gpu.log').read_text(errors='replace').splitlines():
    if re.search(r'Error:|srun: error:',line):
        failure.append(line.strip()[:160])
entered=sum((w/'ranks'/f'{rank:02d}-entered.json').is_file() for rank in range(16))
if o['passed']:
    assert entered==16 and device==read(w/'device.json') and device['step']==o['gpu']['row'][0]
    assert len(device['ranks'])==len(device['rank_SHA256'])==16
    assert len({r['uid_SHA256'] for r in device['ranks']})==1
    files['device.json']=sha(w/'device.json')
    expected=[('allreduce',1024),('allgather',1024),('allreduce',6144),('allgather',6144)]
    import struct
    def bf(value):return struct.pack('<H',struct.unpack('<I',struct.pack('<f',value))[0]>>16)
    for rank,(row,pin) in enumerate(zip(device['ranks'],device['rank_SHA256'],strict=True)):
        file=w/'ranks'/f'{rank:02d}.json'
        assert sha(file)==pin and row==read(file)
        assert row['rank']==rank and row['node']==i['nodes'][rank//8] and row['step']==device['step']
        assert row['library_SHA256']==read(w/'src/BINDING.json')['SHA256']
        assert row['comm_destroyed'] and row['min_free_bytes']>=8*1024**3
        assert [(c['operation'],c['elements']) for c in row['checks']]==expected
        for c in row['checks']:
            n=c['elements']
            value=bf(136)*n if c['operation']=='allreduce' else b''.join(bf(r+1)*n for r in range(16))
            assert c['output_SHA256']==hashlib.sha256(value).hexdigest()
            assert c['checked']==8 and len(c['latency_ns'])==5 and all(t>0 for t in c['latency_ns'])
print(json.dumps(dict(device=device,verified=True,passed=o['passed'],accounting=o['accounting'],phase_files=files,
    outcome_SHA256=sha(w/'outcome.json'),same_boot_and_eight_PCI=True,pre_post_KFD_holders=[0,0],failure=failure[-12:],ranks_entered=entered)))
'''
args=f"REQUEST={intent['request_id']!r}\nCLAIM={intent['claim']!r}\n"
r=subprocess.run(env['ssh'],input=(args+remote).encode(),capture_output=True,timeout=40)
v=json.loads(r.stdout) if r.returncode==0 else dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),failure_types=[x for x in ('AssertionError','Connection closed','Permission denied','IndexError') if x.encode() in r.stderr])
with (D/'first-independent.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(dict(passed=v.get("passed"),verified=v.get("verified"),accounting=v.get("accounting"),failure=v.get("failure"),ranks_entered=v.get("ranks_entered"),ranks_passed=len((v.get("device") or {}).get("ranks",[])),rc=v.get("rc"))))
