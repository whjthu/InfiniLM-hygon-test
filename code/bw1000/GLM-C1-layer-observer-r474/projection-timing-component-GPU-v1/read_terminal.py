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
import hashlib,json,re,subprocess
w=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM474-projection-timing-component-GPU-v1')
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
assert len(rows)==3 and all(row[-1]=='f13r1n05' for row in rows)
for name,digest in i['source_files'].items():
    assert Path(name).name==name and sha(w/'src'/name)==digest
for phase in phases:
    receipt=read(w/(phase+'-receipt.json'))
    log=w/(phase+'.log')
    assert receipt==o[phase] and receipt['terminal_visible'] and receipt['row'] in rows
    assert sha(log)==receipt['log_SHA256'] and log.stat().st_size==receipt['log_bytes']
pre,post=[read(w/phase/'f13r1n05.json') for phase in ['pre','post']]
assert pre['boot_id']==post['boot_id'] and pre['pci']==post['pci'] and len(set(pre['pci']))==8
assert pre['step']==o['pre']['row'][0] and post['step']==o['post']['row'][0]
assert pre['kfd_holder_count']==post['kfd_holder_count']==0 and pre['monotonic_ns']<=post['monotonic_ns']
files={name:sha(w/name) for name in ['gpu-gate/f13r1n05.json','device.json','component.stdout','component.stderr'] if (w/name).is_file()}
failure=[]
for name in ['gpu.log','component.stderr']:
    p=w/name
    if p.is_file():
        for line in p.read_bytes()[-8192:].decode(errors='replace').splitlines():
            if re.search(r'(Error:|AssertionError|host_entry.py|snapshot.py|worker.py|apptainer|srun: error:)',line):
                failure.append(line.strip()[:250])
print(json.dumps(dict(verified=True,passed=o['passed'],accounting=o['accounting'],phase_files=files,
    outcome_SHA256=sha(w/'outcome.json'),same_boot_and_eight_PCI=True,pre_post_KFD_holders=[0,0],failure=failure[-12:])))
'''
args=f"REQUEST={intent['request_id']!r}\nCLAIM={intent['claim']!r}\n"
r=subprocess.run(env['ssh'],input=(args+remote).encode(),capture_output=True,timeout=40)
v=json.loads(r.stdout) if r.returncode==0 else dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),failure_types=[x for x in ('AssertionError','Connection closed','Permission denied','IndexError') if x.encode() in r.stderr])
with (D/'first-independent.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
