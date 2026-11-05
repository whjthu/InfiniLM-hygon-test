"""CPU-only passive inventory. No device imports or TCP connections."""
from pathlib import Path
import json,os,re,socket
D=Path(__file__).resolve().parent;q=json.loads((D/'config.json').read_bytes())
node=socket.gethostname().split('.')[0];assert node in q['nodes']
pci=sorted({p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')})
assert len(pci)==8 and all(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}[.][0-7]',v)for v in pci)
kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir()
pids=sorted(int(p.name)for p in kfd.iterdir()if p.name.isdigit());matching=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit()or int(p.name)==os.getpid():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    if any((b'/node.py'in a and b'Kimi-vocab-parallel-embedding-r461'in a)or b'public_kimi_native'in a or b'public_kimi_text'in a for a in args):matching.append(int(p.name))
ports={}
for port in q['ports']:
    with socket.socket()as s:
        try:s.bind(('0.0.0.0',port));ports[str(port)]=True
        except OSError:ports[str(port)]=False
expected=json.loads(os.environ.get('K461_PIDS','{}')).get(node,[])
alive=[p for p in expected if Path('/proc/%d'%p).exists()]
v=dict(node=node,CPU_only=True,GPU_initialized=False,physical_PCI=pci,kfd_pids=pids,matching=matching,
    expected_PIDs_still_exist=alive,ports_bindable=ports,TCP_connections=0,
    actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
print(json.dumps(v),flush=True)
assert not pids and not matching and not alive and all(ports.values())
