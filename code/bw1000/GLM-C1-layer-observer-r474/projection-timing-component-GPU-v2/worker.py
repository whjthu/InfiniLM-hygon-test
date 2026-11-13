"""Verify the pinned loader closure and run the already linked diagnostic once."""
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
from assigned_kfd import assigned_kfd

root=Path(__file__).resolve().parent.parent
read=lambda path:json.loads(path.read_bytes())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
intent=read(root/'intent.json')
binding=read(root/'src/BINDING.json')
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
gate=read(root/'gpu-gate/f13r1n05.json')
assert gate['step']==step and gate['kfd_holder_count']==0
assert gate['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert all(sha(root/'src'/n)==h for n,h in intent['source_files'].items())
binary=Path(binding['binary'])
assert sha(binary)==binding['binary_SHA256']
original=Path('/refactor/candidates/GLM-C1-layer-observer-r474-CPU-v1/CPU-result.json')
assert sha(original)==binding['parent_receipt_SHA256']
expected=read(original)['new_transitive_library_SHA256'] | {binding['candidate']:binding['candidate_SHA256']}
probe=subprocess.run(['/usr/bin/ldd','-r',str(binary)],capture_output=True,text=True,timeout=30)
text=probe.stdout+'\n'+probe.stderr
assert probe.returncode==0 and 'not found' not in text and 'undefined symbol' not in text
actual=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M))
assert actual==set(expected) and len(expected)==72
for path,pin in expected.items():
    assert sha(Path(path))==pin,path
assert 0<=time.monotonic_ns()-gate['monotonic_ns']<=90_000_000_000
left=intent['work_deadline']-time.time()
assert left>0
with (root/'component.stdout').open('xb') as output, (root/'component.stderr').open('xb') as error:
    proc=subprocess.run([str(binary)],stdout=output,stderr=error,timeout=min(90,left))
assert proc.returncode==0, f'component return code {proc.returncode}'
raw=(root/'component.stdout').read_bytes()
assert 0<len(raw)<16384
*arms,final=[json.loads(line) for line in raw.splitlines() if line.startswith(b'{')]
assert len(arms)==6 and final['completed'] is True and final['runtime_released'] is True
pci=final['pci'].lower()
if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):
    pci='0000:'+pci
assert pci in gate['pci']
kfd=assigned_kfd(pci)
for index,(arm,mode) in enumerate(zip(arms,[False,True,True,False,False,True],strict=True)):
    assert arm['arm']==index and arm['timing'] is mode and arm['bf16_exact'] is True
    assert arm['free_bytes']>=8*1024**3
    samples=arm['samples']
    assert len(samples)==(2 if mode else 0)
    if mode:
        for row,geometry,successor in zip(samples,[(2048,6144,1),(1024,2048,1)],[1,0],strict=True):
            assert len(row)==9 and tuple(row[:3])==geometry and row[3:5]==[0,0] and row[8]==successor
            assert all(type(n) is int and n>=0 for n in row)
            if not successor:assert row[7]==0
with (root/'device.json').open('x') as f:
    json.dump(dict(passed=True,step=step,pci=pci,kfd=kfd,arms=arms,runtime_released=True,
        binary_SHA256=binding['binary_SHA256'],component_stdout_SHA256=hashlib.sha256(raw).hexdigest(),
        verified_provider_count=len(expected)),f,indent=2)
