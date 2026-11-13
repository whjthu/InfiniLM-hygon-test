"""Verify the pinned loader closure and run the already linked diagnostic once."""
import hashlib,json,os,re,statistics,subprocess,time
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
original=Path('/refactor/candidates/GLM-C1-layer-observer-r474-CPU-v1/CPU-result.json')
assert sha(original)==binding['parent_receipt_SHA256']
providers=read(original)['new_transitive_library_SHA256']
assert len(providers)==71
for path,pin in providers.items():
    assert sha(Path(path))==pin,path
for arm,info in binding['binaries'].items():
    binary=Path(info['path'])
    assert sha(binary)==info['SHA256']
    assert sha(Path(info['library']))==info['library_SHA256']
    check=subprocess.run(['/usr/bin/ldd','-r',str(binary)],capture_output=True,text=True,timeout=20)
    text=check.stdout+'\n'+check.stderr
    assert check.returncode==0 and 'not found' not in text and 'undefined symbol' not in text
    actual=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M))
    assert actual==set(providers)|{info['library']},arm
assert 0<=time.monotonic_ns()-gate['monotonic_ns']<=90_000_000_000
arms=[]
reference=None
card=None
for index,arm in enumerate(binding['order']):
    info=binding['binaries'][arm]
    prefix=root/f'arm-{index}'
    left=intent['work_deadline']-time.time()
    assert left>0
    with prefix.with_suffix('.stdout').open('xb') as output, prefix.with_suffix('.stderr').open('xb') as error:
        proc=subprocess.run([info['path']],stdout=output,stderr=error,timeout=min(30,left))
    assert proc.returncode==0, f'arm {index} return code {proc.returncode}'
    raw=prefix.with_suffix('.stdout').read_bytes()
    assert 0<len(raw)<16384
    values,final=[json.loads(line) for line in raw.splitlines()]
    assert final['completed'] is True and final['runtime_released'] is True
    pci=final['pci'].lower()
    if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):
        pci='0000:'+pci
    assert pci in gate['pci']
    device=assigned_kfd(pci)
    if card is None:card=device
    assert card==device
    observed=bytes.fromhex(values['output_hex'])
    assert len(observed)==6144 and values['warmups']==5 and values['min_free_bytes']>=8*1024**3
    samples=values['timing_ns']
    assert len(samples)==32 and all(type(t) is int and t>0 for t in samples)
    if reference is None:reference=observed
    assert reference==observed, f'arm {index} BF16 byte mismatch'
    arms.append(dict(index=index,arm=arm,pci=pci,binary_SHA256=info['SHA256'],
        output_SHA256=hashlib.sha256(observed).hexdigest(),stdout_SHA256=hashlib.sha256(raw).hexdigest(),
        min_free_bytes=values['min_free_bytes'],warmups=5,timing_ns=samples,median_ns=statistics.median(samples)))
ratios=[]
for left,right in zip(arms[::2],arms[1::2],strict=True):
    pair={row['arm']:row['median_ns'] for row in (left,right)}
    ratios.append(pair['single-drain']/pair['baseline'])
with (root/'device.json').open('x') as output:
    json.dump(dict(passed=True,step=step,pci=card['pci'],kfd=card,arms=arms,runtime_released=True,
        paired_median_ratios=ratios,verified_provider_count=72),output,indent=2)
