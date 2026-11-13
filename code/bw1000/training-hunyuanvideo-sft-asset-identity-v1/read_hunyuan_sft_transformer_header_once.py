"""Inspect only the header and optional published identity companion for the native weight."""
from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parent
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
first=B/'hunyuan-sft-original-and-target-asset-identity-v1.json'
e=json.loads(first.read_bytes())['A100']['value']['sft']
name='diffusion_pytorch_model.safetensors'
path=e['weights']+'/transformer/480p_t2v/'+name
expected=e['single_file_candidates'][name]
program=r'''
from collections import Counter
from pathlib import Path
import hashlib,json,math,re
p=Path(WEIGHT)
def info(s):return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
before=info(p.stat())
assert not p.is_symlink() and all(v==EXPECTED[k] for k,v in before.items())
with p.open('rb') as f:
    length=f.read(8)
    assert len(length)==8
    n=int.from_bytes(length,'little')
    assert 0<n<=1024**2
    raw=f.read(n)
    assert len(raw)==n
def pairs(items):
    value=dict(items)
    assert len(value)==len(items)
    return value
doc=json.loads(raw,object_pairs_hook=pairs)
metadata=doc.pop('__metadata__',{})
dtype_width={'BF16':2,'F16':2,'F32':4,'F64':8,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
dtype=Counter()
end=0
elements=0
for name,s in sorted(doc.items(),key=lambda row:tuple(row[1]['data_offsets'])):
    assert set(s)=={'dtype','shape','data_offsets'}
    assert isinstance(name,str) and name and all(type(x) is int and x>=0 for x in s['shape'])
    start,stop=s['data_offsets']
    count=math.prod(s['shape'])
    assert type(start) is type(stop) is int and start==end and stop-start==count*dtype_width[s['dtype']]
    end=stop
    elements+=count
    dtype[s['dtype']]+=1
assert end+n+8==before['bytes'] and info(p.stat())==before
companion=p.with_name(p.name+'.source-sha256')
companion_hash=None
if companion.is_file() and not companion.is_symlink() and 0<companion.stat().st_size<=128:
    v=companion.read_text().strip()
    if re.fullmatch('[a-f0-9]{64}',v):companion_hash=v
print(json.dumps(dict(path=str(p),stat=before,header_bytes=n,header_SHA256=hashlib.sha256(raw).hexdigest(),
    tensor_keys=len(doc),dtype_tensor_counts=dict(dtype),tensor_elements=elements,tensor_payload_bytes=end,
    companion_SHA256=companion_hash,wholefile_SHA256_computed=False,weight_payload_bytes_read=0,GPU_calls=0)))
'''
args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),
      'a100-26','timeout 40 python3 -B -u -']
r=subprocess.run(args,input=(f'WEIGHT={path!r}\nEXPECTED={expected!r}\n'+program).encode(),capture_output=True,timeout=50)
v=dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),preflight_SHA256=hashlib.sha256(first.read_bytes()).hexdigest())
if r.returncode==0:v['value']=json.loads(r.stdout)
else:v['failure_types']=[n for n in ('AssertionError','Connection closed','Permission denied','KeyError') if n.encode() in r.stderr]
with (B/'hunyuan-sft-original-transformer-header-v1.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
