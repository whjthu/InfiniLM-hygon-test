"""Compare the original Qwen first forwards using only already-saved target archives."""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import time

import torch

ROOT=Path('/work')
RANKS=(0,1,4,5)
EXPECTED_LABELS=('training-qwen-first-resumed-forward-v1','training-qwen-resumed-decoder-internals-v1')


def need(ok,reason):
    if not ok:raise ValueError(reason)


def exact(path,pin):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size==pin['bytes'], 'ORIGINAL_ARTIFACT_BYTES')
    hasher=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):hasher.update(chunk)
    after=path.lstat()
    get=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(get(before)==get(after) and hasher.hexdigest()==pin['SHA256'], 'ORIGINAL_ARTIFACT_DIGEST')
    return before.st_size


def tensors(a,b,path):
    need(a.shape==b.shape and a.dtype==b.dtype and a.device.type==b.device.type=='cpu', 'MATCHED_TENSOR_LAYOUT')
    need(a.is_contiguous() and b.is_contiguous(), 'CONTIGUOUS_ORIGINAL_OUTPUT')
    need(a.numel()<=300_000_000, 'ORIGINAL_OUTPUT_ELEMENT_BUDGET')
    different=0
    first=None
    max_abs=error_sq=base_sq=0.0
    left=a.view(-1)
    right=b.view(-1)
    for i in range(0,left.numel(),262144):
        x=left[i:i+262144]
        y=right[i:i+262144]
        if a.is_floating_point():
            need(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), 'NONFINITE_ORIGINAL_OUTPUT')
        equal=x==y
        count=int((~equal).count_nonzero())
        if not count:
            if a.is_floating_point():
                base_sq+=float(x.double().square().sum())
            continue
        different+=count
        if first is None:first=i+int((~equal).nonzero()[0])
        if a.is_floating_point():
            diff=x.double()-y.double()
            max_abs=max(max_abs,float(diff.abs().max()))
            error_sq+=float(diff.square().sum())
            base_sq+=float(x.double().square().sum())
    need(all(math.isfinite(v) for v in (max_abs,error_sq,base_sq)), 'INVALID_NUMERIC_SUMMARY')
    return dict(path=path,shape=list(a.shape),dtype=str(a.dtype),elements=a.numel(),different=different,
        first_flat_mismatch=first,contains_nonfinite=False,
        max_abs=max_abs if a.is_floating_point() else None,
        relative_l2=(math.sqrt(error_sq/base_sq) if base_sq else None) if a.is_floating_point() else None)


def tree(a,b,path='result'):
    need(type(a) is type(b), 'MATCHED_OUTPUT_FIELD_TYPE')
    if isinstance(a,torch.Tensor):
        yield tensors(a,b,path)
    elif isinstance(a,dict):
        need(set(a)==set(b) and all(isinstance(k,str) for k in a), 'MATCHED_OUTPUT_FIELD_KEYS')
        for k in sorted(a):
            yield from tree(a[k],b[k],path+'.'+k)
    elif isinstance(a,(list,tuple)):
        need(len(a)==len(b), 'MATCHED_OUTPUT_FIELD_LENGTH')
        for i,(x,y) in enumerate(zip(a,b)):
            yield from tree(x,y,path+f'[{i}]')
    else:
        need(type(a) in (type(None),str,int,bool,float) and a==b, 'MATCHED_NONTENSOR_FIELD')


def one(rank,entries):
    values=[]
    bytes_read=0
    for side,entry in zip(('a','b'),entries):
        folder=Path('/'+side)/f'rank-{rank:02d}'
        receipt=folder/'receipt.json'
        bytes_read+=exact(receipt,entry['receipt'])
        index=json.loads(receipt.read_bytes())
        need(index['rank']==rank and index['world_size']==8 and index['completed_global_step']==2, 'ORIGINAL_CAPTURE_IDENTITY')
        recorded={f['name']:{k:f[k] for k in ('bytes','SHA256')} for f in index['files']}
        need(all(recorded[n]==pin for n,pin in entry['files'].items()), 'ORIGINAL_CAPTURE_INDEX_PINS')
        path=folder/'forward-01-output.pt'
        bytes_read+=exact(path,entry['files'][path.name])
        value=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        need(value['global_step']==0 and value['micro_ordinal']==1, 'ORIGINAL_FIRST_FORWARD')
        values.append(value['result'])
    before,after=values
    need('logits' in before and 'loss' in before, 'REAL_LOSS_AND_LOGITS_REQUIRED')
    need(before['logits'].shape[-1]==151936, 'ORIGINAL_QWEN_VOCAB_DIM')
    need(before['loss'].numel()==after['loss'].numel()==1, 'ORIGINAL_SCALAR_LOSS')
    fields=list(tree(before,after))
    logits,=[r for r in fields if r['path']=='result.logits']
    loss=[float(r['loss'].item()) for r in values]
    need(all(math.isfinite(x) for x in loss), 'FINITE_ORIGINAL_LOSS')
    if rank==0:need(not any(f['different'] for f in fields), 'BYTE_EQUAL_CONTROL_NOT_NUMERICALLY_EQUAL')
    return dict(rank=rank,first_forward_loss=loss,loss_absolute_delta=abs(loss[0]-loss[1]),
        logits=logits,all_tensor_fields=fields,bytes_stream_hashed=bytes_read,
        initial_hash_matches=entries[0]['files']['initial.pt']==entries[1]['files']['initial.pt'],
        complete_input_and_RNG_archive_hash_matches=entries[0]['files']['forward-01-input.pt']==entries[1]['files']['forward-01-input.pt'],
        same_serialized_output=entries[0]['files']['forward-01-output.pt']==entries[1]['files']['forward-01-output.pt'])


def main():
    started=time.monotonic()
    assert socket.gethostname().split('.')[0]=='f13r1n10'
    assert os.environ['SLURM_JOB_ID']=='1103606' and os.environ['SLURM_STEP_ID'].isdigit()
    assert all(os.environ[name]=='-1' for name in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
    torch.set_num_threads(4)
    pins=json.loads((ROOT/'ARCHIVES.json').read_bytes())
    assert set(pins['ranks'])=={str(r) for r in RANKS}
    assert sum(e['files']['forward-01-output.pt']['bytes'] for v in pins['ranks'].values() for e in v)<=5*2**30
    items=[one(rank,pins['ranks'][str(rank)]) for rank in RANKS]
    assert not torch.cuda.is_initialized()
    result=dict(node=socket.gethostname().split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        verified_ranks=items,old_GPU_steps=['1087656.429','1087656.433'],selected_shape_scope='first native training microbatch',
        actual_GPU_calls=0,private_tensor_export_bytes=0,torch_version=str(torch.__version__),wall_seconds=time.monotonic()-started)
    with (ROOT/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,allow_nan=False)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
