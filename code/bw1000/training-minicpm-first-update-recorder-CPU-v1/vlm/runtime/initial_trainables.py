"""Restore pinned trainable initialization at update zero, without restoring RNG."""
from collections.abc import Mapping
import hashlib
from pathlib import Path
import torch


def validate_initialization_callback_order(callbacks,current):
    if not isinstance(callbacks,list) or sum(c is current for c in callbacks)!=1:
        raise ValueError('initialization callback inventory differs')
    recorders=[i for i,c in enumerate(callbacks) if type(c).__name__=='FirstUpdates']
    if len(recorders)!=1 or callbacks.index(current)>=recorders[0]:
        raise ValueError('native alignment callback must precede the original FirstUpdates recorder')
    baselines=[i for i,c in enumerate(callbacks) if type(c).__name__=='BaselineCallback']
    if len(baselines)>1 or any(i<callbacks.index(current) for i in baselines):
        raise ValueError('initialization must precede the baseline initial-parameter digest')


def named_parameter_sha256(named):
    h=hashlib.sha256()
    for name,p in named:
        value=p.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def restore_initial_trainables(named,optimizer,*,global_step,path,file_bytes,file_SHA256,named_SHA256):
    """Validate every tensor before the first copy; failures never authorize training.

    This loads only trainable tensors. Optimizer, scheduler, data cursor and CPU/
    device RNG remain the destination's fresh state. Resume checkpoints use the
    framework's separate complete-state restoration path.
    """
    if type(global_step) is not int or global_step!=0:
        raise ValueError('initial trainables require fresh update zero')
    state=getattr(optimizer,'state',None)
    if not isinstance(state,Mapping) or len(state)!=0:
        raise ValueError('initial trainables require an existing fresh optimizer with empty state')
    named=list(named)
    if (not named or len({n for n,_ in named})!=len(named)
            or len({id(p) for _,p in named})!=len(named)):
        raise ValueError('duplicate/empty trainable parameter inventory')
    if any(not isinstance(n,str) or type(p) is not torch.nn.Parameter or not p.requires_grad
           or not p.is_leaf or p.is_meta or p.grad is not None for n,p in named):
        raise ValueError('expected fresh unsharded trainable parameters without gradients')
    groups=getattr(optimizer,'param_groups',None)
    if not isinstance(groups,list):raise ValueError('optimizer parameter groups absent')
    ids=[id(p) for group in groups for p in group['params']]
    if len(ids)!=len(named) or set(ids)!={id(p) for _,p in named}:
        raise ValueError('optimizer does not own exactly the selected trainable parameters')
    if (type(file_bytes) is not int or not 0<file_bytes<=256*1024**2
            or any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x)
                   for x in (file_SHA256,named_SHA256))):
        raise ValueError('invalid initial artifact identity or byte budget')
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size!=file_bytes:
        raise ValueError('initial artifact path or length differs')
    # Hash and deserialize through the same open descriptor, avoiding a pathname
    # swap between verification and weights-only loading.
    with path.open('rb') as stream:
        h=hashlib.sha256()
        for data in iter(lambda:stream.read(1024**2),b''):h.update(data)
        if h.hexdigest()!=file_SHA256:raise ValueError('initial artifact SHA differs')
        stream.seek(0);source=torch.load(stream,map_location='cpu',weights_only=True)
    if (not isinstance(source,dict) or set(source)!={'parameters','RNG','manifest','global_step'}
            or type(source['global_step']) is not int or source['global_step']!=0
            or not isinstance(source['parameters'],dict)):
        raise ValueError('expected original fresh trainable-state schema')
    parameters=source['parameters']
    if set(parameters)!={n for n,_ in named}:
        raise ValueError('initial parameter names differ')
    expected_manifest=[];staged=[]
    for name,p in named:
        value=parameters[name]
        if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.is_meta
                or value.layout!=torch.strided or tuple(value.shape)!=tuple(p.shape)
                or value.dtype!=p.dtype or not value.is_contiguous()
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError('initial tensor shape/dtype/layout/finite differs: '+name)
        expected_manifest.append(dict(name=name,shape=list(p.shape),dtype=str(p.dtype),numel=p.numel()))
        staged.append((name,value))
    if source['manifest']!=expected_manifest:
        raise ValueError('initial ordered manifest differs')
    if named_parameter_sha256(staged)!=named_SHA256:
        raise ValueError('initial named parameter identity differs')
    # Every schema/name/shape/dtype/digest/freshness check has passed. Device-copy
    # errors remain fatal; no partial-copy state can be accepted as initialized.
    with torch.no_grad():
        for (_,p),(_,value) in zip(named,staged):p.copy_(value)
    if named_parameter_sha256(named)!=named_SHA256:
        raise RuntimeError('destination trainable bytes differ after initialization')
    return dict(file_SHA256=file_SHA256,file_bytes=file_bytes,named_parameters_SHA256=named_SHA256,
                parameter_tensors=len(named),parameter_numel=sum(p.numel() for _,p in named),
                global_step=0,optimizer_state_entries=0,RNG_restored=False,
                scope='Trainable initialization only; not optimizer/checkpoint recovery or numerical qualification.')
