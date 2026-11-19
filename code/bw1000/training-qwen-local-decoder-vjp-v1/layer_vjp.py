"""Run direct and native-checkpoint VJPs against the same captured decoder boundary."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import torch


MAX_BYTES = 32 * 1024**2
MAX_REPLAY_SECONDS = 90


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        need(value.layout == torch.strided and value.numel() * value.element_size() <= MAX_BYTES, 'Unexpected decoder operand')
        out = value.detach().clone(memory_format=torch.preserve_format)
        need(out.stride() == value.stride(), 'Capture changed decoder operand layout')
        return out
    if type(value) is dict:
        return {key: clone_tree(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(clone_tree(item) for item in value)
    need(value is None or type(value) in (bool,int,float,str), 'Nonempty cache or unexpected decoder argument')
    return value


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if type(value) is dict:
        return {key: cpu_tree(item) for key, item in value.items()}
    if type(value) in (list,tuple):
        return type(value)(cpu_tree(item) for item in value)
    return value


def record(tensor):
    value = tensor.detach().cpu()
    need(bool(torch.isfinite(value).all()), 'Nonfinite decoder operand or gradient')
    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),
                bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def difference(a,b):
    aa,bb = a.detach().cpu().double(),b.detach().cpu().double()
    need(aa.shape == bb.shape, 'Decoder VJP shapes differ')
    diff = aa - bb
    den = float(aa.square().sum().sqrt())
    result = dict(exact=torch.equal(a.detach().cpu(),b.detach().cpu()),
                  max_abs=float(diff.abs().max()),relative_l2=float(diff.square().sum().sqrt())/den if den else None,
                  changed=int(torch.count_nonzero(diff)))
    need(math.isfinite(result['max_abs']) and (result['relative_l2'] is None or math.isfinite(result['relative_l2'])),
         'Nonfinite decoder VJP comparison')
    return result


def versions(layer):
    return {n: (p._version, p.data_ptr()) for n,p in layer.named_parameters()}


def replay(layer, capture, output, rank, hard_deadline):
    # The runtime has ended native accumulation. Autograd returns only dX.
    # Native non-reentrant checkpointing is required to avoid accumulating parameter gradients.
    use_reentrant = getattr(layer._gradient_checkpointing_func, 'keywords', {}).get('use_reentrant')
    if use_reentrant is not False:
        return dict(state='NATIVE_CHECKPOINT_REENTRANT_OR_UNKNOWN_CAPTURE_ONLY',use_reentrant=use_reentrant)
    args = capture['args']
    x = args[0]
    device = x.device
    saved_cpu,saved_gpu = torch.get_rng_state(),torch.cuda.get_rng_state(device)
    started=time.monotonic()
    local_deadline=min(hard_deadline,time.time()+MAX_REPLAY_SECONDS)
    variants = []
    initial_versions=versions(layer)
    try:
        for label in ('direct','checkpoint','checkpoint','direct'):
            if time.time() >= local_deadline:
                raise TimeoutError('Decoder replay deadline exhausted')
            torch.set_rng_state(capture['cpu_rng'])
            torch.cuda.set_rng_state(capture['gpu_rng'], device)
            inp = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
            with torch.enable_grad(), torch.autocast('cuda',dtype=capture['autocast_dtype'],enabled=capture['autocast_enabled']):
                values=(inp,*args[1:])
                if label=='direct':
                    out=torch.nn.Module.__call__(layer,*values,**capture['kwargs'])
                else:
                    out=layer(*values,**capture['kwargs'])
                grad,=torch.autograd.grad(out,inp,capture['grad_out'])
            torch.cuda.synchronize(device)
            variants.append((label,out.detach().cpu(),grad.detach().cpu()))
    finally:
        torch.set_rng_state(saved_cpu)
        torch.cuda.set_rng_state(saved_gpu,device)
    need(initial_versions == versions(layer),'Decoder replay modified a parameter')
    direct,d1,d2,direct2 = variants
    reference_out=capture['output'].detach().cpu()
    reference_grad=capture['grad_in'].detach().cpu()
    samples=[dict(route=label,output=record(o),grad_in=record(g),output_vs_training=difference(reference_out,o),
                  grad_vs_training=difference(reference_grad,g)) for label,o,g in variants]
    return dict(state='ACTUAL_FOUR_VJP_ABBA_COMPLETE',use_reentrant=use_reentrant,elapsed_seconds=time.monotonic()-started,
                variants=samples,direct_repeat=difference(direct[2],direct2[2]),
                checkpoint_repeat=difference(d1[2],d2[2]),direct_vs_checkpoint=difference(direct[2],d1[2]))


def save_operands(output,rank,capture):
    private=output/'decoder-operands'
    private.mkdir(mode=0o700,exist_ok=True)
    # Private path on the experiment filesystem. Metadata is the only reported content.
    artifact=private/f'rank-{rank:02d}.pt'
    fd=os.open(artifact,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:
        torch.save(cpu_tree(capture),f)
    size=artifact.stat().st_size
    need(0<size<=MAX_BYTES,'Private capture size outside frozen bound')
    state=hashlib.sha256()
    with artifact.open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):
            state.update(data)
    return dict(bytes=size,SHA256=state.hexdigest())
