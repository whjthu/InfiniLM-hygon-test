"""Compare DDP bucket sizes on all 504 trainable Qwen LoRA tensor shapes."""
from contextlib import nullcontext
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

LAYERS = 36
SHAPES = ((4096,4096),(4096,1024),(4096,1024),(4096,4096),
          (4096,12288),(4096,12288),(12288,4096))
RANK = 16
PROJECTIONS = LAYERS*len(SHAPES)
TENSORS = 2*PROJECTIONS
ELEMENTS = LAYERS*sum((i+o)*RANK for i,o in SHAPES)
BUCKETS = (25,8,8,25)
MICRO = 4
ROWS = 186
ABS_LIMIT = 5e-5


class LoRA(nn.Module):
    def __init__(self, device, seed):
        super().__init__()
        rng = torch.Generator(device=device).manual_seed(seed)
        for layer in range(LAYERS):
            for index,(inputs,outputs) in enumerate(SHAPES):
                a = torch.randn((RANK,inputs),generator=rng,device=device) * .01
                b = torch.zeros((outputs,RANK),device=device)
                self.register_parameter(f'a{layer}_{index}',nn.Parameter(a))
                self.register_parameter(f'b{layer}_{index}',nn.Parameter(b))
        self.params = tuple(self.parameters())
        if len(self.params)!=504 or sum(p.numel() for p in self.params)!=43_646_976:
            raise ValueError('EXACT_QWEN_LORA_PARAMETER_GEOMETRY')

    def forward(self, narrow, wide):
        terms = []
        for ordinal,(a,b) in enumerate(zip(self.params[::2],self.params[1::2])):
            x = narrow if SHAPES[ordinal % len(SHAPES)][0] == 4096 else wide
            y = F.linear(F.linear(x,a),b).float()
            terms.append((2*y-.03125).square().mean())
        return torch.stack(terms).mean()


def exercise(device, rank, seed, deadline, pilot, measured):
    def budget():
        if time.time()>=deadline:
            raise TimeoutError('BUCKET_TRAIN_DEADLINE')

    def stimuli(seed):
        rng=torch.Generator(device=device).manual_seed(seed+rank)
        return tuple(tuple(torch.randn((ROWS,n),generator=rng,device=device,dtype=torch.bfloat16)*.125
                      for n in (4096,12288)) for _ in range(MICRO))

    def snapshot(model):
        return (torch.cat([p.detach().reshape(-1) for p in model.params]),
                torch.cat([p.grad.detach().reshape(-1) for p in model.params]))

    batch=stimuli(seed+80513)
    anchors=None
    result=[]
    timed=2 if pilot else 15
    warm=1 if pilot else 3
    for cap in BUCKETS:
        budget()
        model=LoRA(device,seed)
        ddp=nn.parallel.DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False,
                                                bucket_cap_mb=cap)
        opt=torch.optim.AdamW(model.params,lr=1e-4,foreach=False)
        durations=[]
        numerical=None
        for index in range(1+warm+timed):
            budget()
            torch.cuda.synchronize(device)
            start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for micro,inputs in enumerate(batch):
                with ddp.no_sync() if micro < MICRO-1 else nullcontext():
                    with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
                        loss=ddp(*inputs)/MICRO
                    loss.backward()
            opt.step()
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            if index==0:
                current=snapshot(model)
                if anchors is None:
                    anchors=current
                errs=[(x-y).abs().amax() for x,y in zip(current,anchors)]
                metrics=torch.stack((errs[0],errs[1],loss.detach(),current[1].abs().amax()))
                if not bool(torch.isfinite(metrics).all()) or not bool(metrics[3]>0):
                    raise ValueError('NONFINITE_OR_ZERO_QWEN_SHAPED_GRADIENT')
                dist.all_reduce(metrics[:2],op=dist.ReduceOp.MAX)
                numerical=metrics.tolist()
                if max(numerical[:2])>ABS_LIMIT:
                    raise ValueError('BUCKET_NUMERIC_GATE')
                del current
            elif index> warm:
                measured()
                worst=torch.tensor(elapsed,device=device,dtype=torch.float32)
                dist.all_reduce(worst,op=dist.ReduceOp.MAX)
                durations.append(worst.item())
        info=ddp._get_ddp_logging_data()
        if info.get('num_parameter_tensors')!=TENSORS or info.get('bucket_cap_bytes')!=cap*1024**2:
            raise ValueError('ACTUAL_DDP_BUCKET_CONFIGURATION')
        result.append(dict(bucket_cap_mib=cap,timed_steps=durations,numeric=numerical,
                           actual_initial_bucket_sizes=info.get('bucket_sizes'),
                           actual_rebuilt_bucket_sizes=info.get('rebuilt_bucket_sizes')))
        del opt,ddp,model
        dist.barrier()
    return dict(caps=result, parameter_tensors=TENSORS,parameter_elements=ELEMENTS,
                tokens=ROWS,microbatches=MICRO, timed_steps_per_arm=timed,
                synthetic_operator_training_only=True)
