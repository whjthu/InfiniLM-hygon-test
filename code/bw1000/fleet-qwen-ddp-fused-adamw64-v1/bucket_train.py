"""Compare fused AdamW groups of 64 to foreach on Qwen-shaped DDP updates."""
from contextlib import nullcontext
import json
from pathlib import Path
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
BUCKET_CAP = 25
FUSED_ORDER = (False, True, True, False)
FUSED_GROUP_SIZE = 64
MOMENT_LIMIT = 5e-6
MICRO = 4
ROW_METADATA = Path(__file__).with_name("ROW_METADATA.json")
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
        return tuple(
            tuple(
                tuple(torch.randn((rows, width), generator=rng, device=device, dtype=torch.bfloat16)*.125
                      for width in (4096, 12288))
                for rows in step_rows)
            for step_rows in rows_by_update)

    def snapshot(model, optimizer):
        parameters = torch.cat([p.detach().reshape(-1) for p in model.params])
        gradients = torch.cat([p.grad.detach().reshape(-1) for p in model.params])
        moments = tuple(torch.cat([optimizer.state[p][key].reshape(-1) for p in model.params])
                        for key in ("exp_avg", "exp_avg_sq"))
        counters = torch.stack([optimizer.state[p]["step"].to(device) for p in model.params])
        return (parameters, gradients, *moments), counters

    if not 0 <= rank < 8:
        raise ValueError('ORIGINAL_LOCAL_RANK_RANGE')
    metadata=json.loads(ROW_METADATA.read_bytes())
    rows_by_update=metadata['rows_by_rank_and_update'][rank]
    if len(rows_by_update) != 2 or any(len(row) != MICRO for row in rows_by_update):
        raise ValueError('ORIGINAL_TWO_UPDATE_ROW_SCHEDULE')
    batch=stimuli(seed+80513)
    anchors={}
    result=[]
    timed=2 if pilot else 45
    warm=1 if pilot else 3
    for fused in FUSED_ORDER:
        budget()
        model=LoRA(device,seed)
        ddp=nn.parallel.DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False,
                                                bucket_cap_mb=BUCKET_CAP, gradient_as_bucket_view=True)
        groups = [{"params": model.params[start:start+FUSED_GROUP_SIZE]}
                  for start in range(0, TENSORS, FUSED_GROUP_SIZE)] if fused else [{"params": model.params}]
        opt = torch.optim.AdamW(groups, lr=1e-4, foreach=not fused, fused=fused)
        group_sizes = [len(group["params"]) for group in opt.param_groups]
        if (group_sizes != ([64] * 7 + [56] if fused else [504])
                or any(group["fused"] != fused or group["foreach"] == fused for group in opt.param_groups)):
            raise ValueError("EXACT_ADAMW_ARMS")
        durations=[]
        optimizer_durations = []
        optimizer_start = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        numerical=[]
        live_grad_storages=None
        steady_allocated=[]
        steady_peaks=[]
        for index in range(1+warm+timed):
            budget()
            torch.cuda.synchronize(device)
            if index > warm:
                torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter()
            timed_step = index > warm
            opt.zero_grad(set_to_none=True)
            if index == 1 and any(p.grad is not None for p in model.params):
                raise ValueError("ORIGINAL_SET_TO_NONE")
            for micro,inputs in enumerate(batch[index % len(batch)]):
                with ddp.no_sync() if micro < MICRO-1 else nullcontext():
                    with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
                        loss=ddp(*inputs)/MICRO
                    loss.backward()
            if timed_step:
                optimizer_start.record()
            opt.step()
            if timed_step:
                optimizer_end.record()
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            if index in (0, 1):
                current, counters = snapshot(model, opt)
                if not bool((counters == index + 1).all()):
                    raise ValueError("ADAMW_UPDATE_COUNTERS")
                if index not in anchors:
                    anchors[index] = current
                errors = [(actual - reference).abs().amax()
                          for actual, reference in zip(current, anchors[index], strict=True)]
                metrics = torch.stack((*errors, loss.detach(), current[1].abs().amax()))
                if not bool(torch.isfinite(metrics).all()) or not bool(metrics[-1] > 0):
                    raise ValueError("NONFINITE_OR_ZERO_QWEN_SHAPED_GRADIENT")
                dist.all_reduce(metrics[:4], op=dist.ReduceOp.MAX)
                record = metrics.tolist()
                if max(record[:2]) > ABS_LIMIT or max(record[2:4]) > MOMENT_LIMIT:
                    raise ValueError("ADAMW_NUMERIC_GATE")
                numerical.append(record)
                del current
            elif index> warm:
                measured()
                worst=torch.tensor(elapsed,device=device,dtype=torch.float32)
                dist.all_reduce(worst,op=dist.ReduceOp.MAX)
                durations.append(worst.item())
                worst_optimizer = torch.tensor(optimizer_start.elapsed_time(optimizer_end)*0.001,
                                               device=device, dtype=torch.float32)
                dist.all_reduce(worst_optimizer, op=dist.ReduceOp.MAX)
                optimizer_durations.append(worst_optimizer.item())
                steady_allocated.append(torch.cuda.memory_allocated(device))
                steady_peaks.append(torch.cuda.max_memory_allocated(device))
                if live_grad_storages is None:
                    live_grad_storages=len({p.grad.untyped_storage().data_ptr() for p in model.params})
        info=ddp._get_ddp_logging_data()
        if (info.get('num_parameter_tensors')!=TENSORS or info.get('bucket_cap_bytes')!=BUCKET_CAP*1024**2
                or info.get('gradient_as_bucket_view')!=1
                or live_grad_storages is None or not 0 < live_grad_storages < TENSORS):
            raise ValueError('ACTUAL_DDP_BUCKET_CONFIGURATION')
        result.append(dict(bucket_cap_mib=BUCKET_CAP, gradient_as_bucket_view=True, fused_adamw=fused,
                           optimizer_group_sizes=group_sizes, timed_optimizer_seconds=optimizer_durations,
                           timed_steps=durations,numeric=numerical,distinct_live_gradient_storages=live_grad_storages,
                           steady_allocated_bytes=steady_allocated,steady_peak_allocated_bytes=steady_peaks,
                           actual_initial_bucket_sizes=info.get('bucket_sizes'),
                           actual_rebuilt_bucket_sizes=info.get('rebuilt_bucket_sizes')))
        del opt,ddp,model
        dist.barrier()
    return dict(caps=result, parameter_tensors=TENSORS,parameter_elements=ELEMENTS,
                rows_by_update=rows_by_update, microbatches=MICRO, timed_steps_per_arm=timed,
                synthetic_operator_training_only=True)
