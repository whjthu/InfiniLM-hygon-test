"""Compare grouped retention directly to set-to-none with DDP bucket views fixed."""
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
GROUPED_ZERO = (False, True, True, False)
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

    def snapshot(model):
        return (torch.cat([p.detach().reshape(-1) for p in model.params]),
                torch.cat([p.grad.detach().reshape(-1) for p in model.params]))

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
    for grouped in GROUPED_ZERO:
        budget()
        model=LoRA(device,seed)
        ddp=nn.parallel.DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False,
                                                bucket_cap_mb=BUCKET_CAP, gradient_as_bucket_view=True)
        opt=torch.optim.AdamW(model.params,lr=1e-4,foreach=False)
        durations=[]
        clear_durations=[]
        clear_start=torch.cuda.Event(enable_timing=True)
        clear_end=torch.cuda.Event(enable_timing=True)
        grouped_calls=0
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
            if timed_step:
                clear_start.record()
            if grouped:
                gradients = [p.grad for p in model.params if p.grad is not None]
                if len(gradients) != (0 if index == 0 else TENSORS):
                    raise ValueError('EXACT_RETAINED_GRADIENT_SET')
                if gradients:
                    with torch.no_grad():
                        torch._foreach_zero_(gradients)
                    grouped_calls += 1
            else:
                opt.zero_grad(set_to_none=True)
            if timed_step:
                clear_end.record()
            if index == 1:
                if not all((p.grad is not None) == grouped for p in model.params):
                    raise ValueError('ACTUAL_GRADIENT_RESET_POLICY')
                if grouped and bool(torch.stack(tuple(p.grad.count_nonzero() for p in model.params)).any()):
                    raise ValueError('COMPLETE_GRADIENT_ZERO')
            for micro,inputs in enumerate(batch[index % len(batch)]):
                with ddp.no_sync() if micro < MICRO-1 else nullcontext():
                    with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
                        loss=ddp(*inputs)/MICRO
                    loss.backward()
            opt.step()
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            if index in (0, 1):
                current=snapshot(model)
                if index not in anchors:
                    anchors[index]=current
                errs=[(x-y).abs().amax() for x,y in zip(current,anchors[index])]
                metrics=torch.stack((errs[0],errs[1],loss.detach(),current[1].abs().amax()))
                if not bool(torch.isfinite(metrics).all()) or not bool(metrics[3]>0):
                    raise ValueError('NONFINITE_OR_ZERO_QWEN_SHAPED_GRADIENT')
                dist.all_reduce(metrics[:2],op=dist.ReduceOp.MAX)
                numerical.append(metrics.tolist())
                if max(numerical[-1][:2])>ABS_LIMIT:
                    raise ValueError('BUCKET_NUMERIC_GATE')
                del current
            elif index> warm:
                measured()
                worst=torch.tensor(elapsed,device=device,dtype=torch.float32)
                dist.all_reduce(worst,op=dist.ReduceOp.MAX)
                durations.append(worst.item())
                worst_clear=torch.tensor(clear_start.elapsed_time(clear_end)*0.001,device=device,dtype=torch.float32)
                dist.all_reduce(worst_clear,op=dist.ReduceOp.MAX)
                clear_durations.append(worst_clear.item())
                steady_allocated.append(torch.cuda.memory_allocated(device))
                steady_peaks.append(torch.cuda.max_memory_allocated(device))
                if live_grad_storages is None:
                    live_grad_storages=len({p.grad.untyped_storage().data_ptr() for p in model.params})
        info=ddp._get_ddp_logging_data()
        if (info.get('num_parameter_tensors')!=TENSORS or info.get('bucket_cap_bytes')!=BUCKET_CAP*1024**2
                or info.get('gradient_as_bucket_view')!=1
                or live_grad_storages is None or not 0 < live_grad_storages < TENSORS):
            raise ValueError('ACTUAL_DDP_BUCKET_CONFIGURATION')
        if grouped_calls != (warm+timed if grouped else 0):
            raise ValueError('GROUPED_ZERO_CALL_COUNT')
        result.append(dict(bucket_cap_mib=BUCKET_CAP,gradient_as_bucket_view=True, grouped_zero=grouped,
                           actual_grouped_zero_calls=grouped_calls, timed_clear_seconds=clear_durations,
                           timed_steps=durations,numeric=numerical,distinct_live_gradient_storages=live_grad_storages,
                           steady_allocated_bytes=steady_allocated,steady_peak_allocated_bytes=steady_peaks,
                           actual_initial_bucket_sizes=info.get('bucket_sizes'),
                           actual_rebuilt_bucket_sizes=info.get('rebuilt_bucket_sizes')))
        del opt,ddp,model
        dist.barrier()
    return dict(caps=result, parameter_tensors=TENSORS,parameter_elements=ELEMENTS,
                rows_by_update=rows_by_update, microbatches=MICRO, timed_steps_per_arm=timed,
                synthetic_operator_training_only=True)
