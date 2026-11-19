"""Compare native fused AdamW grouping over the full Qwen LoRA tensor geometry."""
import json
import math
from pathlib import Path
import statistics
import time

import torch


def exercise(device, card_index, active, end):
    layout = json.loads(Path(__file__).with_name('layout.json').read_bytes())
    shapes = [tuple(row) for row in layout['shapes']]
    elements = sum(math.prod(shape) for shape in shapes)
    if len(shapes) != 504 or elements != 43646976:
        raise ValueError('COMPLETE_QWEN_LORA_LAYOUT')
    generator = torch.Generator(device=device).manual_seed(10649 + card_index)
    initial = [torch.randn(shape, device=device, generator=generator).mul_(.02) for shape in shapes]
    gradients = [torch.randn(shape, device=device, generator=generator).mul_(.0002) for shape in shapes]
    opts = dict(lr=.0001, betas=(.9,.95), eps=1e-8, weight_decay=.1)

    def make_arm(group_size, fused):
        params = [torch.nn.Parameter(t.clone()) for t in initial]
        for param, grad in zip(params, gradients, strict=True):
            param.grad = grad
        groups = [{'params':params[index:index+group_size]} for index in range(0,len(params),group_size)]
        return params, torch.optim.AdamW(groups, fused=fused, foreach=not fused, **opts)

    def delta(first, second):
        maxima = [(a.detach()-b.detach()).abs().amax() for a,b in zip(first,second,strict=True)]
        return float(torch.stack(maxima).amax().item())

    reference = make_arm(504, False)
    sizes=(32,64,128)
    arms={size:make_arm(size,True) for size in sizes}
    numeric = []
    for step in range(1,4):
        for _,opt in (reference,*arms.values()):
            opt.step()
        ref_params,ref_opt = reference
        for label,(params,opt) in arms.items():
            maximum = {'parameter':delta(ref_params,params)}
            for key in ('exp_avg','exp_avg_sq'):
                maximum[key] = delta([ref_opt.state[p][key] for p in ref_params],
                                     [opt.state[p][key] for p in params])
            counters = torch.stack([opt.state[p]['step'] for p in params])
            if not bool((counters==step).all().item()) or any(
                    not math.isfinite(value) or value>5e-6 for value in maximum.values()):
                raise ValueError('FUSED_GROUPING_NUMERICAL_GATE')
            numeric.append(dict(step=step,arm=label,max_abs=maximum))
    changed = int(torch.stack([(p.detach()!=start).any() for p,start in zip(arms[64][0],initial,strict=True)]).sum().item())
    if changed != len(shapes):
        raise ValueError('INCOMPLETE_REAL_OPERATOR_UPDATES')
    del reference, arms

    active()
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
    records = []
    timing_start = time.time()
    if timing_start + 540 > end:
        raise TimeoutError('INSUFFICIENT_ORIGINAL_MEASUREMENT_BUDGET')
    offset=card_index%3
    ordered=sizes[offset:]+sizes[:offset]
    for block, group_size in enumerate(ordered+ordered[::-1]):
        _,optimizer = make_arm(group_size, True)
        for _ in range(5):
            optimizer.step()
        torch.cuda.synchronize(device)
        start = time.time()
        block_end = timing_start + (block+1)*90
        measurements = []
        total = 0
        while time.time() < block_end:
            events[0].record()
            optimizer.step()
            events[1].record()
            events[1].synchronize()
            ms = events[0].elapsed_time(events[1])
            if not math.isfinite(ms) or ms<=0:
                raise ValueError('INVALID_ACTUAL_OPTIMIZER_EVENT')
            if total%4==0:
                measurements.append(ms)
            total += 1
        seconds = time.time() - start
        if total<20 or len(measurements)<5:
            raise ValueError('INCOMPLETE_MATCHED_OPERATOR_WINDOW')
        records.append(dict(group_size=group_size,groups=len(optimizer.param_groups),epoch=start,
                            timed_updates=total,sampled_events=len(measurements),wall_seconds=seconds,
                            updates_per_second=total/seconds,event_p50_ms=statistics.median(measurements),
                            event_p95_ms=sorted(measurements)[math.ceil(.95*len(measurements))-1]))
    return dict(tensors=len(shapes),elements=elements,card_index=card_index,changed_tensors=changed,
                synthetic_values=True,numeric=numeric,blocks=records)
