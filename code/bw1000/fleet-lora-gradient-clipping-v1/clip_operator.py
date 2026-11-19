"""Compare stock and packed LoRA gradient norm clipping after four BF16 microbatches."""
import math
import statistics
import time

import torch


def exercise(device, shape, active, end):
    rows, hidden, width = shape
    generator = torch.Generator(device=device).manual_seed(10649)
    def tensor(shape, dtype, scale):
        return torch.randn(shape, generator=generator, dtype=dtype, device=device).mul_(scale)

    frozen = tensor((hidden, width), torch.bfloat16, hidden ** -0.5)
    inputs = [tensor((rows, hidden), torch.bfloat16, 1.0) for _ in range(4)]
    targets = [tensor((rows, width), torch.bfloat16, 1.0) for _ in range(4)]
    initial = [tensor((hidden, 16), torch.float32, hidden ** -0.5),
               torch.zeros((16, width), device=device, dtype=torch.float32)]
    limit = 1e-6
    flat = torch.empty(sum(p.numel() for p in initial), device=device, dtype=torch.float32)

    def packed_clip(params):
        offset = 0
        for p in params:
            flat[offset:offset+p.numel()].copy_(p.grad.reshape(-1))
            offset += p.numel()
        norm = torch.linalg.vector_norm(flat)
        torch._foreach_mul_([p.grad for p in params], (limit / (norm + 1e-6)).clamp(max=1.0))
        return norm

    def model(packed):
        params = [torch.nn.Parameter(value.clone()) for value in initial]
        optimizer = torch.optim.AdamW(params, lr=3e-4, betas=(.9, .999), weight_decay=.01,
                                      foreach=False, fused=False)
        return params, optimizer, packed

    def update(params, optimizer, packed, step, events=None):
        optimizer.zero_grad(set_to_none=True)
        left, right = params
        for micro, inputs_m in enumerate(inputs):
            base = inputs_m @ frozen
            low_rank = (inputs_m @ left.to(torch.bfloat16)) @ right.to(torch.bfloat16)
            residual = (base + low_rank * 2).float() - targets[(micro + step) % 4]
            loss = residual.square().mean() * .25
            loss.backward()
        if events is not None:
            events[1].record()
        if packed:
            norm = packed_clip(params)
        else:
            norm = torch.nn.utils.clip_grad_norm_(params, limit, foreach=True)
        optimizer.step()
        if events is not None:
            events[2].record()
        return loss.detach(), norm

    arms = [model(False), model(True)]
    differences = []
    clip_coefficients = []
    for step in range(1, 4):
        losses = [update(*arm, step) for arm in arms]
        ref, cand = arms
        numbers = [float(loss.item()) for loss, _ in losses]
        norms = [float(norm.item()) for _, norm in losses]
        if not all(math.isfinite(v) and v > limit for v in norms):
            raise ValueError('CLIPPING_DID_NOT_ENGAGE')
        if abs(norms[0]-norms[1]) > max(norms)*1e-5:
            raise ValueError('CLIP_NORM_MISMATCH')
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError('NONFINITE_LOSS')
        diffs = [abs(numbers[0] - numbers[1])]
        for first, second in zip(ref[0], cand[0]):
            pairs = [(first, second), (first.grad, second.grad)]
            pairs.extend((ref[1].state[first][key], cand[1].state[second][key])
                         for key in ('exp_avg', 'exp_avg_sq'))
            if int(ref[1].state[first]['step'].item()) != step or int(cand[1].state[second]['step'].item()) != step:
                raise ValueError('OPTIMIZER_COUNTER')
            for a, b in pairs:
                diff = float((a.detach() - b.detach()).abs().max().item())
                if not math.isfinite(diff):
                    raise ValueError('NONFINITE_STATE_DELTA')
                diffs.append(diff)
        if max(diffs) > 5e-5:
            raise ValueError('PACKED_CLIP_NUMERIC_MISMATCH')
        differences.append(max(diffs))
        clip_coefficients.append(limit / (norms[0]+1e-6))
    changed = [int((param.detach() != original).count_nonzero().item())
               for param, original in zip(arms[0][0], initial)]
    if not all(changed):
        raise ValueError('UPDATE_DID_NOT_CHANGE_BOTH_TENSORS')

    active()
    # Keep processing updates while later devices finish initialization.
    warm_params, warm_optimizer = model(False)
    step = 0
    while time.time() < end - 156:
        update(warm_params, warm_optimizer, step)
        torch.cuda.synchronize(device)
        step += 1
    records = []
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
    for block, packed in enumerate((False, True, True, False)):
        params, optimizer, chosen = model(packed)
        step = 0
        for _ in range(3):
            update(params, optimizer, chosen, step)
            step += 1
        torch.cuda.synchronize(device)
        started = time.time()
        per_step = []
        optimizer_ms = []
        while time.time() < end - 2:
            events[0].record()
            loss, _ = update(params, optimizer, chosen, step, events)
            events[-1].synchronize()
            step_ms = events[0].elapsed_time(events[2])
            if not math.isfinite(step_ms):
                raise ValueError('NONFINITE_TIMING')
            per_step.append(step_ms)
            optimizer_ms.append(events[1].elapsed_time(events[2]))
            step += 1
            # Four equal absolute-time blocks keep the fleet on useful updates together.
            block_deadline = end - (3 - block) * 39
            if time.time() >= block_deadline:
                break
        if not per_step or not math.isfinite(float(loss.item())):
            raise ValueError('EMPTY_OR_NONFINITE_MEASURED_BLOCK')
        records.append(dict(packed=packed, updates=len(per_step), epoch=started,
                            seconds=time.time()-started, step_ms=statistics.median(per_step),
                            clip_and_optimizer_ms=statistics.median(optimizer_ms)))
    return dict(shape=list(shape), rank=16, microbatches=4, numeric_updates=3,
                numeric_max_abs=max(differences), updated_elements=changed,
                clip_coefficients=clip_coefficients, max_norm=limit, blocks=records)
