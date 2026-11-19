"""Measure the cost of fusing the trainable-gradient reductions in one node."""
import math
import statistics
import time

import torch
import torch.distributed as dist

ORDER = (False, True, True, False)
WARMUP_DEADLINE = 117
BLOCK_ENDS = (160, 203, 246, 289)
WORK_END = 297
CHUNK = 8
TOLERANCE = 5e-5


def reduce_grads(grads, fused, flat, world):
    if fused:
        offset = 0
        for grad in grads:
            flat[offset:offset + grad.numel()].copy_(grad.view(-1))
            offset += grad.numel()
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for grad in grads:
            grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
            offset += grad.numel()
    else:
        for grad in grads:
            dist.all_reduce(grad)
            grad.div_(world)


def exercise(device, shape, active, epoch, rank, world, numeric_only=False):
    rows, hidden, width = shape

    def sample(generator, shape, dtype, scale):
        return torch.randn(shape, generator=generator, dtype=dtype, device=device).mul_(scale)

    common = torch.Generator(device=device).manual_seed(10649)
    local = torch.Generator(device=device).manual_seed(10650 + rank)
    frozen = sample(common, (hidden, width), torch.bfloat16, hidden ** -0.5)
    inputs = [sample(local, (rows, hidden), torch.bfloat16, 1.0) for _ in range(4)]
    targets = [sample(local, (rows, width), torch.bfloat16, 1.0) for _ in range(4)]
    initial = [sample(common, (hidden, 16), torch.float32, hidden ** -0.5),
               torch.zeros((16, width), device=device, dtype=torch.float32)]
    flat = torch.empty(sum(p.numel() for p in initial), device=device, dtype=torch.float32)
    decision = torch.zeros((), device=device, dtype=torch.int32)

    def model():
        params = [torch.nn.Parameter(value.clone()) for value in initial]
        optimizer = torch.optim.AdamW(params, lr=3e-4, betas=(.9, .999), weight_decay=.01,
                                      foreach=False, fused=False)
        return params, optimizer

    def update(params, optimizer, step, fused, events=None, validate=False):
        optimizer.zero_grad(set_to_none=True)
        left, right = params
        for micro, inputs_m in enumerate(inputs):
            base = inputs_m @ frozen
            low_rank = (inputs_m @ left.to(torch.bfloat16)) @ right.to(torch.bfloat16)
            residual = (base + low_rank * 2).float() - targets[(micro + step) % 4]
            loss = residual.square().mean() * .25
            loss.backward()
        grads = [p.grad for p in params]
        if validate:
            reference = []
            for grad in grads:
                original = grad.detach().clone()
                replicas = [torch.empty_like(grad) for _ in range(world)]
                dist.all_gather(replicas, original)
                reference.append(torch.stack([replica.cpu().double() for replica in replicas]).mean(0))
        if events:
            events[1].record()
        reduce_grads(grads, fused, flat, world)
        if events:
            events[2].record()
        optimizer.step()
        if events:
            events[3].record()
        error = 0.0
        if validate:
            for grad, expected in zip(grads, reference):
                diff = float((grad.detach().cpu().double() - expected).abs().max().item())
                if not math.isfinite(diff):
                    raise ValueError('NONFINITE_GRADIENT_DIFFERENCE')
                error = max(error, diff)
        return loss.detach(), error

    scalar, fused = model(), model()
    maxima = []
    for step in range(1, 4):
        a, ae = update(*scalar, step, False, validate=True)
        b, be = update(*fused, step, True, validate=True)
        numbers = [float(a.item()), float(b.item())]
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('NONFINITE_LOSS')
        diffs = [abs(numbers[0] - numbers[1]), ae, be]
        for left, right in zip(scalar[0], fused[0]):
            pairs = [(left, right), (left.grad, right.grad)]
            ls, rs = scalar[1].state[left], fused[1].state[right]
            if int(ls['step'].item()) != step or int(rs['step'].item()) != step:
                raise ValueError('OPTIMIZER_COUNTER')
            pairs += [(ls[key], rs[key]) for key in ('exp_avg', 'exp_avg_sq')]
            for x, y in pairs:
                diff = float((x.detach() - y.detach()).abs().max().item())
                if not math.isfinite(diff):
                    raise ValueError('NONFINITE_STATE_DELTA')
                diffs.append(diff)
        maxima.append(max(diffs))
    for params, optimizer in (scalar, fused):
        for param in params:
            for value in (param, optimizer.state[param]['exp_avg'], optimizer.state[param]['exp_avg_sq']):
                replica = value.detach().clone()
                dist.broadcast(replica, src=0)
                if not torch.equal(replica, value.detach()):
                    raise ValueError('REPLICA_STATE_MISMATCH')
    if max(maxima) > TOLERANCE:
        raise ValueError('COLLECTIVE_NUMERIC_MISMATCH')
    changed = [int((param.detach() != original).count_nonzero().item())
               for param, original in zip(scalar[0], initial)]
    if not all(changed):
        raise ValueError('NO_UPDATE')
    result = dict(shape=list(shape), rank=16, world=world, microbatches=4, numeric_updates=3,
                  numeric_max_abs=max(maxima), updated_elements=changed, flat_gradient_bytes=flat.numel()*4,
                  exact_replicas_after_numeric=True)
    if numeric_only:
        return result

    def more(until):
        if rank == 0:
            decision.fill_(int(time.time() < until))
        dist.broadcast(decision, src=0)
        return bool(decision.item())

    active()
    warm = model()
    step = 0
    while more(epoch + WARMUP_DEADLINE):
        for _ in range(CHUNK):
            update(*warm, step, False)
            step += 1
        torch.cuda.synchronize(device)
    records = []
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
    for use_fused, stop in zip(ORDER, BLOCK_ENDS):
        params, optimizer = model()
        for warmup in range(3):
            update(params, optimizer, warmup, use_fused)
        torch.cuda.synchronize(device)
        step_ms, comm_ms = [], []
        started = time.time()
        step = 3
        while more(epoch + stop):
            for _ in range(CHUNK):
                events[0].record()
                loss, _ = update(params, optimizer, step, use_fused, events)
                events[3].synchronize()
                step_ms.append(events[0].elapsed_time(events[3]))
                comm_ms.append(events[1].elapsed_time(events[2]))
                step += 1
        if not step_ms or not math.isfinite(float(loss.item())):
            raise ValueError('EMPTY_OR_NONFINITE_TIMED_BLOCK')
        if not all(math.isfinite(x) and x >= 0 for x in [*step_ms, *comm_ms]):
            raise ValueError('INVALID_EVENT_TIMING')
        records.append(dict(fused=use_fused, calls_per_update=1 if use_fused else 2, updates=len(step_ms),
                            epoch=started, finish=time.time(), step_ms=statistics.median(step_ms),
                            reduction_with_pack_ms=statistics.median(comm_ms)))
    result['blocks'] = records
    return result
