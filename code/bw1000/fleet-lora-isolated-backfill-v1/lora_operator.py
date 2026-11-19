"""Compare rank-16 BF16 adapter forward and backward with PyTorch autograd."""
import statistics
import time

import torch


def exercise(device, shape, on_verified):
    rows, width, projected = shape
    rank = 16
    generator = torch.Generator(device=device).manual_seed(rows+projected+23)

    def data(*dimensions):
        return torch.randint(-2, 3, dimensions, generator=generator, device=device,
                             dtype=torch.int8).bfloat16().mul_(0.125)

    x, a, b, grad = data(rows,width), data(rank,width), data(projected,rank), data(rows,projected)
    z, dz = torch.empty((rows,rank),device=device,dtype=torch.bfloat16), torch.empty((rows,rank),device=device,dtype=torch.bfloat16)
    y, dx, da, db = (torch.empty_like(q) for q in (grad,x,a,b))
    result = y, dx, da, db

    def reused():
        torch.mm(x,a.T,out=z)
        torch.mm(z,b.T,out=y)
        torch.mm(grad,b,out=dz)
        torch.mm(dz,a,out=dx)
        torch.mm(dz.T,x,out=da)
        torch.mm(grad.T,z,out=db)

    def eager():
        with torch.enable_grad():
            targets = tuple(v.detach().requires_grad_(True) for v in (x,a,b))
            out = (targets[0]@targets[1].T)@targets[2].T
            gradients = torch.autograd.grad(out,targets,grad_outputs=grad)
        return (out.detach(),*gradients)

    def verify(want):
        checks = []
        for actual, expected in zip(result,want):
            relative = (actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-6)
            if not torch.isfinite(actual).all() or float(relative)>0.005:
                raise ValueError('BF16_ADAPTER_REFERENCE_MISMATCH')
            checks.append(float(relative))
        return checks

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph_setup_seconds = None
    with torch.cuda.stream(stream):
        expected = eager()
        reused()
        eager_relative = verify(expected)
        stream.synchronize()
        stage = 'GRAPH_CREATE'
        try:
            start = time.monotonic()
            graph = torch.cuda.CUDAGraph()
            stage = 'GRAPH_CAPTURE'
            with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                for _ in range(8):
                    reused()
            graph_setup_seconds = time.monotonic()-start
            stage = 'GRAPH_CHANGED_INPUT_AUTOGRAD'
            x.neg_()
            changed = eager()
            results = []
            for _ in range(3):
                stage = 'GRAPH_REPLAY'
                graph.replay()
                stage = 'GRAPH_VERIFY_CHANGED_INPUT'
                results.append(verify(changed))
        except Exception as exc:
            # The caller records the first visible stage and exits this card without another synchronisation.
            raise RuntimeError(stage+': '+type(exc).__name__+': '+str(exc)) from exc
        changed_relative = results
        on_verified()
        arms = {'autograd': (eager, 1), 'reuse': (reused, 1), 'graph': (graph.replay, 8)}
        order = list(arms)
        measured = {k:[] for k in order}
        for pair in range(4):
            for label in order[pair%len(order):]+order[:pair%len(order)]:
                operation, quantum = arms[label]
                begin = time.monotonic()
                count = 0
                while time.monotonic()-begin < 6:
                    for _ in range(8//quantum):
                        operation()
                    stream.synchronize()
                    count += 8
                elapsed = time.monotonic()-begin
                measured[label].append(dict(wall_seconds=elapsed,complete_forward_backward_calls=count,
                                            calls_per_second=count/elapsed))
    return dict(shape=shape,rank=rank,comparison_relative_l2=eager_relative,changed_input_relative_l2=changed_relative,
                graph_setup_seconds=graph_setup_seconds,measurement=measured,
                median_calls_per_second={k:statistics.median(r['calls_per_second'] for r in v) for k,v in measured.items()})
