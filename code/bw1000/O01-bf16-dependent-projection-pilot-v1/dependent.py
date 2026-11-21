"""Staged native V projection and dense BF16 causal probability×V proxy."""

import time

from contract import HEADS, HEAD_DIM, K, LANES, M, N, PHASES, need


def fixture(torch, device, seed):
    generator = torch.Generator().manual_seed(seed)
    host_weight = torch.randint(-2, 3, (K, N), generator=generator, dtype=torch.int8)
    host_inputs = [torch.randint(-2, 3, (M, K), generator=generator, dtype=torch.int8) for _ in range(LANES)]
    cpu_columns = [(x.float() @ host_weight[:, :16].float()).bfloat16() for x in host_inputs]
    host_probability = torch.zeros((HEADS, M, M), dtype=torch.bfloat16)
    for index in range(M):
        host_probability[:, index, index] = 1.0 if index == 0 else 0.75
        if index:
            host_probability[:, index, index-1] = 0.25
    inputs = [x.to(device, dtype=torch.bfloat16) for x in host_inputs]
    weight, probability = host_weight.to(device, dtype=torch.bfloat16), host_probability.to(device)
    slab = torch.empty_like(inputs[0])
    projections = [torch.empty((M, N), dtype=torch.bfloat16, device=device) for _ in inputs]
    consumers = [torch.empty((HEADS, M, HEAD_DIM), dtype=torch.bfloat16, device=device) for _ in inputs]
    values = [x.view(M, HEADS, HEAD_DIM).transpose(0, 1) for x in projections]
    projection_gold = [torch.mm(x, weight) for x in inputs]
    consumer_gold = []
    for gold in projection_gold:
        value = gold.cpu().float().view(M, HEADS, HEAD_DIM).transpose(0, 1)
        expected = value.clone()
        expected[:, 1:] = .75*value[:, 1:] + .25*value[:, :-1]
        consumer_gold.append(expected.bfloat16())
    placement = [slab, weight, probability, *inputs, *projections, *consumers]
    addresses = [x.data_ptr() for x in placement]
    need(all(x.dtype == torch.bfloat16 and x.is_contiguous() for x in placement)
         and len(set(addresses)) == len(addresses)
         and all(value.data_ptr()==projection.data_ptr() and value.stride()==(HEAD_DIM,N,1)
                 for value, projection in zip(values, projections, strict=True)),
         "resident BF16 tensor, alias or value layout differs")

    def direct(lane, marks=None):
        slab.copy_(inputs[lane])
        if marks is not None:
            marks[1].record()
        torch.mm(slab, weight, out=projections[lane])
        if marks is not None:
            marks[2].record()
        torch.bmm(probability, values[lane], out=consumers[lane])
        if marks is not None:
            marks[3].record()

    def pair():
        for lane in range(LANES):
            direct(lane)

    def poison():
        for output in (*projections,*consumers):
            output.fill_(float("nan"))

    def flip():
        for x in inputs:
            x.neg_()

    def verify(negative=False):
        need([x.data_ptr() for x in placement] == addresses, "resident dependent buffer pointer changed")
        for result, projected, cpu_columns16 in zip(projections, projection_gold, cpu_columns, strict=True):
            expected = projected.neg() if negative else projected
            equal = torch.equal(result, expected) if negative else torch.equal(result.view(torch.int16), expected.view(torch.int16))
            need(equal and torch.equal(result[:, :16].cpu(), cpu_columns16.neg() if negative else cpu_columns16),
                 "dependent projection differs from BF16 or CPU reference columns")
        for result, expected in zip(consumers, consumer_gold, strict=True):
            need(torch.equal(result.cpu(), expected.neg() if negative else expected),
                 "complete dependent BF16 consumer differs from the independent CPU reference")

    return direct, pair, poison, flip, verify


def prepare(torch, stream, functions, resources):
    direct, pair, poison, flip, verify = functions
    stage_ms = []
    for lane in range(LANES):
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        marks[0].record()
        direct(lane, marks)
        stream.synchronize()
        stage_ms.append({key: marks[i].elapsed_time(marks[i+1]) for i,key in enumerate(PHASES)})
    verify()
    poison()
    stream.synchronize()
    started = time.perf_counter_ns()
    pair()
    stream.synchronize()
    eager_first_ns = time.perf_counter_ns()-started
    verify()
    started = time.perf_counter_ns()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        pair()
    stream.synchronize()
    capture_ns = time.perf_counter_ns()-started
    resources.callback(graph.reset)
    poison()
    stream.synchronize()
    started = time.perf_counter_ns()
    graph.replay()
    stream.synchronize()
    graph_first_ns = time.perf_counter_ns()-started
    verify()
    flip()
    try:
        poison()
        graph.replay()
        stream.synchronize()
        verify(negative=True)
    finally:
        flip()
    poison()
    pair()
    stream.synchronize()
    verify()
    proof=dict(capture_ns=capture_ns, eager_first_pair_ns=eager_first_ns, graph_first_pair_ns=graph_first_ns,
               diagnostic_stage_ms_by_lane=stage_ms, graph_changed_input_cpu_proof=True)
    return dict(eager=pair, graph=graph.replay), proof
