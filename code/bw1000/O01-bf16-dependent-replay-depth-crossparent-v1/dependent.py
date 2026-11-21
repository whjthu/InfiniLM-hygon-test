"""Staged native V projection and dense BF16 causal probability×V proxy."""

import time

from contract import DEPTH, HEADS, HEAD_DIM, K, LANES, M, N, PHASES, need, polarity_order


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
    inputs = [inputs, [x.neg() for x in inputs]]
    slab = torch.empty_like(inputs[0][0])
    projections = [[torch.empty((M,N), dtype=torch.bfloat16, device=device) for _ in bank] for bank in inputs]
    consumers = [[torch.empty((HEADS,M,HEAD_DIM), dtype=torch.bfloat16, device=device) for _ in bank] for bank in inputs]
    flat_projections = projections[0]+projections[1]
    flat_consumers = consumers[0]+consumers[1]
    values = [[x.view(M,HEADS,HEAD_DIM).transpose(0,1) for x in bank] for bank in projections]
    projection_gold = [torch.mm(x, weight) for x in inputs[0]]
    consumer_gold = []
    for gold in projection_gold:
        value = gold.cpu().float().view(M, HEADS, HEAD_DIM).transpose(0, 1)
        expected = value.clone()
        expected[:, 1:] = .75*value[:, 1:] + .25*value[:, :-1]
        consumer_gold.append(expected.bfloat16())
    placement = [slab, weight, probability, *inputs[0], *inputs[1], *flat_projections, *flat_consumers]
    addresses = [x.data_ptr() for x in placement]
    need(all(x.dtype == torch.bfloat16 and x.is_contiguous() for x in placement)
         and len(set(addresses)) == len(addresses)
         and all(value.data_ptr()==projection.data_ptr() and value.stride()==(HEAD_DIM,N,1)
                 for value, projection in zip(values[0]+values[1], flat_projections, strict=True)),
         "resident BF16 tensor, alias or value layout differs")

    def direct(polarity, lane, marks=None):
        slab.copy_(inputs[polarity][lane])
        if marks is not None:
            marks[1].record()
        torch.mm(slab, weight, out=projections[polarity][lane])
        if marks is not None:
            marks[2].record()
        torch.bmm(probability, values[polarity][lane], out=consumers[polarity][lane])
        if marks is not None:
            marks[3].record()

    def pair(polarity):
        for lane in range(LANES):
            direct(polarity,lane)

    def poison():
        for output in (*flat_projections,*flat_consumers):
            output.fill_(float("nan"))

    def flip():
        for source in inputs[0]+inputs[1]:
            source.neg_()

    def verify(polarities, flipped=False):
        need([x.data_ptr() for x in placement] == addresses, "resident dependent buffer pointer changed")
        for polarity in polarities:
            negative = bool(polarity)^flipped
            for result, projected, cpu_columns16 in zip(projections[polarity], projection_gold, cpu_columns, strict=True):
                expected = projected.neg() if negative else projected
                equal = torch.equal(result,expected) if negative else torch.equal(result.view(torch.int16),expected.view(torch.int16))
                need(equal and torch.equal(result[:,:16].cpu(), cpu_columns16.neg() if negative else cpu_columns16),
                     "dependent projection differs from BF16 or CPU reference columns")
            for result, expected in zip(consumers[polarity], consumer_gold, strict=True):
                need(torch.equal(result.cpu(),expected.neg() if negative else expected),
                     "complete dependent BF16 consumer differs from the independent CPU reference")

    return direct, pair, poison, flip, verify


def prepare(torch, stream, functions, resources):
    direct, pair, poison, flip, verify = functions
    stage_ms = []
    for lane in range(LANES):
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        marks[0].record()
        direct(0,lane,marks)
        stream.synchronize()
        stage_ms.append({key: marks[i].elapsed_time(marks[i+1]) for i,key in enumerate(PHASES)})
    pair(1)
    stream.synchronize()
    verify((0,1))
    variants, cost = {}, {}
    for mode in DEPTH:
        graphs, capture_ns, first_replay_ns = [], [], []
        for opening in (0,1):
            started = time.perf_counter_ns()
            graph = torch.cuda.CUDAGraph()
            resources.callback(graph.reset)
            with torch.cuda.graph(graph,stream=stream,capture_error_mode="thread_local"):
                for polarity in polarity_order(mode,opening):
                    pair(polarity)
            stream.synchronize()
            capture_ns.append(time.perf_counter_ns()-started)
            poison()
            stream.synchronize()
            started = time.perf_counter_ns()
            graph.replay()
            stream.synchronize()
            first_replay_ns.append(time.perf_counter_ns()-started)
            verify(sorted(set(polarity_order(mode,opening))))
            graphs.append(graph.replay)
        variants[mode] = tuple(graphs)
        cost[mode] = dict(capture_ns=capture_ns, first_replay_ns=first_replay_ns)
    flip()
    try:
        for mode, graphs in variants.items():
            poison()
            graphs[0]()
            stream.synchronize()
            verify(sorted(set(polarity_order(mode,0))), flipped=True)
            graphs[1]()
            stream.synchronize()
            verify((0,1), flipped=True)
    finally:
        flip()
    return variants, dict(by_depth=cost, diagnostic_stage_ms_by_lane=stage_ms,
                          graph_changed_input_across_replays_cpu_proof=True)
