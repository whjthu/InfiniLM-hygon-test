"""Fixed 64 resident BF16 GEMMs, eager versus one HIP graph replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time


HERE = Path(__file__).resolve().parent
M, K, N, DEPTH = 184, 4096, 1024, 64
ORDER = ('eager', 'graph', 'graph', 'eager')
PINS = {
    'scope.py': '3a84b9b82cf02764fc811bbad8454e7afc12fd6c09d9e5ed0e4f5387699e4983',
    'setup_gate.py': 'a802daa826f21a31292e81e421a57dc6bb61bd91a5a57426f5d0863239cc239d',
    'contention_worker.py': '4fe6194201848a0b065abdde41beea71400e02aac2b3c64f985b7a9470d305ad',
    'duplex_worker.py': '48310c24b6a0582a65dd8b5fa02d216784a5ae3b9dd57c570aed13b7fe6adb7f',
    'hip_runtime.py': 'e551c1faa38e6472be7861a572354c72311a9b21d4ad2e73d7bb0a3fa403d534',
    'node_sample.py': 'e10a17f576da010cd50937a8f91eac1c8f66098dde6d887c8373d00a615c8931',
    'qualified_sampler.py': '09fc94b7f527c159bd489bacbf7718dad8615ba61cd3f361a44f441da14a0430',
    'schedule.py': 'ead627410a66ae1b278a31fb45d30f1b44b8c366c47eed74ad3fd5855fb62824',
}
for member, pin in PINS.items():
    path = HERE / member
    if path.resolve(strict=True) != path or sha256(path.read_bytes()).hexdigest() != pin:
        raise ValueError('unbound shared process source: '+member)
sys.path.insert(0,str(HERE))
import contention_worker as prior
from schedule import Clock, NANO, bounds, need
from scope import validate_worker_plan
from setup_gate import ARM_LEAD_NS, SETUP_LEAD_NS, PhaseGate


def projection(torch, device, seed):
    """Use the original resident M184 two-input matrix construction."""
    generator = torch.Generator().manual_seed(seed)
    cpu_weight = torch.randint(-2, 3, (K, N), generator=generator, dtype=torch.int8)
    cpu_inputs = [torch.randint(-2, 3, (M, K), generator=generator, dtype=torch.int8) for _ in range(2)]
    gold = [(x.float() @ cpu_weight[:, :16].float()).bfloat16() for x in cpu_inputs]
    weight = cpu_weight.to(device, dtype=torch.bfloat16)
    inputs = [x.to(device, dtype=torch.bfloat16) for x in cpu_inputs]
    outputs = [torch.empty((M, N), device=device, dtype=torch.bfloat16) for _ in inputs]
    references = [torch.mm(x, weight) for x in inputs]
    torch.cuda.synchronize(device)
    owners = (weight, *inputs, *outputs)
    need(all(t.is_contiguous() and t.dtype == torch.bfloat16 for t in owners), 'resident contiguous BF16 required')
    geometry = dict(M=M,K=K,N=N,input_strides=list(inputs[0].stride()),
                    weight_strides=list(weight.stride()),output_strides=list(outputs[0].stride()))
    addresses = tuple(x.data_ptr() for x in owners)

    def project():
        for index in range(DEPTH):
            lane = index % len(inputs)
            torch.mm(inputs[lane], weight, out=outputs[lane])

    def verify(negative=False):
        need(tuple(x.data_ptr() for x in owners) == addresses, 'resident tensor pointer changed')
        for output, expected, cpu in zip(outputs,references,gold,strict=True):
            if negative:
                equal = torch.equal(output,expected.neg())
                first = torch.equal(output[:,:16].cpu(),cpu.neg())
            else:
                equal = torch.equal(output.view(torch.int16),expected.view(torch.int16))
                first = torch.equal(output[:,:16].cpu(),cpu)
            need(equal and first, 'full BF16 result or CPU first16 reference differs')

    def poison():
        for output in outputs:
            output.fill_(float('nan'))

    def flip():
        for value in inputs:
            value.neg_()

    return project,verify,poison,flip,geometry


def capture(torch, stream, project):
    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter_ns()
    with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
        project()
    stream.synchronize()
    return graph,time.perf_counter_ns()-start


def prove_replay(graph, stream, verify, poison, flip):
    flip()
    try:
        poison()
        graph.replay()
        stream.synchronize()
        verify(negative=True)
    finally:
        flip()
    poison()
    graph.replay()
    stream.synchronize()
    verify()


def grid_for(plan):
    old = bounds(plan['campaign_epoch'],plan['shared_start_epoch'],plan['deadline_epoch'])
    return tuple((name,start,end) for name,(_,start,end) in zip(ORDER,old,strict=True))


def main(out, plan):
    process_enter_epoch_ns = time.time_ns()
    from duplex_worker import bind, fresh_inventory, KFD, need as need_pci, original_affinity, pci_at, save, V

    node = socket.gethostname().split('.',1)[0]
    parent, step = os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID']
    selected = validate_worker_plan(plan, node, parent, step)
    grid = grid_for(plan)
    remaining = plan['deadline_epoch']-time.time()
    need(0<remaining<=175 and time.time()<plan['shared_start_epoch'],'fixed plan already started or expired')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,remaining)
    expected = selected['PCI']
    need_pci(len(expected)==len(set(expected))==8 and set(expected)==set(fresh_inventory())
             and not any(KFD.iterdir()),'fresh PCI/KFD gate failed')
    import torch

    torch.set_num_threads(1)
    need(bool(torch.version.hip) and torch.cuda.device_count()==8,'eight visible HIP devices required')
    hip = bind()
    mapping = [pci_at(hip,index) for index in range(8)]
    need(hip.scalar('GetDeviceCount')==8 and set(mapping)==set(expected),'actual HIP ordinal differs')
    affinity = original_affinity(mapping)
    out = out/node
    out.mkdir(mode=0o700)
    clock = Clock()
    stop, collector_stop = threading.Event(), threading.Event()
    ready = PhaseGate(clock, grid[0][1] - SETUP_LEAD_NS, 8, stop.is_set)
    arm_gates = tuple(PhaseGate(clock, start - ARM_LEAD_NS, 8, stop.is_set) for _, start, _ in grid)
    signal.signal(signal.SIGUSR1,lambda *_:stop.set())
    status = dict(request=plan['request'],node=node,step=parent+'.'+step,planned_cards=8,grid=grid,
                  selected_PCI=mapping,torch_version=str(torch.__version__),torch_HIP=torch.version.hip,
                  process_enter_epoch_ns=process_enter_epoch_ns)
    save(out/'intent.json',status)
    sampler = threading.Thread(target=prior.monitor,args=(out/'busy.jsonl',clock,grid,mapping,collector_stop),daemon=True)
    sampler.start()

    def owner(card):
        cpu,local,numa = affinity[card]
        row = dict(card=card,PCI=mapping[card],cpu=cpu,local_cpu=local,PCI_NUMA=numa,planned_arms=4,
                   arms=[],normal_release=False,error=None)
        started = clock.wall()
        marks = row['setup_phase_epoch_ns'] = dict(owner_enter=started)
        where = 'setup'
        try:
            os.sched_setaffinity(0,{cpu})
            device = torch.device('cuda',card)
            with torch.no_grad(),torch.cuda.device(device):
                hip.call('SetDevice',card)
                need(hip.scalar('GetDevice')==card,'HIP current device differs')
                row['provider_SHA256'] = prior.provider_identity()
                row['memory_before'] = hip.mem()
                marks['provider_ready'] = clock.wall()
                seed = int.from_bytes(sha256(mapping[card].encode()).digest()[:8],'little')
                row['seed'] = seed
                project,verify,poison,flip,geometry = projection(torch,device,seed)
                row['geometry'] = geometry
                marks['projection_ready'] = clock.wall()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                current = V(stream.cuda_stream)
                with (hip.stream() as markers,torch.cuda.stream(stream),ExitStack() as resources):
                    before = (current,markers[1],markers[2])
                    # Drain on errors before releasing marker events and static buffers.
                    try:
                        poison()
                        project()
                        stream.synchronize()
                        verify()
                        row['memory_before_capture'] = hip.mem()
                        marks['eager_exact'] = clock.wall()
                        where = 'capture'
                        graph,capture_ns = capture(torch,stream,project)
                        resources.callback(graph.reset)
                        row['capture_ns'] = capture_ns
                        row['memory_after_capture'] = hip.mem()
                        marks['capture_ready'] = clock.wall()
                        where = 'changed_and_restored_input_replay'
                        prove_replay(graph,stream,verify,poison,flip)
                        row['changed_and_restored_input_replay_passed'] = True
                        marks['input_replay_exact'] = clock.wall()
                        row['setup_seconds'] = (marks['input_replay_exact'] - started) / NANO
                        where = 'setup_receipt'
                        save(out/f'setup-card-{card}.json', dict(card=card, PCI=mapping[card], request=status['request'],
                             node=node, step=status['step'], phase_epoch_ns=marks, setup_seconds=row['setup_seconds']))
                        where = 'setup_gate'
                        row['setup_gate'] = asdict(ready.wait(card, where))
                        for index,(mode,start,end) in enumerate(grid):
                            where = f'{mode}_arm_{index}_prepare'
                            run = project if mode == 'eager' else graph.replay
                            # Arm-local poison prevents previous-mode output reuse.
                            poison()
                            stream.synchronize()
                            where = f'{mode}_arm_{index}_gate'
                            release = arm_gates[index].wait(card, where)
                            arm = dict(index=index,mode=mode,scheduled_start_epoch_ns=start,scheduled_end_epoch_ns=end,
                                       ready_epoch_ns=release.arrived_epoch_ns,passed=False,
                                       phase_gate=asdict(release))
                            row['arms'].append(arm)
                            where = f'{mode}_arm_{index}_measure'
                            prior.measure(hip,before,None,None,(None,)*4,run,False,stop,clock,start,end,arm)
                            verify()
                            arm.update(passed=True,full_BF16_equality=True,CPU_first16=True,
                                       gemm_per_s=arm['gemm_calls']*NANO/arm['wall_ns'],
                                       captured_gemms_per_replay=DEPTH if mode=='graph' else 0)
                            save(out/f'card-{card}-arm-{index}.json',arm)
                    finally:
                        stream.synchronize()
                row['memory_after_graph_release'] = hip.mem()
            row['normal_release'] = True
        except Exception as error:
            for gate in (ready, *arm_gates):
                gate.abort(card, where, type(error).__name__)
            row['error'] = dict(type=type(error).__name__,stage=where,reason=str(error)[:130])
        save(out/f'card-{card}.json',row)
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(owner,range(8)))
    collector_stop.set()
    sampler.join(timeout=2)
    save(out/'setup-summary.json', dict(request=status['request'], node=node, step=status['step'],
        process_enter_epoch_ns=process_enter_epoch_ns, grid=grid, setup_gate=ready.summary(),
        arm_gates=[gate.summary() for gate in arm_gates],
        cards=[dict(card=row['card'], marks=row['setup_phase_epoch_ns'], setup_gate=row.get('setup_gate'),
                    error=row['error']) for row in rows]))
    valid = sum(row['normal_release'] and len(row['arms'])==len(ORDER)
                and all(arm['passed'] for arm in row['arms']) for row in rows)
    save(out/'terminal.json',dict(status,valid_cards=valid,expected_arms=8*len(ORDER),
                                  completed_arms=sum(sum(a['passed'] for a in r['arms']) for r in rows),
                                  missing_cards=8-valid,measurement='GEMM completions per full active wall and separate driver samples'))
    signal.setitimer(signal.ITIMER_REAL,0)
    return int(valid!=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('plan',type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.output,json.loads(args.plan.read_bytes())))
