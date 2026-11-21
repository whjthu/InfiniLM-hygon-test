"""Compare resident BF16 projection with and without independent pinned HIP transfers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes as C
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
M, K, N = 184, 4096, 1024
DEPTH, COPY_COUNT = 64, 8
COPY_SIZE = 32 * 1024**2
PINS = {
    'duplex_worker.py': '48310c24b6a0582a65dd8b5fa02d216784a5ae3b9dd57c570aed13b7fe6adb7f',
    'hip_runtime.py': 'e551c1faa38e6472be7861a572354c72311a9b21d4ad2e73d7bb0a3fa403d534',
    'node_sample.py': 'e10a17f576da010cd50937a8f91eac1c8f66098dde6d887c8373d00a615c8931',
    'qualified_sampler.py': '09fc94b7f527c159bd489bacbf7718dad8615ba61cd3f361a44f441da14a0430',
    'schedule.py': 'ead627410a66ae1b278a31fb45d30f1b44b8c366c47eed74ad3fd5855fb62824',
}
for member, pin in PINS.items():
    source = HERE/member
    if source.resolve(strict=True) != source or sha256(source.read_bytes()).hexdigest() != pin:
        raise ValueError('unbound staged dependency: '+member)
sys.path.insert(0,str(HERE))
import duplex_worker as dma
from schedule import Clock, NANO, SAMPLE_OFFSETS, bounds, need


def provider_identity():
    """Identify mapped HIP runtime libraries after real torch and direct HIP setup."""
    maps = Path('/proc/self/maps').read_text().splitlines()
    hits = set()
    for line in maps:
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = fields[5]
            if path.startswith('/') and Path(path).name.startswith(('libgalaxyhip.so','libamdhip64.so')):
                hits.add(Path(path).resolve(strict=True))
    need(hits == {Path(dma.HIP_PATH)}, 'torch and direct HIP do not map the same fixed provider')
    return dma.HIP_SHA256


def prepare_projection(torch, device, seed):
    generator = torch.Generator().manual_seed(seed)
    cpu_weight = torch.randint(-2, 3, (K, N), generator=generator, dtype=torch.int8)
    cpu_inputs = [torch.randint(-2, 3, (M, K), generator=generator, dtype=torch.int8) for _ in range(2)]
    gold = [(x.float() @ cpu_weight[:, :16].float()).bfloat16() for x in cpu_inputs]
    weight = cpu_weight.to(device, dtype=torch.bfloat16)
    inputs = [x.to(device, dtype=torch.bfloat16) for x in cpu_inputs]
    outputs = [torch.empty((M, N), device=device, dtype=torch.bfloat16) for _ in inputs]
    references = [torch.mm(x, weight) for x in inputs]
    torch.cuda.synchronize(device)
    need(weight.is_contiguous() and all(x.is_contiguous() for x in (*inputs,*outputs)),
         'resident BF16 layout is not contiguous')
    dimensions = dict(M=M,K=K,N=N,input_strides=list(inputs[0].stride()),
                      weight_strides=list(weight.stride()),output_strides=list(outputs[0].stride()))
    def submit():
        for i in range(DEPTH):
            j = i % len(inputs)
            torch.mm(inputs[j], weight, out=outputs[j])

    def check():
        need(all(torch.equal(result, expected) and torch.equal(result[:, :16].cpu(), cpu)
                 for result, expected, cpu in zip(outputs,references,gold,strict=True)),
             'full BF16 output or first16 CPU reference mismatch')

    return submit, check, dimensions


def interval(hip, marker, stream):
    hip.call('EventRecord', marker, stream)


def burst(hip, compute, up, down, buffers, project, stressed, clock):
    src,dst,gpu_out,gpu_in = buffers
    def queue_copy(slot, address, owner, direction):
        stream,first,last = slot
        interval(hip,first,stream)
        for _ in range(COPY_COUNT):
            hip.call('MemcpyAsync',address,owner,COPY_SIZE,direction,stream)
        interval(hip,last,stream)

    wall_start = clock.monotonic()
    if stressed:
        queue_copy(up,gpu_out[1],src[1],dma.H2D)
        queue_copy(down,dst[1],gpu_in[1],dma.D2H)
    interval(hip,compute[1],compute[0])
    compute_start = clock.monotonic()
    project()
    interval(hip,compute[2],compute[0])
    hip.call('StreamSynchronize',compute[0])
    compute_end = clock.monotonic()
    if stressed:
        hip.call('StreamSynchronize',up[0])
        hip.call('StreamSynchronize',down[0])
    wall_end = clock.monotonic()
    return dict(start_ns=wall_start,end_ns=wall_end,compute_host_ns=compute_end-compute_start,
                compute_stream_ms=dma.elapsed_ms(hip,compute),
                h2d_stream_ms=dma.elapsed_ms(hip,up) if stressed else 0.,
                d2h_stream_ms=dma.elapsed_ms(hip,down) if stressed else 0.)


def measure(hip, compute, up, down, buffers, project, stressed, halt, clock, expected_start, expected_end, row):
    start = clock.toward(expected_start, halt.is_set)
    row['actual_start_epoch_ns'] = clock.wall_from_mono(start)
    row['wait_until_start_ns'] = row['actual_start_epoch_ns'] - row['ready_epoch_ns']
    cutoff = expected_end-clock.anchor
    totals = dict(compute_host_ns=0,compute_stream_ms=0.,h2d_stream_ms=0.,d2h_stream_ms=0.)
    count = 0
    end = start
    while clock.monotonic() < cutoff:
        if halt.is_set():
            raise InterruptedError('cooperative stop during active arm')
        stats = burst(hip,compute,up,down,buffers,project,stressed,clock)
        end = stats['end_ns']
        for key in totals:
            totals[key] += stats[key]
        count += 1
    finished = clock.monotonic()
    clock.verify()
    need(count > 0 and cutoff <= finished <= cutoff + 500_000_000 and cutoff-150_000_000 <= end,
         'scheduled active window incomplete or overran')
    row.update(totals, groups=count, gemm_calls=count*DEPTH, useful_rows=count*DEPTH*M,
               wall_ns=finished-start, actual_end_epoch_ns=clock.wall_from_mono(end),
               collection_end_epoch_ns=clock.wall_from_mono(finished),
               h2d_payload_bytes=count*COPY_COUNT*COPY_SIZE if stressed else 0,
               d2h_payload_bytes=count*COPY_COUNT*COPY_SIZE if stressed else 0)


def monitor(path, clock, grid, mapping, stopped):
    devices = {p.resolve().parent.name:p for p in Path('/sys/class/drm').glob('renderD*/device/gpu_busy_percent')}
    with path.open('x',encoding='utf8') as stream:
        for index,(mode,target,_) in enumerate(grid):
            for offset in SAMPLE_OFFSETS:
                instant = target+offset*NANO
                row = dict(arm=index,mode=mode,planned_epoch_ns=instant,busy_percent=None)
                try:
                    clock.toward(instant,stopped.is_set,150_000_000)
                    begin = clock.wall()
                    values = {pci:int(devices[pci].read_text()) for pci in mapping}
                    end = clock.wall()
                    need(end-begin<=150_000_000 and all(0<=value<=100 for value in values.values()),
                         'busy sample skew or driver value invalid')
                    row.update(observed_begin_epoch_ns=begin,observed_end_epoch_ns=end,busy_percent=values)
                except (OSError,KeyError,ValueError,InterruptedError) as error:
                    row['error'] = str(error)[:100]
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())


def worker(out, plan):
    grid = bounds(plan['campaign_epoch'],plan['shared_start_epoch'],plan['deadline_epoch'])
    node = socket.gethostname().split('.',1)[0]
    parent, step = os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID']
    selected = plan['selected'][node]
    need(parent == selected['parent'] and parent in ('1087656','1103606') and step.isdecimal(),
         'worker differs from the original parent')
    remaining = plan['deadline_epoch']-time.time()
    need(0<remaining<=175 and time.time()<plan['shared_start_epoch'],'fixed plan already exhausted or started')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,remaining)
    expected = selected['PCI']
    need(len(set(expected))==8 and set(expected)==set(dma.fresh_inventory())
         and not any(dma.KFD.iterdir()), 'fresh PCI/KFD gate failed')
    import torch

    torch.set_num_threads(1)
    need(bool(torch.version.hip) and torch.cuda.device_count()==8,'eight visible HIP devices required')
    hip = dma.bind()
    mapping = [dma.pci_at(hip,index) for index in range(8)]
    need(hip.scalar('GetDeviceCount')==8 and set(mapping)==set(expected),'actual HIP ordinal differs')
    affinity = dma.original_affinity(mapping)
    root = out/node
    root.mkdir(mode=0o700)
    clock = Clock()
    stop, monitor_stop, all_ready = threading.Event(),threading.Event(),threading.Barrier(8)
    signal.signal(signal.SIGUSR1,lambda *_:stop.set())
    dma.save(root/'intent.json',dict(request=plan['request'],node=node,step=parent+'.'+step,grid=grid,
                                    planned_cards=8,HIP_SHA256=dma.HIP_SHA256,torch_HIP=torch.version.hip,
                                    torch_version=torch.__version__,projection=dict(M=M,K=K,N=N,depth=DEPTH),
                                    PCI=mapping))
    sampler = threading.Thread(target=monitor,args=(root/'busy.jsonl',clock,grid,mapping,monitor_stop),daemon=True)
    sampler.start()

    def owner(card):
        cpu,local,numa = affinity[card]
        device = torch.device('cuda',card)
        row = dict(card=card,PCI=mapping[card],cpu=cpu,local_cpu=local,PCI_NUMA=numa,arms=[],error=None,
                   node=node,step=parent+'.'+step,planned_arms=len(grid),normal_release=False)
        start_setup = clock.wall()
        try:
            os.sched_setaffinity(0,{cpu})
            with torch.no_grad(),torch.cuda.device(device),ExitStack() as owners:
                hip.call('SetDevice',card)
                need(hip.scalar('GetDevice')==card,'HIP owner mismatch')
                row['memory_before'] = hip.mem()
                seed = int.from_bytes(sha256(mapping[card].encode()).digest()[:8],'little')
                row['input_seed'] = seed
                project, check, geometry = prepare_projection(torch,device,seed)
                row['geometry'] = geometry
                row['runtime_SHA256'] = provider_identity()
                source,destination = (dma.new_buffer(hip,owners,True,COPY_SIZE) for _ in range(2))
                gpu_out,gpu_in = (dma.new_buffer(hip,owners,False,COPY_SIZE) for _ in range(2))
                buffers = (source,destination,gpu_out,gpu_in)
                pins = dma.prime(hip,*buffers,COPY_SIZE,seed)
                row['dma_input_SHA256'] = pins
                row['memory_after_setup'] = hip.mem()
                torch_stream = torch.cuda.Stream(device=device)
                current = dma.V(torch_stream.cuda_stream)
                with (hip.stream() as up,hip.stream() as down,hip.stream() as markers,
                      torch.cuda.stream(torch_stream),ExitStack() as drain):
                    drain.callback(hip.call,'StreamSynchronize',current)
                    compute = (current,markers[1],markers[2])
                    for stressed in (False,True):
                        burst(hip,compute,up,down,buffers,project,stressed,clock)
                    check()
                    dma.audit(hip,*buffers,COPY_SIZE,pins)
                    row['setup_seconds'] = (clock.wall()-start_setup)/NANO
                    all_ready.wait(timeout=max(.01,min(25,plan['shared_start_epoch']-time.time())))
                    for index,(mode,scheduled_start,scheduled_end) in enumerate(grid):
                        stressed = mode == 'compute_dma'
                        if stressed:
                            C.memset(destination[1],0,COPY_SIZE)
                            hip.call('MemsetAsync',gpu_out[1],0,COPY_SIZE,up[0])
                            hip.call('StreamSynchronize',up[0])
                        arm = dict(index=index,mode=mode,scheduled_start_epoch_ns=scheduled_start,
                                   scheduled_end_epoch_ns=scheduled_end,ready_epoch_ns=clock.wall(),
                                   passed=False)
                        row['arms'].append(arm)
                        measure(hip,compute,up,down,buffers,project,stressed,stop,clock,scheduled_start,scheduled_end,arm)
                        check()
                        if stressed:
                            dma.audit(hip,*buffers,COPY_SIZE,pins)
                        arm['passed'] = True
                        arm['dma_payload_and_guards_passed'] = stressed
                        dma.save(root/f'card-{card}-arm-{index}.json',arm)
            row['normal_release'] = True
        except Exception as error:
            all_ready.abort()
            row['error'] = dict(type=type(error).__name__,reason=str(error)[:140])
        dma.save(root/f'card-{card}.json',row)
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        recorded = list(pool.map(owner,range(8)))
    monitor_stop.set()
    sampler.join(timeout=2)
    valid = sum(r['normal_release'] and len(r['arms'])==len(grid) and all(a['passed'] for a in r['arms'])
                for r in recorded)
    dma.save(root/'terminal.json',dict(request=plan['request'],node=node,step=parent+'.'+step,planned_cards=8,
                                     valid_cards=valid,units='resident BF16 projection + asynchronous copy',
                                     sample_only_busy_has_separate_evidence=True,clock_inter_node_bounded=False))
    signal.setitimer(signal.ITIMER_REAL,0)
    return int(valid!=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('plan',type=Path)
    args = parser.parse_args()
    raise SystemExit(worker(args.output,json.loads(args.plan.read_bytes())))
