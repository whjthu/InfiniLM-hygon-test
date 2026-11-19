"""Compare eager, output reuse, and graph replay on two measured small GEMMs."""
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time

import torch

from base_worker import write_json


SHAPES = ((1, 4096, 1024), (16, 4096, 1024))
CHAIN = 32
INPUTS = 4


def exercise(device, shape):
    m, k, n = shape
    generator = torch.Generator().manual_seed(m + 583)
    cpu_weight = torch.randint(-2, 3, (k, n), generator=generator, dtype=torch.int8)
    cpu_inputs = [torch.randint(-2, 3, (m, k), generator=generator, dtype=torch.int8) for _ in range(INPUTS)]
    expected = [(x.float() @ cpu_weight[:, :16].float()).bfloat16() for x in cpu_inputs]
    weight = cpu_weight.to(device=device, dtype=torch.bfloat16)
    inputs = [x.to(device=device, dtype=torch.bfloat16) for x in cpu_inputs]
    outputs = [torch.empty((m, n), dtype=torch.bfloat16, device=device) for _ in inputs]

    def verify(results, gold):
        return all(torch.equal(a[:, :16].cpu(), b) for a,b in zip(results,gold))

    def chain(reuse):
        results = list(outputs)
        for i in range(CHAIN):
            j = i % INPUTS
            if reuse:
                torch.mm(inputs[j], weight, out=outputs[j])
            else:
                results[j] = torch.mm(inputs[j], weight)
        return results

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = None
    graph_failure = None
    graph_setup_wall_s = None
    with torch.cuda.stream(stream):
        for reuse in (False, True):
            if not verify(chain(reuse),expected):
                raise ValueError('EAGER_CPU_REFERENCE_MISMATCH')
        stream.synchronize()
        started = time.monotonic()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                chain(True)
            torch.cuda.synchronize(device)
            graph_setup_wall_s = time.monotonic() - started
            # Deliberately change inputs after capture so replay cannot pass on stale outputs.
            for target, source in zip(inputs,cpu_inputs):
                target.copy_(-source)
            graph.replay()
            if not verify(outputs,[-x for x in expected]):
                raise ValueError('GRAPH_CHANGED_INPUT_REFERENCE_MISMATCH')
            for target, source in zip(inputs,cpu_inputs):
                target.copy_(source)
            graph.replay()
            if not verify(outputs,expected):
                raise ValueError('GRAPH_RESTORED_INPUT_REFERENCE_MISMATCH')
        except Exception as exc:
            graph_failure = dict(type=type(exc).__name__,message=str(exc)[:220])
            graph = None
            torch.cuda.synchronize(device)
            for target, source in zip(inputs,cpu_inputs):
                target.copy_(source)
            if not verify(chain(True),expected):
                raise ValueError('EAGER_POST_CAPTURE_FALLBACK_MISMATCH') from exc

        arms = {'eager':lambda:chain(False),'reuse':lambda:chain(True)}
        if graph is not None:
            arms['graph'] = graph.replay
        keys=list(arms)
        result={key:[] for key in keys}
        # Rotate arm order. Measure repeated host submission and completion over fixed windows.
        for round_id in range(6):
            for label in keys[round_id%len(keys):]+keys[:round_id%len(keys)]:
                begin=time.monotonic()
                completed=0
                while time.monotonic()-begin < 5:
                    for _ in range(8):
                        arms[label]()
                    stream.synchronize()
                    completed+=CHAIN*8
                wall=time.monotonic()-begin
                result[label].append(dict(wall_seconds=wall,completed_calls=completed,calls_per_second=completed/wall))
    return dict(shape=shape,graph_setup_wall_s=graph_setup_wall_s,graph_failure=graph_failure,timed=result,
                median_calls_per_second={k:statistics.median(v['calls_per_second'] for v in a) for k,a in result.items()},
                fixed_inputs=INPUTS,distinct_changed_input_reference=True)


def main(root,epoch):
    torch.set_num_threads(1)
    if torch.version.hip is None or torch.cuda.device_count()!=8:
        raise RuntimeError('EXPECTED_HIP_AND_EIGHT_VISIBLE_DCU')
    node=socket.gethostname().split('.')[0]
    step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    barrier=threading.Barrier(8,timeout=90)
    stop=threading.Event()
    busy=[]

    def sample():
        paths={p.resolve().parent.name:p for p in Path('/sys/class/drm').glob('renderD*/device/gpu_busy_percent')}
        while not stop.is_set():
            when=time.time()
            try:
                values={k:int(p.read_text()) for k,p in paths.items()}
                if len(values)==8 and all(0<=v<=100 for v in values.values()):
                    busy.append(dict(epoch=when,busy=values))
                    if len(busy)%5==0:
                        write_json(root/f'live-{node}.json',dict(node=node,step=step,epoch=when,samples=busy[-5:]))
            except OSError:
                pass
            stop.wait(2)

    def run(index):
        device=torch.device('cuda',index)
        with torch.cuda.device(device),torch.no_grad():
            torch.cuda.synchronize(device)
            if barrier.wait()==0:
                write_json(root/f'ready-{node}.json',dict(node=node,step=step,visible=8,epoch=time.time()))
            barrier.wait()
            values=[exercise(device,s) for s in SHAPES]
            torch.cuda.synchronize(device)
            return dict(device=index,shapes=values,peak_allocated=torch.cuda.max_memory_allocated(device))

    monitor=threading.Thread(target=sample,daemon=True)
    monitor.start()
    cards,errors=[],[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures={pool.submit(run,i):i for i in range(8)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    cards.append(future.result())
                except Exception as exc:
                    errors.append(dict(device=futures[future],type=type(exc).__name__,message=str(exc)[:200]))
                    barrier.abort()
    finally:
        stop.set()
        monitor.join(timeout=5)
    write_json(root/f'final-{node}.json',dict(node=node,step=step,launch_epoch=epoch,planned_cards=8,
               cards=cards,error=errors,driver_busy=busy))
    return int(bool(errors) or len(cards)!=8)


if __name__=='__main__':
    raise SystemExit(main(Path(sys.argv[1]),float(sys.argv[2])))
