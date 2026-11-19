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


from lora_operator import exercise

SHAPES = ((128,4096,1024),(128,4096,4096),(128,4096,12288),(512,4096,1024),
          (512,4096,4096),(512,4096,12288),(4096,4096,1024),(4096,4096,4096),(4096,4096,12288))

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
            rank=int(os.environ['SLURM_PROCID'])*8+index
            pair=[SHAPES[(rank+j*4)%len(SHAPES)] for j in range(2)]
            values=[exercise(device,s) for s in pair]
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
