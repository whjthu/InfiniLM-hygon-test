"""Pair one shared process against eight isolated processes on one node."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time


SHAPE = (512, 4096, 4096)


def save(path, record):
    with path.open('x', encoding='utf-8') as f:
        json.dump(record, f, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def step_id():
    return os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']


def run_card(output, card, epoch, barrier=None):
    import torch
    from lora_operator import exercise

    if barrier is None:
        torch.set_num_threads(1)
    observed_cards = torch.cuda.device_count()
    if torch.version.hip is None or observed_cards != 8:
        raise RuntimeError('EXPECTED_HIP_AND_EIGHT_ASSIGNED_CARDS')
    device = torch.device('cuda', card)
    started = time.monotonic()
    results = []
    failure = None
    try:
        with torch.cuda.device(device), torch.no_grad():
            torch.cuda.synchronize(device)
            save(output/f'ready-{card}.json', dict(step=step_id(), card=card, pid=os.getpid(),
                     hip=torch.version.hip, observed_cards=observed_cards, epoch=time.time()))
            if barrier is not None:
                barrier.wait()
            results.append(exercise(device, SHAPE))
            torch.cuda.synchronize(device)
    except Exception as e:
        failure = dict(kind=type(e).__name__, capture_error='when stream is capturing' in str(e),
                       stage=str(e).split(':',1)[0] if str(e).startswith('GRAPH_') else None,
                       message_SHA256=hashlib.sha256(str(e).encode()).hexdigest())
    save(output/f'card-{card}.json', dict(step=step_id(), card=card, node=socket.gethostname().split('.')[0],
                epoch=epoch, runtime_seconds=time.monotonic()-started, pid=os.getpid(), results=results,
                failure=failure, peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if failure is None else None))
    return int(failure is not None)


def shared(output, epoch):
    import torch
    torch.set_num_threads(1)
    barrier = threading.Barrier(8,timeout=60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        workers = [pool.submit(run_card,output,card,epoch,barrier) for card in range(8)]
        return int(any([w.result()!=0 for w in workers]))


def supervise(output, epoch):
    hard_end = time.monotonic()+min(166, epoch+240-time.time())
    workers = {}
    stdout = []
    failure = None
    control = None
    control_result = None
    def stop(_sig, _frame):
        raise InterruptedError('OWNED_CARD_SUPERVISOR_INTERRUPTED')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,stop)
    try:
        shared_dir = output/'shared'
        shared_dir.mkdir()
        isolated_dir = output/'isolated'
        isolated_dir.mkdir()
        log = (shared_dir/'process.log').open('xb')
        stdout.append(log)
        control = subprocess.Popen([sys.executable,'-B',__file__,str(shared_dir),str(epoch),'--shared'],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            control_result = control.wait(timeout=min(65,max(0,hard_end-time.monotonic()-80)))
        except subprocess.TimeoutExpired:
            control.kill()
            control_result = control.wait(timeout=3)
        save(shared_dir/'process-result.json',dict(pid=control.pid,returncode=control_result,step=step_id(),epoch=epoch))
        for card in range(8):
            log = (isolated_dir/f'card-{card}.log').open('xb')
            stdout.append(log)
            workers[card] = subprocess.Popen([sys.executable, '-B', __file__, str(isolated_dir), str(epoch), '--card',str(card)],
                                              stdout=log,stderr=subprocess.STDOUT)
        for child in workers.values():
            child.wait(timeout=max(0,hard_end-time.monotonic()-10))
    except BaseException as e:
        failure = type(e).__name__
    finally:
        owned = ([control] if control is not None else [])+list(workers.values())
        for child in owned:
            if child.poll() is None:
                child.terminate()
        for child in owned:
            if child.poll() is None:
                try:
                    child.wait(timeout=max(0, min(2,hard_end-time.monotonic())))
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=1)
        for f in stdout:
            f.close()
    returns = {k:v.poll() for k,v in workers.items()}
    save(output/'node-terminal.json',dict(step=step_id(), epoch=epoch, exit_codes=returns,
                                         worker_PIDs={k:v.pid for k,v in workers.items()}, failure=failure,
                                         shared_process_returncode=control_result))
    return int(failure is not None or len(returns)!=8 or any(v!=0 for v in returns.values()))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('output',type=Path)
    cli.add_argument('epoch',type=float)
    cli.add_argument('--card',type=int,choices=range(8))
    cli.add_argument('--shared',action='store_true')
    args=cli.parse_args()
    if args.shared:
        raise SystemExit(shared(args.output,args.epoch))
    raise SystemExit(supervise(args.output,args.epoch) if args.card is None else
                     run_card(args.output,args.card,args.epoch))
