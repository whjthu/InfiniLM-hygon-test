"""Run matched Qwen-shaped AdamW updates inside independent eight-device DDP groups."""
import ctypes
import json
import re
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import tempfile
from datetime import timedelta

from node_sample import fresh_inventory
from qualified_sampler import collect, node_map

def pci_for_device(index, expected):
    hip=ctypes.CDLL('libamdhip64.so')
    hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    buf=ctypes.create_string_buffer(32)
    if hip.hipDeviceGetPCIBusId(buf,len(buf),index)!=0:
        raise RuntimeError('HIP_PCI')
    pci=buf.value.decode().lower()
    if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):
        pci='0000:'+pci
    if pci not in expected:
        raise RuntimeError('HIP_PCI_OUTSIDE_SLURM_NODE_INVENTORY')
    return pci


def receipt(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('x', encoding='utf-8') as target:
        json.dump(data, target, allow_nan=False)
    os.replace(temp, path)


def card_run(directory, card, identity, epoch, store, pilot):
    import torch
    import torch.distributed as dist
    from bucket_train import exercise

    torch.set_num_threads(1)
    if torch.version.hip is None or torch.cuda.device_count() != 8:
        raise RuntimeError('EXPECTED_HIP_AND_EIGHT_ASSIGNED_DEVICES')
    device = torch.device('cuda', card)
    pci=pci_for_device(card,json.loads((directory/"inventory.json").read_bytes())["PCI_KFD"])
    identity=dict(identity,pci=pci)
    results = []
    error = None
    def measured():
        path = directory / f'active-{card}.json'
        if not path.exists():
            receipt(path, dict(identity, card=card, pid=os.getpid(), epoch=time.time()))

    rank = int(os.environ['SLURM_PROCID'])*8 + card + (320 if identity['step'].startswith('1103606.') else 0)
    try:
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats(device)
            dist.init_process_group('nccl',init_method=store.as_uri(),rank=card,world_size=8,
                                    timeout=timedelta(seconds=42),device_id=device)
            deadline=epoch if pilot else epoch+490
            results.append(exercise(device,card,962137+(rank-card),deadline,pilot,measured))
            torch.cuda.synchronize(device)
            dist.destroy_process_group()
    except Exception as exc:
        error = dict(kind=type(exc).__name__, stage=str(exc).split(':', 1)[0]
                     if str(exc).startswith('GRAPH_') else 'NUMERIC_OR_RUNTIME')
    receipt(directory / f'card-{card}.json',
            dict(identity, card=card, pid=os.getpid(), results=results, error=error,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if error is None else None))
    return int(error is not None)


def node_run(directory, identity, epoch, sample_epoch=None):
    with tempfile.TemporaryDirectory(prefix='qwen-ddp-file-',dir='/tmp') as tmp:
        return supervise(directory,identity,epoch,sample_epoch,Path(tmp)/"store")


def supervise(directory, identity, epoch, sample_epoch, store):
    inventory = fresh_inventory()
    directory.mkdir(mode=0o700)
    receipt(directory/"inventory.json", dict(identity, PCI_KFD=inventory))
    if any(Path('/sys/class/kfd/kfd/proc').iterdir()):
        receipt(directory/'admission-error.json', dict(identity, cause='KFD_PROCESS_PRESENT'))
        return 2
    children = {}
    logs = []
    error = None

    stop_sampler = threading.Event()

    def sample():
        try:
            for offset in (45, 55, 65, 80, 100, 125, 155, 200, 260, 340, 390):
                target = sample_epoch + offset
                if time.time() - target > .25:
                    continue
                if stop_sampler.wait(max(0, target-time.time())):
                    break
                points = collect(Path('/sys/class/drm'), sorted(inventory), target)
                if fresh_inventory() != inventory:
                    raise ValueError('PCI_KFD_CHANGED')
                receipt(directory/f'busy-{offset}.json', dict(identity, rows=points))
        except Exception as exc:
            receipt(directory/'busy-error.json', dict(identity, kind=type(exc).__name__))

    sampler = threading.Thread(target=sample, daemon=True)
    if sample_epoch is not None:
        sampler.start()

    def interrupted(_sig, _frame):
        raise InterruptedError('OWNED_WORKER_INTERRUPTED')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        for card in range(8):
            log = (directory / f'card-{card}.log').open('xb')
            logs.append(log)
            children[card] = subprocess.Popen(
                [sys.executable,'-B',__file__,'--card',str(directory),str(epoch),str(card),str(store),str(int(sample_epoch is None))],
                stdout=log, stderr=subprocess.STDOUT)
        for child in children.values():
            child.wait(timeout=max(0, epoch+(0 if sample_epoch is None else 510)-time.time()))
    except (OSError, InterruptedError, subprocess.TimeoutExpired) as exc:
        error = type(exc).__name__
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
        for log in logs:
            log.close()
    stop_sampler.set()
    if sample_epoch is not None:
        sampler.join(timeout=5)
    exits = {card: child.returncode for card, child in children.items()}
    post_holders = len(list(Path('/sys/class/kfd/kfd/proc').iterdir()))
    if all(rc==0 for rc in exits.values()):
        receipts=[json.loads((directory/f'card-{i}.json').read_bytes()) for i in exits]
        if {row['pci'] for row in receipts}!=set(inventory):
            error='PHYSICAL_CARD_BIJECTION'
    if fresh_inventory() != inventory or post_holders:
        error = 'POST_DEVICE_INVENTORY_OR_HOLDERS'
    receipt(directory/'terminal.json', dict(identity, error=error, exits=exits,
                                             post_kfd_holders=post_holders))
    return int(error is not None or len(exits) != 8 or any(rc != 0 for rc in exits.values()))


def main(root, epoch):
    owners = node_map(json.loads((root.parent/'ledger.json').read_bytes()))
    node = socket.gethostname().split('.')[0]
    parent = os.environ['SLURM_JOB_ID']
    step = os.environ['SLURM_STEP_ID']
    if owners[node][0] != parent or not step.isdecimal():
        raise ValueError('SLURM_PLACEMENT_MISMATCH')
    intent = json.loads((root.parent/'intent.json').read_bytes())
    if node not in intent['parents'][parent]:
        raise ValueError('NODE_OUTSIDE_ADMITTED_WORK')
    directory = root/node
    identity = dict(node=node, step=parent+'.'+step, launch_epoch=epoch)
    return node_run(directory, identity, epoch, intent["sample_epoch"])


if __name__ == '__main__':
    if sys.argv[1]=='--card':
        location=Path(sys.argv[2])
        identity=json.loads((location/'inventory.json').read_bytes())
        identity.pop('PCI_KFD')
        raise SystemExit(card_run(location,int(sys.argv[4]),identity,float(sys.argv[3]),Path(sys.argv[5]),sys.argv[6]=='1'))
    raise SystemExit(main(Path(sys.argv[1]),float(sys.argv[2])))
