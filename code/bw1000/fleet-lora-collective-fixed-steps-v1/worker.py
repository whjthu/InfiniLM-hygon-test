"""Run and supervise one eight-device RCCL group on this node."""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from node_sample import fresh_inventory
from qualified_sampler import node_map

SHAPE = (512, 4096, 4096)


def receipt(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('x', encoding='utf-8') as target:
        json.dump(data, target, allow_nan=False)
    os.replace(temp, path)


def card_run(directory, card, identity, rendezvous):
    import torch
    from collectives import exercise
    import torch.distributed as dist

    torch.set_num_threads(1)
    if torch.version.hip is None or torch.cuda.device_count() != 8:
        raise RuntimeError('EXPECTED_HIP_AND_EIGHT_ASSIGNED_DEVICES')
    device = torch.device('cuda', card)
    results = []
    error = None
    def measured():
        path = directory / f'active-{card}.json'
        if not path.exists():
            receipt(path, dict(identity, card=card, pid=os.getpid(), epoch=time.time()))

    started = time.monotonic()
    stage = "rendezvous"
    try:
        with torch.cuda.device(device):
            if not rendezvous.is_absolute() or rendezvous.parent.parent != Path('/tmp'):
                raise ValueError('NODE_LOCAL_FILESTORE_PATH')
            stage = 'communicator_init'
            dist.init_process_group('nccl', init_method=rendezvous.as_uri(), rank=card, world_size=8,
                                    timeout=timedelta(seconds=45), device_id=device)
            initialized = time.monotonic()
            stage = 'numerical_or_measurement'
            result = exercise(device, SHAPE, measured, card, 8)
            stage = 'normal_destroy'
            torch.cuda.synchronize(device)
            dist.destroy_process_group()
            result['rendezvous_and_init_seconds'] = initialized - started
            results.append(result)
    except Exception as exc:
        detail = str(exc)
        markers = ('Watchdog', 'timeout', 'timed out', 'ncclSystemError', 'unhandled system error',
                   'connection refused', 'resource exhausted', 'mismatch')
        error = dict(kind=type(exc).__name__, stage=stage,
                     classes=[x for x in markers if x.lower() in detail.lower()],
                     detail_SHA256=hashlib.sha256(detail.encode()).hexdigest())
    receipt(directory / f'card-{card}.json',
            dict(identity, card=card, pid=os.getpid(), results=results, error=error,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if error is None else None))
    return int(error is not None)


def node_run(directory, identity, epoch):
    inventory = fresh_inventory()
    directory.mkdir(mode=0o700)
    receipt(directory/"inventory.json", dict(identity, PCI_KFD=inventory))
    if any(Path('/sys/class/kfd/kfd/proc').iterdir()):
        receipt(directory/'admission-error.json', dict(identity, cause='KFD_PROCESS_PRESENT'))
        return 2

    def interrupted(_sig, _frame):
        raise InterruptedError('OWNED_WORKER_INTERRUPTED')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix="lora-collective-",dir="/tmp") as local:
        return supervise(directory, identity, epoch, Path(local)/"store", inventory)


def supervise(directory, identity, epoch, rendezvous, inventory):
    children = {}
    logs = []
    error = None
    try:
        for card in range(8):
            log = (directory / f'card-{card}.log').open('xb')
            logs.append(log)
            children[card] = subprocess.Popen(
                [sys.executable, '-B', __file__, str(directory.parent), str(epoch), str(card), str(rendezvous)],
                stdout=log, stderr=subprocess.STDOUT)
        for child in children.values():
            child.wait(timeout=max(0, epoch+325-time.time()))
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
    exits = {card: child.returncode for card, child in children.items()}
    post_holders = len(list(Path('/sys/class/kfd/kfd/proc').iterdir()))
    if fresh_inventory() != inventory or post_holders:
        error = 'POST_DEVICE_INVENTORY_OR_HOLDERS'
    receipt(directory/'terminal.json', dict(identity, error=error, exits=exits,
                                             post_kfd_holders=post_holders))
    return int(error is not None or len(exits) != 8 or any(rc != 0 for rc in exits.values()))


def main(root, epoch, card, rendezvous):
    owners = node_map(json.loads((root.parent/'ledger.json').read_bytes()))
    node = socket.gethostname().split('.')[0]
    parent = os.environ['SLURM_JOB_ID']
    step = os.environ['SLURM_STEP_ID']
    if owners[node][0] != parent or not step.isdecimal():
        raise ValueError('SLURM_PLACEMENT_MISMATCH')
    claim = json.loads((root.parent/'CLAIM.json').read_bytes())
    if node not in claim['parents'][parent]:
        raise ValueError('UNEXPECTED_NODE')
    directory = root/node
    identity = dict(node=node, step=parent+'.'+step, launch_epoch=epoch)
    if card is not None:
        return card_run(directory, card, identity, rendezvous)
    return node_run(directory, identity, epoch)


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1]), float(sys.argv[2]),
                          int(sys.argv[3]) if len(sys.argv) == 5 else None,
                          Path(sys.argv[4]) if len(sys.argv) == 5 else None))
