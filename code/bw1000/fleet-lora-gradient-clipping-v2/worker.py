"""Run LoRA accumulation and AdamW with one independent HIP process per local device."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from node_sample import fresh_inventory
from qualified_sampler import collect, node_map

SHAPES = tuple((rows, 4096, width) for rows in (512, 1024)
               for width in (1024, 4096, 12288))


def receipt(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('x', encoding='utf-8') as target:
        json.dump(data, target, allow_nan=False)
    os.replace(temp, path)


def card_run(directory, card, identity):
    import torch
    from clip_operator import exercise

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

    rank = int(os.environ['SLURM_PROCID'])*8 + card
    try:
        with torch.cuda.device(device):
            results.append(exercise(device, SHAPES[rank % len(SHAPES)], measured, identity["launch_epoch"]+292))
            torch.cuda.synchronize(device)
    except Exception as exc:
        error = dict(kind=type(exc).__name__, stage=str(exc).split(':', 1)[0]
                     if str(exc).startswith('GRAPH_') else 'NUMERIC_OR_RUNTIME')
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
    children = {}
    logs = []
    error = None

    def sample():
        try:
            rows = collect(Path('/sys/class/drm'), sorted(inventory), epoch+120)
            if fresh_inventory() != inventory:
                raise ValueError("PCI_KFD_CHANGED")
            receipt(directory/'busy.json', dict(identity, rows=rows))
            for offset in (151, 190, 229, 269):
                points = collect(Path('/sys/class/drm'), sorted(inventory), epoch+offset)
                if fresh_inventory() != inventory:
                    raise ValueError('PCI_KFD_CHANGED')
                receipt(directory/f'busy-{offset}.json', dict(identity, rows=points))
        except Exception as exc:
            receipt(directory/'busy-error.json', dict(identity, kind=type(exc).__name__))

    sampler = threading.Thread(target=sample, daemon=True)
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
                [sys.executable, '-B', __file__, str(directory.parent), str(epoch), str(card)],
                stdout=log, stderr=subprocess.STDOUT)
        for child in children.values():
            child.wait(timeout=max(0, epoch+385-time.time()))
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
    sampler.join(timeout=max(0, min(125, epoch+388-time.time())))
    exits = {card: child.returncode for card, child in children.items()}
    post_holders = len(list(Path('/sys/class/kfd/kfd/proc').iterdir()))
    if fresh_inventory() != inventory or post_holders:
        error = 'POST_DEVICE_INVENTORY_OR_HOLDERS'
    receipt(directory/'terminal.json', dict(identity, error=error, exits=exits,
                                             post_kfd_holders=post_holders))
    return int(error is not None or len(exits) != 8 or any(rc != 0 for rc in exits.values()))


def main(root, epoch, card):
    owners = node_map(json.loads((root.parent/'ledger.json').read_bytes()))
    node = socket.gethostname().split('.')[0]
    parent = os.environ['SLURM_JOB_ID']
    step = os.environ['SLURM_STEP_ID']
    if owners[node][0] != parent or not step.isdecimal():
        raise ValueError('SLURM_PLACEMENT_MISMATCH')
    directory = root/node
    identity = dict(node=node, step=parent+'.'+step, launch_epoch=epoch)
    if card is not None:
        return card_run(directory, card, identity)
    return node_run(directory, identity, epoch)


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1]), float(sys.argv[2]),
                          int(sys.argv[3]) if len(sys.argv) == 4 else None))
