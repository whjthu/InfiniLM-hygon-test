"""Exercise the exact operator on every DCU in one node before the fleet is eligible."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from node_sample import fresh_inventory
from worker import pci_for_device

KFD = Path('/sys/class/kfd/kfd/proc')


def worker(deadline, card):
    import torch

    from text_linear_operator import exercise

    torch.set_num_threads(1)
    if not torch.version.hip or torch.cuda.device_count() != 8:
        raise ValueError('EIGHT_ASSIGNED_HIP_DEVICES')
    device = torch.device('cuda', card)
    pci = pci_for_device(card, fresh_inventory())
    with torch.cuda.device(device):
        result = exercise(device, 800 + card, lambda: None, deadline, calls_per_block=8, cases=2)
    if len(result["cases"]) != 2:
        raise ValueError("PILOT_CASE_COUNT")
    for row in result["cases"]:
        if any(
            arm["max_abs"] is None or arm["max_abs"] > 0.0625 or arm["padding_nonzero"] or arm["invalid_events"] or arm["changed"]
            for arm in row["arms"].values()
        ):
            raise ValueError("PILOT_FINITE_TIMED_COMPARISON")
    print(json.dumps(dict(pci=pci, result=result), allow_nan=False), flush=True)


def main(path, deadline):
    pins = fresh_inventory()
    if any(KFD.iterdir()):
        raise ValueError('PILOT_PRE_KFD')
    children = []
    errors = []
    cards = []
    try:
        for card in range(8):
            children.append(subprocess.Popen([sys.executable, '-B', __file__, '--worker', str(deadline), str(card)],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for child in children:
            stdout, stderr = child.communicate(timeout=max(1, deadline-time.time()))
            if child.returncode:
                errors.append(stderr[-2500:])
            else:
                cards.append(json.loads(stdout))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
    if errors:
        sys.stderr.write('\n'.join(errors))
        return 1
    if fresh_inventory() != pins or any(KFD.iterdir()):
        raise ValueError('PILOT_POST_PCI_OR_KFD')
    if len(cards) != 8 or {r['pci'] for r in cards} != set(pins) or not os.environ['SLURM_STEP_ID'].isdecimal():
        raise ValueError('PILOT_OBSERVED_PCI_AND_STEP')
    evidence = dict(cards=cards,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],post_clean=True)
    with Path(path).open('x', encoding='utf-8') as output:
        json.dump(evidence, output, allow_nan=False)
    return 0


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(float(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(main(sys.argv[1], float(sys.argv[2])))
