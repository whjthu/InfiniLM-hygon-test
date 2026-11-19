"""Exercise the exact operator on one visible DCU before the fleet is eligible."""
import json
import os
from pathlib import Path
import subprocess
import sys

from node_sample import fresh_inventory
from worker import pci_for_device

KFD = Path('/sys/class/kfd/kfd/proc')


def worker(deadline):
    import torch

    from fc2_operator import exercise

    torch.set_num_threads(1)
    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('ONE_ASSIGNED_HIP_DEVICE')
    device = torch.device('cuda', 0)
    pci = pci_for_device(0, fresh_inventory())
    result = exercise(device, 0, lambda: None, deadline, calls_per_block=128)
    rows = result['blocks']
    if (len(rows) != 8 or set(row['arm'] for row in rows) != {'cublas', 'cublaslt'}
            or any(row['calls'] != 128 or row['max_abs_against_first_cublas'] > .05 for row in rows)):
        raise ValueError('PILOT_BOUNDED_NUMERICAL_CHECK')
    expected = {'cublas': '_BlasBackend.Cublas', 'cublaslt': '_BlasBackend.Cublaslt'}
    if any(row['observed_preference'] != expected[row['arm']] for row in rows):
        raise ValueError('PILOT_DISTINCT_OBSERVED_PREFERENCES')
    print(json.dumps(dict(pci=pci, result=result), allow_nan=False), flush=True)


def main(path, deadline):
    pins = fresh_inventory()
    if any(KFD.iterdir()):
        raise ValueError('PILOT_PRE_KFD')
    child = subprocess.run([sys.executable, '-B', __file__, '--worker', str(deadline)],
                           capture_output=True, text=True, timeout=43)
    if child.returncode:
        sys.stderr.write(child.stderr[-2500:])
        return child.returncode
    if fresh_inventory() != pins or any(KFD.iterdir()):
        raise ValueError('PILOT_POST_PCI_OR_KFD')
    evidence = json.loads(child.stdout)
    if evidence['pci'] not in pins or not os.environ['SLURM_STEP_ID'].isdecimal():
        raise ValueError('PILOT_OBSERVED_PCI_AND_STEP')
    evidence.update(step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], post_clean=True)
    with Path(path).open('x', encoding='utf-8') as output:
        json.dump(evidence, output, allow_nan=False)
    return 0


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(float(sys.argv[2]))
    else:
        raise SystemExit(main(sys.argv[1], float(sys.argv[2])))
