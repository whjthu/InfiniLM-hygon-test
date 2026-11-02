"""Immutable actual device entry; CPU imports do not initialize a GPU."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import os
import sys


def main(pin, deadline):
    d = Path(__file__).resolve().parent
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    mf = d / 'GPU_SOURCE_MANIFEST.json'
    assert sha(mf) == pin
    m = json.loads(mf.read_bytes())
    assert m['GPU_GO'] and not m['Model_GO']
    assert all(sha(d / p) == h for p, h in m['files_SHA256'].items())
    source = Path('/refactor/incoming/Kimi-I192-bounded-BF16-r401')
    sm = source / 'SOURCE_MANIFEST.json'
    assert sha(sm) == m['CPU_source_manifest_SHA256']
    assert all(sha(source / p) == h for p, h in json.loads(sm.read_bytes())['files_SHA256'].items())
    runtime_path = Path('/refactor/results/Kimi401-GPU-runtime.json')
    execution = json.loads(runtime_path.read_bytes())
    assert execution['deadline_unix'] == deadline
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    cudart = ctypes.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12')
    pci = cudart.cudaDeviceGetPCIBusId
    pci.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    pci.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(64)
    assert pci(buf, 64, 0) == 0
    domain,bus,device=buf.value.decode().lower().split(':')
    identity=f'{int(domain,16):04x}:{int(bus,16):02x}:{device}'
    execution['device_identity'] = dict(visible_count=1, current=0, PCI_bus_ID=identity)
    temp = runtime_path.with_suffix('.device.tmp')
    temp.write_text(json.dumps(execution, indent=2) + '\n'); temp.replace(runtime_path)
    # The immutable GPU overlay changes only evidence-before-failstop Python;
    # C++/objects/MD remain exactly CPU801. plan.py is byte-identical to CPU source.
    sys.path.insert(0, str(d))
    from GPU_run import main as run
    output_path = Path('/refactor/results/Kimi401-' + os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'] + '.json')
    run(runtime_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--deadline-unix', required=True, type=float)
    args = parser.parse_args()
    main(args.manifest_sha, args.deadline_unix)
