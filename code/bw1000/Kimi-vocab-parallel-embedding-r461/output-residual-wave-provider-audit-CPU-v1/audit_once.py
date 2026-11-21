"""Re-read the .678 Kimi link's original provider list in its pinned CPU container."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
import time


CANDIDATES = Path('/refactor/candidates')
SOURCE = CANDIDATES / 'Kimi461-output-residual-wave-component-link-CPU-v2'
PARENT, NODE = '1103606', 'f12r2n13'
CUDART = Path('/opt/dtk/cuda/cuda/lib64/libcudart.so.12')
CUDART_SHA = 'ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e'
REQUIRED = {'_infinilm.cpython-310-x86_64-linux-gnu.so', 'libinfinilm_runtime.so'}
ORIGINAL = {
    SOURCE / 'CPU-result.json': '0e5b4d7630bff9fee65bbcdcb2ae75515e8ec2972e4fab24735d1a4b4763a2e4',
    SOURCE / 'full-tree-passive.log': '0e03e8a1d8893063eaead39c0a8edf4e7a3537ef588185384fc15f0706e2927c',
    SOURCE / 'full-tree.so': '8fde97a0954117687a5e00d4026855f5aa3a3e3c6202290cbfe96ce4241224be',
    CANDIDATES / 'Kimi461-output-residual-tile64-CPU-v1/CPU-result.json': '6d9421acefdb6091694ffe8429c3c87b90239e00eea9c95afb28a23844cb65e1',
    CANDIDATES / 'Kimi461-output-residual-stage-diagnostic-CPU-v1/CPU-result.json': '8b40413bbf8a9ef6cef6fe3b79c1efd5ba0977d994e052cdc0866303b3b94f43',
}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def fixed(path):
    need(path.is_file() and path.resolve(strict=True) == path, 'original scoped file identity differs')
    with path.open('rb') as stream:
        raw = stream.read(4*1024**2 + 1)
    need(0 < len(raw) <= 4*1024**2 and sha256(raw).hexdigest() == ORIGINAL[path], 'original scoped bytes differ')
    return raw


def canonical_providers(text, resolve):
    """Match the lookup rule used by the failed linker, retaining the original aliases."""
    need('not found' not in text and 'undefined symbol' not in text, 'original log has unresolved relocations')
    aliases = {}
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        path = words[2] if len(words) > 2 and words[1] == '=>' else words[0]
        if path.startswith('/'):
            aliases[path] = resolve(path)
    need(bool(aliases), 'original log has no absolute provider paths')
    return aliases


def path_gates(loaded, qualified, prior, cuda):
    return dict(cuda_provider_present=cuda in loaded,
                required_basenames_missing=sorted(REQUIRED - {Path(x).name for x in loaded}),
                prior_paths_missing=sorted(prior - loaded),
                undeclared_loaded_paths=sorted(loaded - qualified.keys()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--epoch', type=float, required=True)
    args = parser.parse_args()
    epoch, deadline = args.epoch, args.epoch + 300
    job, step = (os.environ[k] for k in ('SLURM_JOB_ID','SLURM_STEP_ID'))
    host = socket.gethostname().split('.')[0]
    visible = [os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')]
    need(job == PARENT and host == NODE and step.isdecimal() and len(os.sched_getaffinity(0)) == 4
         and visible == ['-1']*3 and not any(os.environ.get(k) for k in ('SLURM_STEP_GPUS','LD_PRELOAD','LD_AUDIT'))
         and epoch <= time.time() < deadline - 15, 'original zero-card scope or budget differs')
    prior = json.loads(fixed(SOURCE / 'CPU-result.json'))
    need(prior['step'] == '1103606.678' and prior['completed'] is True and prior['passed'] is False
         and prior['reason'] == 'component provider identity differs'
         and [x['phase'] for x in prior['commands']][-1] == 'full-tree-passive',
         'original host verification did not stop at the provider rule')
    fixed(SOURCE / 'full-tree.so')
    prior330 = json.loads(fixed(CANDIDATES / 'Kimi461-output-residual-tile64-CPU-v1/CPU-result.json'))
    prior335 = json.loads(fixed(CANDIDATES / 'Kimi461-output-residual-stage-diagnostic-CPU-v1/CPU-result.json'))
    need(prior330['actual_CPU'] == '1103606.330' and prior335['actual_CPU'] == '1103606.335'
         and all(v['completed'] is True and v['passed'] is True for v in (prior330, prior335)), 'prior host controls differ')
    aliases = canonical_providers(fixed(SOURCE / 'full-tree-passive.log').decode(),
                                 lambda path: str(Path(path).resolve(strict=True)))
    cuda = str(CUDART.resolve(strict=True))
    qualified = {**prior330['actual_passive_providers'], cuda:CUDART_SHA}
    loaded = set(aliases.values())
    gates = path_gates(loaded,qualified,set(prior335['passive_providers']),cuda)
    path_pass = gates['cuda_provider_present'] and not any(x for k,x in gates.items() if k != 'cuda_provider_present')
    report = dict(source='1103606.678',step=f'{job}.{step}',host=host,epoch=epoch,elapsed_seconds=None,
                  loaded_count=len(loaded),aliases=aliases,original_path_rule_pass=path_pass,path_gates=gates,
                  byte_mismatches=[],original_digests_complete=False,provider_bytes_read=0,
                  GPU_calls=0,native_loading_calls=0,link_calls=0,compiled_objects=0)
    if path_pass:
        for name in sorted(loaded):
            h = sha256()
            with Path(name).open('rb') as stream:
                before = os.fstat(stream.fileno())
                need(stat.S_ISREG(before.st_mode),'provider is not a regular file')
                for raw in iter(lambda: stream.read(1024**2), b''):
                    report['provider_bytes_read'] += len(raw)
                    need(time.time() < deadline-12 and report['provider_bytes_read'] <= 8*1024**3,
                         'original provider digest budget exhausted')
                    h.update(raw)
                after = os.fstat(stream.fileno())
                need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) ==
                     (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'provider changed while reading')
            if h.hexdigest() != qualified[name]:
                report['byte_mismatches'].append(dict(path=name,pinned=qualified[name],observed=h.hexdigest()))
        report['original_digests_complete'] = True
    report['elapsed_seconds'] = time.time()-epoch
    with args.result.open('x',encoding='utf8') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
