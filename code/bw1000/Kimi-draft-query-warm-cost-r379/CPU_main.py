"""Actual container CPU dispatcher/supervisor closure; no torch or weights."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

D = pathlib.Path(__file__).resolve().parent
R = pathlib.Path('/refactor')
O = R/'candidates/Kimi-draft-query-warm-cost-r379-CPU-v1'
O.mkdir(exist_ok=False)
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
started = time.perf_counter()
q = dict(completed=False, passed=False, actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
         GPU=0, Native_dlopen=False, checkpoint_loaded=False, target_loaded=False)
try:
    assert all(os.environ.get(name) == '' for name in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'])
    manifest = D/'CPU_SOURCE_MANIFEST.json'
    q['source_manifest_SHA256'] = sha(manifest)
    assert q['source_manifest_SHA256'] == os.environ['KIMI_R379_CPU_SHA']
    source = json.loads(manifest.read_bytes())
    assert all(sha(D/name) == expected for name, expected in source['files_SHA256'].items())
    assert all(sha(path) == expected for path, expected in source['dependencies_SHA256'].items())
    completed = subprocess.run([sys.executable, '-B', str(D/'check_CPU.py'),
                               '--runtime-dir', str(R/'incoming/Kimi-long-draft-RPC-CPU-r333')],
                              text=True, capture_output=True, timeout=120)
    (O/'checks.log').write_text(completed.stdout+completed.stderr)
    q['checks_returncode'] = completed.returncode
    assert completed.returncode == 0
    q['checks'] = json.loads([line for line in completed.stdout.splitlines() if line.startswith('{')][-1])
    assert q['checks']['passed'] and not q['checks']['GPU_initialized'] and not q['checks']['checkpoint_loaded']
    q['passed'] = True
except BaseException as error:
    q.update(error=repr(error), traceback=traceback.format_exc())
finally:
    q.update(completed=True, elapsed_s=time.perf_counter()-started)
    (O/'CPU-result.json').write_text(json.dumps(q, indent=2)+'\n')
    print(json.dumps(q))
raise SystemExit(0 if q['passed'] else 1)
