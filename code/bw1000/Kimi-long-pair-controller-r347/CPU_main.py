"""CPU-only actual orchestrator contracts; Slurm/Native remain recording boundaries."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from contract import sha, verify_bundle


def main():
    d = Path(__file__).resolve().parent
    assert all(os.environ.get(k) == '' for k in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'])
    assert not os.environ.get('LD_PRELOAD') and 'torch' not in sys.modules
    manifest_sha = os.environ['KIMI_LONG_CONTROLLER_SOURCE_SHA_R347']
    verify_bundle(d, manifest_sha)
    output = Path('/refactor/candidates/Kimi-long-pair-controller-r347')
    output.mkdir(exist_ok=False)
    report = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  completed=False, passed=False, GPU=0, Native_dlopen=False,
                  actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                  CPU_only=True, controller_manifest_SHA256=manifest_sha,
                  source_manifest_SHA256=manifest_sha, actual_target_or_draft_submitted=False,
                  actual_model_weights_loaded=False, no_GPU_or_Model_qualification=True)
    start = time.monotonic()
    try:
        result = subprocess.run([sys.executable, '-B', str(d/'check_controller_CPU.py')],
                                capture_output=True, text=True, timeout=240)
        log = output/'checks.log'; log.write_text(result.stdout+result.stderr, encoding='utf8')
        report.update(test_process_rc=result.returncode, log_SHA256=sha(log))
        assert result.returncode == 0, 'CPU orchestration checks failed; preserve log'
        lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
        checks = json.loads(lines[-1])
        assert checks['passed'] and checks['completed'] and checks['cases']
        report.update(checks=checks, passed=True)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
    finally:
        report.update(completed=True, elapsed_s=time.monotonic()-start)
        (output/'CPU-result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
        print(json.dumps({k:report[k] for k in ['completed', 'passed', 'actual_step', 'elapsed_s']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
