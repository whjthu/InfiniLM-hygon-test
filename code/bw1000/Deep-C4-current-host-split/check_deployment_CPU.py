"""Verify the prepared model package on CPU without importing or loading Native."""
from pathlib import Path
import json
import os
import subprocess

from support import D, PACKAGE, ENTRY_MF, sha, verify

LIB = Path("/refactor/candidates/Deep-C4-current-host-split-CPU-v1/libDeep_c4_host_split_v1.so")
NATIVE = "1b729a2d29d39f084addc4978fbe3db7bd4b685360c384f7dfe2ae8908fc9e7f"
RECEIPT = "775a26846037e5925cf3e3343569c8c39140cd78d65fbee44bf8c72118d9ea28"


def main():
    assert os.environ["SLURM_JOB_ID"] == "910407" and os.environ["SLURM_STEP_ID"].isdecimal()
    assert all(os.environ.get(k) == "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    proof = verify(sha(D / "SOURCE_MANIFEST.json"), require_ready=False)
    assert not proof["execution_READY"] and proof["entry_CPU"] is None
    assert sha(LIB) == NATIVE and sha(LIB.with_name("CPU-build.json")) == RECEIPT
    result = subprocess.run(
        ["python3", "-B", str(PACKAGE / "entry.py"), "--source-sha", ENTRY_MF,
         "--config", str(D / "config.json"), "--config-sha", sha(D / "config.json")],
        check=True, text=True, capture_output=True, timeout=45)
    schema = json.loads(result.stdout)
    assert schema["source_schema"] and not any(schema[k] for k in ("Native_dlopen", "Runtime", "weights", "GPU", "Model_READY"))
    path = Path("/refactor/candidates/Deep-C4-current-host-split-entry-CPU-v1")
    path.mkdir()
    q = dict(completed=True, passed=True, entry_nonserve_executed=True, source_manifest_SHA256=ENTRY_MF,
             deployment_manifest_SHA256=sha(D / "SOURCE_MANIFEST.json"),
             actual_CPU=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
             Native_SHA256=NATIVE, Native_CPU_receipt_SHA256=RECEIPT,
             Native_dlopen=False, SDK_Runtime=False, weights_opened=False, GPU=False, Model_READY=False)
    with (path / "result.json").open("x", encoding="utf8") as stream:
        json.dump(q, stream, indent=2)
    print(json.dumps(q))


if __name__ == "__main__":
    main()
