"""Verify the prepared model package on CPU without importing or loading Native."""
from pathlib import Path
import json
import os
import subprocess

import sys
sys.path.insert(0, "/refactor/incoming/Deep-C4-compact-bank-pair/deployment-v1")
from support import D, PACKAGE, ENTRY_MF, sha, verify

LIB = Path("/refactor/candidates/Deep-C4-compact-bank-CPU-private-state-v2/libDeep_c4_compact_bank_v1.so")
NATIVE = "7fcea960c5f218234ed8a4733fa7bd94b31055ff7396755c2afd6f5bba502f90"
RECEIPT = "710e6046deb71019f7e5e015206d08c630100b9999f9cc884307903e50e9bf38"


def main():
    assert os.environ["SLURM_JOB_ID"] == "910407" and os.environ["SLURM_STEP_ID"].isdecimal()
    assert all(os.environ.get(k) == "-1" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    claim_path = Path(__file__).with_name("CPU_RESOURCE_CLAIM.json")
    assert sha(claim_path) == os.environ["DEEP_C4_PAIR_ENTRY_CPU_CLAIM_SHA"]
    claim = json.loads(claim_path.read_bytes())
    assert claim["CPU_GO"] and claim["once"] and not claim["GPU_GO"] and not claim["Model_GO"]
    for name, value in claim["files_SHA256"].items():
        assert sha(claim_path.parent / name) == value
    assert sha(D / "SOURCE_MANIFEST.json") == claim["deployment_manifest_sha256"]
    assert ENTRY_MF == claim["entry_manifest_sha256"]
    proof = verify(sha(D / "SOURCE_MANIFEST.json"), require_ready=False)
    assert not proof["execution_READY"] and proof["entry_CPU"] is None
    assert sha(LIB) == NATIVE and sha(LIB.with_name("CPU-build.json")) == RECEIPT
    result = subprocess.run(
        ["python3", "-B", str(PACKAGE / "entry.py"), "--source-sha", ENTRY_MF,
         "--config", str(D / "config.json"), "--config-sha", sha(D / "config.json")],
        check=True, text=True, capture_output=True, timeout=45)
    schema = json.loads(result.stdout)
    assert schema["source_schema"] and not any(schema[k] for k in ("Native_dlopen", "Runtime", "weights", "GPU", "Model_READY"))
    path = Path("/refactor/candidates/Deep-C4-compact-bank-pair-entry-CPU-v1")
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
