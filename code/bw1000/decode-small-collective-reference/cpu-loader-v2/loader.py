"""Audit the exact already-linked DeepSeek CCL DSO under two loader environments."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BASE = Path("/refactor/candidates/Deep-decode-chain-reference-cpu-v1")
CONTRACT = Path("/refactor/incoming/Deep-decode-chain-reference/cpu-v1/BUILD_CONTRACT.json")
DSO = BASE / "libDecodeSmallCollective.so"
ORIGINAL_PRODUCER = "910376.922"
ORIGINAL_DSO = "3bbcc833031caafa9a245ffe0eb832f9e92df9febef0344cf25d125389e4dbf7"
ORIGINAL_MD = "85bcc3979fdc2f717baeab5fa5561ad6771351e5fb3c3d0363b0d48392c47373"
LEGACY_DIRS = ("/opt/dtk/hip/lib", "/opt/hyhal/lib", "/.singularity.d/libs")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def inspect(path, search):
    env = {**os.environ, "LD_LIBRARY_PATH": search, "LC_ALL": "C"}
    run = subprocess.run(["/usr/bin/ldd", str(path)], env=env, capture_output=True,
                         text=True, timeout=25)
    mapped = dict(re.findall(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(", run.stdout, re.M))
    missing = re.findall(r"^\s*(\S+)\s+=>\s+not found\s*$", run.stdout, re.M)
    return run.returncode, mapped, sorted(set(missing))


def main():
    claim_sha = os.environ["DEEP_DECODE_CHAIN_LOADER_CLAIM_SHA"]
    here = Path(__file__).resolve().parent
    assert digest(here / "CPU_RESOURCE_CLAIM.json") == claim_sha
    claim = json.loads((here / "CPU_RESOURCE_CLAIM.json").read_bytes())
    assert claim["CPU_GO"] and not claim["GPU_GO"] and not claim["Model_GO"]
    assert digest(Path(__file__)) == claim["loader_sha256"]
    assert os.environ["SLURM_JOB_ID"] == claim["parent_job"]
    assert all(os.environ[name] == "" for name in
               ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
    assert digest(DSO) == ORIGINAL_DSO and digest(BASE / "raw_ar.d") == ORIGINAL_MD
    assert digest(CONTRACT) == claim["original_contract_sha256"]
    plan = json.loads(CONTRACT.read_bytes())
    before = os.environ["LD_LIBRARY_PATH"]
    assert before == claim["original_search"]
    after = ":".join([before, *LEGACY_DIRS])
    base_rc, _, base_missing = inspect(DSO, before)
    rc, mapped, missing = inspect(DSO, after)
    expected = plan["resolved_runtime_paths"]
    exports = subprocess.check_output(["/usr/bin/nm", "-D", "--defined-only", str(DSO)],
                                      text=True, timeout=15)
    names = sorted(row.split()[-1] for row in exports.splitlines()
                   if " decode_ccl_ar_" in row)
    required = {}
    for soname, path in expected.items():
        if mapped.get(soname) == path:
            pin = digest(Path(path))
            if path not in plan["link_input_sha256"] or plan["link_input_sha256"][path] == pin:
                required[soname] = {"path": path, "sha256": pin}
    passed = rc == 0 and not missing and required.keys() == expected.keys() and names == plan["required_exports"]
    result = dict(passed=passed, actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  linked_producer=ORIGINAL_PRODUCER, native_sha256=ORIGINAL_DSO,
                  dependency_file_sha256=ORIGINAL_MD, build_contract_sha256=digest(CONTRACT),
                  claim_sha256=claim_sha, native_exports=names, required_runtime_libraries=required,
                  observed_runtime_paths=mapped, baseline_ldd_rc=base_rc,
                  baseline_missing=base_missing, candidate_ldd_rc=rc, candidate_missing=missing,
                  resolved_search=after, compiled=False, gpu_executed=False, python_dlopen=False)
    out = Path(claim["destination"])
    out.mkdir()
    with (out / "CPU_BUILD_RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
