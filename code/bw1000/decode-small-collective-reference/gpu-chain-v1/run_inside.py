"""Bind the published claim and loader receipt before entering the existing worker."""
from hashlib import sha256
import json
import os
from pathlib import Path
import socket

root = Path(__file__).resolve().parent
raw = (root / "GPU_RESOURCE_CLAIM.json").read_bytes()
assert sha256(raw).hexdigest() == os.environ["DEEP_RAW_CHAIN_CLAIM_SHA"]
q = json.loads(raw)
assert q["GPU_GO"] and q["once"] and not q["Model_GO"]
assert (socket.gethostname().split(".")[0], os.environ["SLURM_JOB_ID"]) == (q["node"], q["parent_job"])
for name, pin in q["files_SHA256"].items():
    assert sha256((root / name).read_bytes()).hexdigest() == pin
receipt = Path(q["loader_receipt_path"])
data = receipt.read_bytes()
assert sha256(data).hexdigest() == q["loader_receipt_sha256"]
r = json.loads(data)
assert r["passed"] and r["actual_step"] == "910376.923" and r["linked_producer"] == "910376.922"
assert r["native_sha256"] == q["native_sha256"] and not r["candidate_missing"]
env = dict(os.environ, LD_LIBRARY_PATH=r["resolved_search"])
argv = ["/usr/bin/python3", "-B", "-u", str(root / "worker.py"),
        "--tp", "8", "--deep-chain", "--native", q["native_path"],
        "--native-sha256", q["native_sha256"], "--build-receipt", str(receipt),
        "--build-receipt-sha256", q["loader_receipt_sha256"], "--timeout", "30",
        "--output-dir", q["output_dir"]]
os.execve(argv[0], argv, env)
