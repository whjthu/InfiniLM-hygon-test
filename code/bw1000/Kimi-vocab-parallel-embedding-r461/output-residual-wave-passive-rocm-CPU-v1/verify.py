"""Resolve the original Kimi .678 full-tree.so with ROCm's qualified libnuma."""

from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time


LABEL = "Kimi461-original678-wave-passive-rocm-CPU-v1"
PARENT, NODE = "1103606", "f12r2n13"
REQUEST = "f02bc5837890441fbe80a1fab68b598b"
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OUTPUT = ROOT / "results" / LABEL
AUDIT_SHA = "1d39f824fc8074c8c9d5bbf989cc03c323fdc68233327db80b35ae8bf5710c3b"
NUMA = "/.singularity.d/libs/libnuma.so.1"
NUMA_SHA = "4a0ebd05ec13854e7a875d8187d0153b57188f2a247ae20025e09a31075261d1"
OTHER_NUMA = "/usr/lib/x86_64-linux-gnu/libnuma.so.1.0.0"
LDD_SIZE = 32 * 1024
BYTE_LIMIT = 8 * 1024**3
ABSOLUTE_LINE = re.compile(r"\s*(?:\S+\s+=>\s+)?/[^\s]+\s+\(0x[0-9a-f]+\)\s*")
VDSO_LINE = re.compile(r"\s*linux-vdso\.so\.1\s+\(0x[0-9a-f]+\)\s*")


def audit_module():
    path = Path(__file__).resolve().with_name("audit_once.py")
    with path.open("rb") as stream:
        source = stream.read(8193)
    if path.resolve(strict=True) != path or len(source) > 8192 or sha256(source).hexdigest() != AUDIT_SHA:
        raise ValueError("original .729 audit source differs")
    spec = importlib.util.spec_from_file_location("kimi_original_provider_audit", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def inspect_loader(audit, raw, resolve):
    audit.need(0 < len(raw) <= LDD_SIZE, "passive loader output exceeds its declared bound")
    loader = raw.decode("utf8")
    audit.need(all(ABSOLUTE_LINE.fullmatch(line) or VDSO_LINE.fullmatch(line)
                   for line in loader.splitlines() if line.strip()),
               "unexpected loader diagnostic or unresolved relocation")
    return audit.canonical_providers(loader, resolve)


def closure(audit, raw, qualified, prior, cuda, resolve, digest, report):
    aliases = inspect_loader(audit, raw, resolve)
    loaded = set(aliases.values())
    report.update(loaded_count=len(loaded), aliases=aliases,
                  path_gates=audit.path_gates(loaded, qualified, prior, cuda))
    audit.need(NUMA in prior and qualified.get(NUMA) == NUMA_SHA
               and NUMA in loaded and OTHER_NUMA not in loaded,
               "original .330/.335 canonical ROCm libnuma differs")
    gates = report["path_gates"]
    audit.need(gates["cuda_provider_present"] and not any(
        gates[key] for key in ("required_basenames_missing", "prior_paths_missing", "undeclared_loaded_paths")
    ), "original .678 closure path rule differs")
    report["original_path_rule_pass"] = True
    for name in sorted(loaded):
        actual = digest(name)
        if actual != qualified[name]:
            report["byte_mismatches"].append(dict(path=name, pinned=qualified[name], observed=actual))
    report["original_digests_complete"] = True
    audit.need(not report["byte_mismatches"], "original .678 provider SHA256 rule differs")


def write_once(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(epoch):
    audit = audit_module()
    deadline = epoch + 280
    now = time.time()
    job, step = (os.environ[key] for key in ("SLURM_JOB_ID", "SLURM_STEP_ID"))
    host = socket.gethostname().split(".")[0]
    visible = [os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")]
    audit.need(job == PARENT and host == NODE and step.isdecimal() and len(os.sched_getaffinity(0)) == 4
               and visible == ["-1"] * 3
               and not any(os.environ.get(key) for key in ("SLURM_STEP_GPUS", "LD_PRELOAD", "LD_AUDIT"))
               and math.isfinite(epoch) and 0 < epoch <= now < epoch + 240,
               "audit requires its original one-node zero-GRES scope and epoch")
    audit.need(OUTPUT.is_dir() and OUTPUT.resolve(strict=True) == OUTPUT, "audit output scope differs")
    report = dict(source_step="1103606.678", source_state="FAILED", observed_step=f"{job}.{step}",
                  epoch=epoch, deadline=deadline, request=REQUEST, node=host, passed=False,
                  original_path_rule_pass=False, original_digests_complete=False,
                  byte_mismatches=[], provider_bytes_read=0, provider_count_hashed=0,
                  passive_loader_calls=0, GPU_calls=0, native_application_loads=0,
                  new_link_calls=0, compiled_objects=0, waves_checked=["full-tree"])
    write_once(OUTPUT / "worker-attempt.json", dict(request=REQUEST, step=report["observed_step"], epoch=epoch))
    dso = audit.SOURCE / "full-tree.so"
    try:
        original = json.loads(audit.fixed(audit.SOURCE / "CPU-result.json"))
        audit.need(original["step"] == "1103606.678" and original["completed"] is True
                   and original["passed"] is False and original["reason"] == "component provider identity differs"
                   and original["commands"][-1]["phase"] == "full-tree-passive",
                   "original linked artifact scope differs")
        audit.fixed(audit.SOURCE / "full-tree-passive.log")
        audit.need(len(audit.fixed(dso)) == 556104, "original linked artifact size differs")
        report["original_dso_SHA256"] = audit.ORIGINAL[dso]
        cpu330 = json.loads(audit.fixed(audit.CANDIDATES / "Kimi461-output-residual-tile64-CPU-v1/CPU-result.json"))
        cpu335 = json.loads(audit.fixed(audit.CANDIDATES / "Kimi461-output-residual-stage-diagnostic-CPU-v1/CPU-result.json"))
        audit.need(cpu330["actual_CPU"] == "1103606.330" and cpu335["actual_CPU"] == "1103606.335"
                   and all(row["completed"] is True and row["passed"] is True for row in (cpu330, cpu335)),
                   "provider reference qualification differs")
        cuda = str(audit.CUDART.resolve(strict=True))
        qualified = {**cpu330["actual_passive_providers"], cuda: audit.CUDART_SHA}
        prior = set(cpu335["passive_providers"])
        remaining = min(25, deadline - time.time() - 10)
        audit.need(remaining > 0, "passive loader deadline reached")
        report["passive_loader_calls"] = 1
        # The ldd process is read-only and scoped to the SHA-pinned trusted artifact.
        with (OUTPUT / "passive-ldd.log").open("xb") as log:
            run = subprocess.run(["/usr/bin/ldd", "-r", str(dso)], stdout=log, stderr=subprocess.STDOUT,
                                 check=False, timeout=remaining, env={**os.environ, "LC_ALL": "C"})
        with (OUTPUT / "passive-ldd.log").open("rb") as log:
            raw = log.read(LDD_SIZE + 1)
        report.update(loader_exit_code=run.returncode, loader_bytes=len(raw),
                      loader_SHA256=sha256(raw).hexdigest() if len(raw) <= LDD_SIZE else None)
        audit.need(run.returncode == 0, "passive loader exited unsuccessfully")

        def digest(name):
            h = sha256()
            with Path(name).open("rb") as file:
                before = os.fstat(file.fileno())
                audit.need(stat.S_ISREG(before.st_mode), "provider is not a regular file")
                for chunk in iter(lambda: file.read(1024**2), b""):
                    report["provider_bytes_read"] += len(chunk)
                    audit.need(time.time() < deadline - 5 and report["provider_bytes_read"] <= BYTE_LIMIT,
                               "provider digest budget exhausted")
                    h.update(chunk)
                current = Path(name).stat()
                after = os.fstat(file.fileno())
                identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                audit.need(all((row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns) == identity
                               for row in (current, after)), "provider changed while reading")
            report["provider_count_hashed"] += 1
            return h.hexdigest()

        closure(audit, raw, qualified, prior, cuda, lambda name: str(Path(name).resolve(strict=True)),
                digest, report)
        audit.fixed(dso)
        report["original_dso_rechecked"] = True
        report["passed"] = True
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        report["reason"] = str(error)[:512]
    report["elapsed_seconds"] = time.time() - now
    write_once(OUTPUT / "repair-result.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("expected launch epoch")
    raise SystemExit(main(float(sys.argv[1])))
