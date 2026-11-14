"""Run one resident GLM step and persist original accounting and cleanup."""

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import hashlib
import signal
import subprocess
import time

from selected_native import ARMS
from audit import verify_owners
from shared import D, HOST, NODES, PARENT, REQUIRED_FILES, H, load_pinned, namespace, require, save, verify_manifest
from slurm_accounting import wait_terminal
from bootstrap import PORT as BOOTSTRAP_PORT
from budget import GPU_SECONDS, INSIDE_SECONDS, WHOLE_SECONDS, WORK_SECONDS, remaining

COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"
PYTHON = "/usr/local/python3.12/bin/python3"
ARM = "socket"


def query(*command, timeout=10):
    return subprocess.check_output(command, text=True, timeout=timeout).splitlines()


def expand(nodes):
    return set(query("scontrol", "show", "hostnames", nodes))


def check_idle():
    parents = query("squeue", "-h", "-j", PARENT, "-o", "%T|%N")
    require(len(parents) == 1, "I02 allocation missing")
    state, hosts = parents[0].split("|")
    nodes = expand(hosts)
    require(state == "RUNNING" and len(nodes) == 40 and set(NODES) <= nodes, "I02 allocation changed")
    check_live_steps(query("scontrol", "show", "step", PARENT, "-o"))


def check_live_steps(lines):
    require(bool(lines), "controller returned no steps for the running allocation")
    for line in lines:
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        step = fields.get("StepId", "")
        require(re.fullmatch(re.escape(PARENT) + r"\.(?:[0-9]+|batch|extern)", step),
                "unrecognized controller step")
        if step.endswith((".batch", ".extern")):
            continue
        hosts = fields.get("NodeList")
        require(hosts not in {None, "", "(null)", "None", "(none)"},
                "controller step has unresolved nodes")
        nodes = expand(hosts)
        require(nodes and not (set(NODES) & nodes), "live or unresolved Slurm step overlaps I02 pair")


def account(label, deadline):
    remaining = deadline - time.monotonic()
    require(remaining > 0, "accounting deadline exhausted")
    rows = [line.split("|") for line in query(
        "sacct", "-n", "-P", "-j", PARENT, "--format=" + COLUMNS,
        timeout=min(10, remaining)) if "|" + label + "|" in line]
    require(len(rows) <= 1, "duplicate step label")
    if rows:
        require(len(rows[0]) == 6 and re.fullmatch(PARENT + r"\.\d+", rows[0][0]),
                "unexpected step accounting")
    return rows[0] if rows else None


def command(label, phase, out):
    gpu = phase == "gate"
    return [
        "srun", "--jobid=" + PARENT, "--job-name=" + label, "--exact", "--immediate=5",
        "--nodes=2", "--ntasks=2", "--ntasks-per-node=1", "--distribution=block:block",
        "--cpu-bind=none", "--nodelist=" + ",".join(NODES), "--kill-on-bad-exit=1",
        "--cpus-per-task=" + ("64" if gpu else "1"), "--mem=" + ("224000M" if gpu else "1024M"),
        "--gres=" + ("dcu:8" if gpu else "none"), "--time=" + (str(GPU_SECONDS // 60) if gpu else "1"),
        PYTHON, "-B", str(D / "host_probe.py"), phase, str(out),
    ]


def run_step(label, phase, out, env, end, whole_end):
    before = time.monotonic()
    require(before < end, "step budget exhausted")
    log = out / (phase + ".log")
    receipt = out / (phase + "-receipt.json")
    result = dict(phase=phase, label=label, state="SUBMISSION_POSSIBLE", terminal=False,
                  passed=False, accounting_columns=COLUMNS.split(","), observations=[])
    with receipt.open("x") as stream:
        json.dump(result, stream)
        stream.flush()
        os.fsync(stream.fileno())
    child = None
    try:
        with log.open("xb") as stream:
            child = subprocess.Popen(command(label, phase, out), env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            result["srun_PID"] = child.pid
            stat = Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ", 1)[1].split()
            result["srun_startticks"] = int(stat[19])
            save(receipt, result)
            child.wait(timeout=max(.01, end - time.monotonic()))
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if child is not None and child.poll() is None:
            result["forced"] = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        deadline = min(whole_end, time.monotonic() + 24)

        def observe():
            row = account(label, deadline)
            result["observations"].append(row)
            save(receipt, result)
            return row

        try:
            observed = wait_terminal(observe, deadline, interval=1, max_observations=16)
            row = observed["row"]
            result.update(terminal=observed["terminal_visible"], row=row)
            result["passed"] = bool(
                row and row[2:4] == ["COMPLETED", "0:0"] and child and child.returncode == 0
                and "error" not in result and not result.get("forced") and expand(row[5]) == set(NODES)
            )
        except Exception as exc:
            result["accounting_error"] = type(exc).__name__ + ": " + str(exc)
        result["state"] = "TERMINAL" if result["terminal"] else "UNKNOWN"
        result["seconds"] = time.monotonic() - before
        save(receipt, result)
    return result


def evidence(out, phase, step):
    result = {}
    for node in NODES:
        row = json.loads((out / phase / (node + ".json")).read_bytes())
        if phase == "gate":
            mapping = row["pci_to_kfd"]
            require(set(mapping) == set(row["pci"]) and len(set(mapping.values())) == 8,
                    "leased GPU identities differ")
        require(row["step"] == step and row["node"] == node and row["kfd_holder_count"] == 0
                and set(row["ports_bindable"]) == {"43843", "43844", str(BOOTSTRAP_PORT)}
                and all(row["ports_bindable"].values()) and len(set(row["pci"])) == 8,
                "GPU ownership evidence differs from accounting")
        result[node] = row
    return result


def verify_released(pre, post):
    for node in NODES:
        require(pre[node]["pci"] == post[node]["pci"] and pre[node]["boot_id"] == post[node]["boot_id"]
                and pre[node]["monotonic_ns"] <= post[node]["monotonic_ns"],
                "post-run hardware identity changed")


def run_resident(run, claim_sha, source_sha, epoch, whole_end):
    label, container_out = namespace(run, ARM)
    out = HOST / "results" / label
    check_idle()
    for phase in ("pre", "gate", "post"):
        require(account(label + "-" + phase, min(whole_end, time.monotonic() + 12)) is None,
                "resident step was already submitted")
    out.mkdir(mode=0o700)
    for phase in ("pre", "gate", "post"):
        (out / phase).mkdir(mode=0o700)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(
        ("SLURM_", "PMI_", "PMIX_", "CUDA_", "HIP_", "ROCR_", "GPU_"))
        or k in {"SLURM_CONF", "SLURM_CONF_SERVER"}}
    summary = dict(run_id=run, claim_SHA256=claim_sha, epoch=epoch,
                   planned_requests=8, terminal=False, passed=False, cleanup_verified=False,
                   planned_measured_pairs=3, stages={})
    outcome = out / "outcome.json"
    save(outcome, summary)

    def step(phase, env, until):
        result = run_step(label + "-" + phase, phase, out, env, until, whole_end)
        summary["stages"][phase] = result
        save(outcome, summary)
        return result

    try:
        pre = step("pre", clean, min(whole_end - 120, time.monotonic() + 45))
        require(pre["passed"], "pre-run Slurm probe failed")
        before = evidence(out, "pre", pre["row"][0])
        key = secrets.token_hex(32)
        manifest = dict(schema="glm474-selected-value-worker-v1", run_id=run, arm=ARM, epoch=epoch,
                        parent=PARENT, nodes=NODES, output=str(container_out), Native_SHA256=ARMS[ARM][1],
                        GPU_GO=True, Model_execution_authorized=True, Model_qualified=False,
                        source_manifest_SHA256=source_sha, claim_SHA256=claim_sha,
                        transport_key_SHA256=hashlib.sha256(bytes.fromhex(key)).hexdigest())
        verify_manifest(manifest, run, ARM, epoch)
        save(out / "RUN_MANIFEST.json", manifest)
        manifest_sha = H(out / "RUN_MANIFEST.json")
        env = dict(clean, GLM474_PAIR_ID=run, GLM474_PAIR_ARM=ARM,
                   GLM470_EPOCH=str(epoch), GLM470_GPU_MF=manifest_sha, GLM474_TRANSPORT=ARM,
                   GLM474_TRANSPORT_KEY=key)
        gpu = step("gate", env, min(whole_end - 120, time.monotonic() + remaining(epoch, GPU_SECONDS)))
        require(gpu["terminal"], "GPU accounting unresolved; retain reservation")
        summary["terminal"] = True
        post = step("post", clean, min(whole_end - 30, time.monotonic() + 45))
        require(post["passed"], "independent post-run step failed")
        verify_released(before, evidence(out, "post", post["row"][0]))
        summary["cleanup_verified"] = True
        save(outcome, summary)
        require(gpu["passed"], "GPU step failed; independent cleanup succeeded")
        gate = evidence(out, "gate", gpu["row"][0])
        summary["audit"] = verify_owners(out, manifest_sha, gpu["row"][0], gate, ARMS[ARM][1])
        summary["passed"] = True
        return summary
    except Exception as exc:
        summary["failure"] = {"type": type(exc).__name__, "message": str(exc)[:200]}
        raise
    finally:
        save(outcome, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("claim_sha256")
    parser.add_argument("--epoch", type=float, required=True)
    args = parser.parse_args()
    claim = load_pinned(D / "GPU_RESOURCE_CLAIM.json", args.claim_sha256)
    resources = dict(GPU_cards=16, CPU_per_node=64, memory_MiB_per_node=224000,
                     GPU_seconds=GPU_SECONDS, work_seconds=WORK_SECONDS,
                     inside_seconds=INSIDE_SECONDS, whole_seconds=WHOLE_SECONDS,
                     max_runs=1, planned_requests=8, model_context=32768,
                     max_new_tokens=4096, memory_floor_bytes=8589934592)
    require(all(type(claim[k]) is int and claim[k] == v for k, v in resources.items()),
            "resident resource claim differs")
    require(claim["schema"] == "glm474-selected-value-model-claim-v1"
            and claim["run_id"] == args.run_id and claim["parent"] == PARENT and claim["nodes"] == NODES
            and claim["GPU_GO"] is True and claim["native_arms"] == {k: v[1] for k, v in ARMS.items()},
            "resident claim is invalid")
    source_sha = claim["source_manifest_SHA256"]
    source = load_pinned(D / "SOURCE_MANIFEST.json", source_sha)
    host = load_pinned(D / "HOST_MANIFEST.json", claim["host_manifest_SHA256"])
    require(set(host["files_SHA256"]) == {
        "host_launch.py", "host_container.sh", "host_probe.py", "snapshot.py",
        "slurm_accounting.py", "audit.py", "dispatch_host.py"
    }, "incomplete HOST source manifest")
    require(source["files_SHA256"].keys() == REQUIRED_FILES, "incomplete worker source manifest")
    for files in (source, host):
        for name, digest in files["files_SHA256"].items():
            require(Path(name).name == name and H(D / name) == digest, "source digest differs")
    label, _ = namespace(args.run_id, ARM)
    require(remaining(args.epoch, GPU_SECONDS) > GPU_SECONDS - 90, "original launch epoch is too old")
    lock = HOST / "results" / (label + "-intent")
    with lock.open("x") as stream:
        json.dump(dict(claim=args.claim_sha256, PID=os.getpid(), epoch=args.epoch), stream)
        stream.flush()
        os.fsync(stream.fileno())
    run_resident(args.run_id, args.claim_sha256, source_sha, args.epoch,
                 time.monotonic() + remaining(args.epoch, WHOLE_SECONDS) - 25)


if __name__ == "__main__":
    main()
