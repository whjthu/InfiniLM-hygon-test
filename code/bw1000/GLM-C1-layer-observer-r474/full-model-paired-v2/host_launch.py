"""Run each frozen GLM arm at most once, serially, under the existing I02 job."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from native_arm import ARMS
from shared import D, HOST, NODES, PARENT, REQUIRED_FILES, H, load_pinned, namespace, require, save, verify_manifest
from slurm_accounting import wait_terminal
from budget import GPU_SECONDS, INSIDE_SECONDS, PAIR_SECONDS, WORK_SECONDS, remaining, require_pair_capacity

COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList"
PYTHON = "/usr/local/python3.12/bin/python3"
ARM_ORDER = ("baseline", "single-drain")


def query(*command, timeout=10):
    return subprocess.check_output(command, text=True, timeout=timeout).splitlines()


def expand(nodes):
    return set(query("scontrol", "show", "hostnames", nodes))


def check_idle():
    parents = query("squeue", "-h", "-j", PARENT, "-o", "%T|%N")
    require(len(parents) == 1, "I02 allocation missing")
    state, hosts = parents[0].split("|")
    require(state == "RUNNING" and set(NODES) <= expand(hosts), "I02 allocation changed")
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


def run_step(label, phase, out, env, end):
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
        deadline = time.monotonic() + 24

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
                and all(row["ports_bindable"].values()) and len(set(row["pci"])) == 8,
                "GPU ownership evidence differs from accounting")
        result[node] = row
    return result


def verify_released(pre, post):
    for node in NODES:
        require(pre[node]["pci"] == post[node]["pci"] and pre[node]["boot_id"] == post[node]["boot_id"]
                and pre[node]["monotonic_ns"] <= post[node]["monotonic_ns"],
                "post-run hardware identity changed")


def verify_owners(out, arm, manifest, gpu_step, gate):
    from resident_trial import phases, schedule

    pins = json.loads((D / "INPUT_BINDING.json").read_bytes())["private_plan"]["requests"]
    request_fields = ("ordinal", "case_id", "suite", "prompt_IDs_SHA256", "body_SHA256")
    output_hashes = None
    barrier_digests = None
    for node_index, node in enumerate(NODES):
        worker = json.loads((out / f"worker-{node_index}.json").read_bytes())
        require(worker["completed"] and worker["passed"] and worker["actual_GPU"] == gpu_step
                and worker["identity"]["arm"] == arm and worker["identity"]["run_manifest_SHA256"] == manifest,
                "leader did not complete the selected run")
        require(not worker.get("forced_TERM") and not worker.get("forced_KILL")
                and [row["phase"] for row in worker["rounds"]] == phases(),
                "incomplete or forced leader shutdown")
        digests = [row["digest"] for row in worker["rounds"]]
        require(barrier_digests in (None, digests), "leader barrier digests differ")
        barrier_digests = digests
        pci = set()
        for device in range(8):
            rank = node_index * 8 + device
            owner = json.loads((out / f"owner-{rank:02d}.json").read_bytes())
            require(owner["global_rank"] == rank and owner["node"] == node
                    and owner["actual_GPU"] == gpu_step and owner["Native_arm"] == arm
                    and owner["Native_SHA256"] == ARMS[arm][1] and owner["GPU_manifest_SHA256"] == manifest
                    and owner["completed"] and owner["passed"] and owner["normal_close"]
                    and owner["Runtime_weak_expired_CABI_assertion"] and owner["resets"] == 66
                    and len(owner["requests"]) == 33 and owner["all78layers_loaded"]
                    and owner["loaded_parameters"] == owner["checkpoint_key_count"]
                    and owner["Native_forward_attempts"] == owner["Native_forward_completed"]
                    and owner["driver_memory"]["min_observed"] >= 8 * 1024**3
                    and not owner["driver_memory"]["errors"]
                    and all(v["free"] >= 8 * 1024**3 for v in owner["memory_boundaries"]),
                    "owner did not complete the selected run")
            for item, request in zip(schedule(), owner["requests"]):
                pin = pins[item["case_index"]]
                require(all(request[field] == pin[field] for field in request_fields)
                        and request["diagnostic"] == item["diagnostic"]
                        and request["completed"] and request["reset_before"] and request["reset_after"]
                        and bool(request["natural_EOS"]) != bool(request["cap_hit"]),
                        "rank request identity or completion differs")
            hashes = [request["generated_token_ids_SHA256"] for request in owner["requests"]]
            require(output_hashes in (None, hashes), "output hashes disagree across ranks")
            output_hashes = hashes
            pci.add(owner["physical_PCI"])
        require(pci == set(gate[node]["pci_to_kfd"]), "owner PCI differs from assigned KFD inventory")


def run_arm(run_id, arm, claim_sha, source_sha, results):
    label, container_out = namespace(run_id, arm)
    out = HOST / "results" / label
    check_idle()
    for phase in ("pre", "gate", "post"):
        require(account(label + "-" + phase, time.monotonic() + 12) is None, "step already submitted")
    out.mkdir(mode=0o700)
    for phase in ("pre", "gate", "post"):
        (out / phase).mkdir(mode=0o700)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(
        ("SLURM_", "PMI_", "PMIX_", "CUDA_", "HIP_", "ROCR_", "GPU_"))
        or k in {"SLURM_CONF", "SLURM_CONF_SERVER"}}
    pre = run_step(label + "-pre", "pre", out, clean, time.monotonic() + 45)
    require(pre["passed"], "pre-run Slurm probe did not pass")
    before = evidence(out, "pre", pre["row"][0])
    if results:
        verify_released(results["baseline"]["hardware"], before)
    epoch = time.time()
    manifest = dict(schema="glm474-paired-worker-v2", run_id=run_id, arm=arm, epoch=epoch,
                    parent=PARENT, nodes=NODES, output=str(container_out), Native_SHA256=ARMS[arm][1],
                    GPU_GO=True, Model_execution_authorized=True, Model_qualified=False,
                    source_manifest_SHA256=source_sha, claim_SHA256=claim_sha)
    verify_manifest(manifest, run_id, arm, epoch)
    save(out / "RUN_MANIFEST.json", manifest)
    manifest_sha = H(out / "RUN_MANIFEST.json")
    env = dict(clean, GLM474_PAIR_ID=run_id, GLM474_PAIR_ARM=arm,
               GLM470_EPOCH=str(epoch), GLM470_GPU_MF=manifest_sha)
    gpu = run_step(label + "-gate", "gate", out, env, time.monotonic() + remaining(epoch, GPU_SECONDS))
    require(gpu["terminal"], "GPU accounting unresolved; reservation retained")
    post = run_step(label + "-post", "post", out, clean, time.monotonic() + 45)
    require(post["passed"], "post-run Slurm probe did not pass")
    verify_released(before, evidence(out, "post", post["row"][0]))
    require(gpu["passed"], "GPU step failed; post-run ownership checks passed")
    gate = evidence(out, "gate", gpu["row"][0])
    if results:
        require(all(results["baseline"]["hardware"][node]["pci_to_kfd"] == gate[node]["pci_to_kfd"]
                    for node in NODES), "paired leased KFD mapping changed")
    verify_owners(out, arm, manifest_sha, gpu["row"][0], gate)
    return gpu["row"][0], gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("claim_sha256")
    parser.add_argument("--pair-epoch", type=float, required=True)
    args = parser.parse_args()
    claim = load_pinned(D / "GPU_RESOURCE_CLAIM.json", args.claim_sha256)
    resources = dict(GPU_cards=16, CPU_per_node=64, memory_MiB_per_node=224000,
                     GPU_seconds=GPU_SECONDS, work_seconds=WORK_SECONDS,
                     inside_seconds=INSIDE_SECONDS, whole_seconds=PAIR_SECONDS)
    require(all(type(claim[k]) is int and claim[k] == v for k, v in resources.items()),
            "wrong paired run resource claim")
    require(claim["run_id"] == args.run_id and claim["parent"] == PARENT and claim["nodes"] == NODES
            and claim["GPU_GO"] is True and claim["max_runs_per_arm"] == 1
            and claim["native_arms"] == {k: v[1] for k, v in ARMS.items()}, "invalid paired claim")
    require(claim["schema"] == "glm474-paired-model-claim-v2", "wrong paired claim schema")
    require_pair_capacity(args.pair_epoch, len(ARM_ORDER))
    source_sha = claim["source_manifest_SHA256"]
    source = load_pinned(D / "SOURCE_MANIFEST.json", source_sha)
    host_source = load_pinned(D / "HOST_MANIFEST.json", claim["host_manifest_SHA256"])
    require(set(host_source["files_SHA256"]) == {
        "host_launch.py", "host_container.sh", "host_probe.py", "snapshot.py", "slurm_accounting.py"
    }, "incomplete HOST source manifest")
    for files in (source["files_SHA256"], host_source["files_SHA256"]):
        for name, digest in files.items():
            require(Path(name).name == name and H(D / name) == digest, "packaged source differs")
    require(source["files_SHA256"].keys() == REQUIRED_FILES, "incomplete worker manifest")
    labels = [namespace(args.run_id, arm)[0] for arm in ARM_ORDER]
    require(not any((HOST / "results" / label).exists() for label in labels), "run already reserved")
    lock = HOST / "results" / ("GLM474-pair-" + args.run_id + "-intent")
    with lock.open("x") as stream:
        json.dump(dict(claim=args.claim_sha256, PID=os.getpid(), epoch=time.time(), arms=ARM_ORDER), stream)
        stream.flush()
        os.fsync(stream.fileno())
    results = {}
    for index, arm in enumerate(ARM_ORDER):
        require_pair_capacity(args.pair_epoch, len(ARM_ORDER) - index)
        step, hardware = run_arm(args.run_id, arm, args.claim_sha256, source_sha, results)
        results[arm] = dict(step=step, hardware=hardware)
        save(HOST / "results" / ("GLM474-pair-" + args.run_id + "-progress.json"), results)
        print(arm, step, flush=True)


if __name__ == "__main__":
    main()
