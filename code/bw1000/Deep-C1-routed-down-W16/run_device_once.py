from hashlib import sha256
from pathlib import Path
import json, os, re, socket, subprocess, sys, time

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
D = Path(__file__).resolve().parent
LABEL = "Deep-C1-routed-down-W16-device-v1"
OUT = R / "results" / LABEL
PARENT = "910374"
NODE = "a02r4n05"
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "REVOKED", "DEADLINE"}


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def exact_rows():
    raw = subprocess.check_output(
        ["sacct", "-S", "2026-11-05", "-j", PARENT, "-n", "-P",
         "-o", "JobIDRaw,JobName%180,State,ExitCode,NodeList%100"],
        text=True, timeout=20)
    return [line.split("|") for line in raw.splitlines() if line]


def hosts(value):
    return subprocess.check_output(["scontrol", "show", "hostnames", value],
                                   text=True, timeout=15).splitlines()


def terminal(row):
    return row[2].split()[0].rstrip("+") in TERMINAL


def main(commit, expected):
    claim_file = D / "GPU_RESOURCE_CLAIM.json"
    assert re.fullmatch("[0-9a-f]{40}", commit) and digest(claim_file) == expected
    c = json.loads(claim_file.read_bytes())
    assert (c["GPU_GO"], c["Model_GO"], c["parent"], c["node"], c["cards"],
            c["max_wall_s"], c["once"]) == (True, False, PARENT, NODE, 8, 600, True)
    for name, sha in c["source_files_SHA256"].items():
        assert Path(name).name == name and digest(D / name) == sha
    outer = Path(c["outer"])
    assert digest(outer) == c["outer_SHA256"]
    build = R / Path(c["build_receipt"]).relative_to("/refactor")
    assert digest(build) == c["build_receipt_SHA256"]
    b = json.loads(build.read_bytes())
    assert b["completed"] and b["actual_step"] == "910374.1453" and b["library_SHA256"] == c["library_SHA256"]
    assert digest(R / Path(b["library"]).relative_to("/refactor")) == c["library_SHA256"]
    job = subprocess.check_output(["scontrol", "show", "job", PARENT, "-o"], text=True, timeout=20)
    assert "JobState=RUNNING" in job and "UserId=aclsylqidf(" in job
    assert NODE in hosts(re.search(r"(?:^| )NodeList=([^ ]+)", job).group(1))
    for row in exact_rows():
        assert not row[1].startswith(LABEL)
        if re.fullmatch(PARENT + r"\.\d+", row[0]) and not terminal(row):
            assert NODE not in hosts(row[4])
    assert not list((R / "logs").glob(LABEL + "-*"))
    OUT.mkdir()
    state = dict(state="INTENT", publication=commit, resource_claim_SHA256=expected,
                 origin_HOST=socket.gethostname(), PID=os.getpid(),
                 startticks=Path("/proc/self/stat").read_text().split()[21], GPU=None)

    def save():
        p = OUT / "watch.tmp"
        p.write_text(json.dumps(state, indent=2) + "\n")
        p.replace(OUT / "watch.json")

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "PMI_", "PMIX_", "CUDA_", "HIP_", "ROCR_"))}
    base = ["srun", "--jobid=" + PARENT, "--overlap", "--exact", "--immediate=20",
            "--cpu-bind=none", "-N1", "-n1", "-w", NODE, "--export=ALL"]

    def run(phase, resource, command, wait):
        name = LABEL + "-" + phase
        argv = base + resource + ["--job-name=" + name,
                                  "--output=" + str(R / "logs" / (name + "-%J.log"))] + command
        state.update(state=phase, phase_started_unix=time.time(), current_argv=argv)
        save()
        p = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL)
        state.update(active_srun_PID=p.pid, active_srun_startticks=Path("/proc/%d/stat" % p.pid).read_text().split()[21])
        save()
        code = p.wait(timeout=wait)
        state.update(last_srun_rc=code)
        save()
        return code

    def consume(phase):
        limit = time.monotonic() + 35
        while time.monotonic() < limit:
            rows = [row for row in exact_rows() if row[1] == LABEL + "-" + phase]
            assert len(rows) <= 1
            if rows and terminal(rows[0]):
                assert hosts(rows[0][4]) == [NODE]
                return rows[0]
            time.sleep(3)
        raise TimeoutError("original step terminal accounting not visible")

    try:
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            env[name] = "-1"
        cpu = ["-c4", "--gres=none", "--mem=8192M", "-t", "00:02:00"]
        observer = ["python3", "-B", str(D / "observe_node.py")]
        assert run("pre", cpu, observer + ["pre", str(OUT)], 155) == 0
        state["pre"] = consume("pre")
        assert state["pre"][2:4] == ["COMPLETED", "0:0"] and (OUT / "pre.json").is_file()
        for row in exact_rows():
            if re.fullmatch(PARENT + r"\.\d+", row[0]) and not terminal(row):
                assert NODE not in hosts(row[4])
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            del env[name]
        env["WAVE_INSIDE_R2323"] = "Deep-C1-routed-down-W16/device-v1/GPU_inside.sh"
        actual = dict(c, remaining_deadline_unix=time.time() + c["max_wall_s"],
                      publication_commit=commit, resource_claim_SHA256=expected)
        with (OUT / "GPU_ACTUAL_CLAIM.json").open("x", encoding="utf8") as stream:
            json.dump(actual, stream, indent=2)
        gpu_rc = run("gpu", ["-c32", "--gres=dcu:8", "--mem=128000M", "-t", "00:10:00",
                             "--kill-on-bad-exit=1"], ["bash", str(outer), "probe"], 635)
        state["GPU"] = consume("gpu")
        save()
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            env[name] = "-1"
        post_rc = run("post", cpu, observer + ["post", str(OUT)], 155)
        state["post"] = consume("post")
        assert post_rc == 0 and state["post"][2:4] == ["COMPLETED", "0:0"] and (OUT / "post.json").is_file()
        assert gpu_rc == 0 and state["GPU"][2:4] == ["COMPLETED", "0:0"]
        state["state"] = "DEVICE_TERMINAL_PENDING_INDEPENDENT_CONSUMER"
    except Exception as error:
        state.update(state="STOPPED_RECONCILE_ORIGINAL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main(*sys.argv[1:])
