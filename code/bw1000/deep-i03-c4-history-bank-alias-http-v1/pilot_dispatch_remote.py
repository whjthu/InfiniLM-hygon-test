import base64
import hashlib
import json
import subprocess
import sys
import time

wire = json.loads(base64.b64decode(WIRE))
claim = wire["claim"]
label = claim["label"]
parent, node = claim["parent"], claim["node"]
assert (parent, node, claim["gpu_step_count_max"]) == ("1087656", "f12r3n12", 1)
assert claim["deadline"]["seconds"] == 7200
blobs = {name: base64.b64decode(data) for name, data in wire["files"].items()}
for name, raw in blobs.items():
    assert hashlib.sha256(raw).hexdigest() == claim["files_SHA256"][name]
def compact(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
def emit(kind, **data):
    print(json.dumps({"controller": kind, **data}, separators=(",", ":")), flush=True)
def check(args, timeout=15):
    result = subprocess.run(args, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed with exit {result.returncode}")
    return result.stdout.decode()
def accounting():
    data = check(["sacct", "-n", "-P", "-j", parent,
                  "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw"])
    rows = []
    for line in data.splitlines():
        columns = line.split("|")
        if len(columns) == 5 and columns[0].startswith(parent + "."):
            rows.append(dict(zip(("step", "label", "state", "exit_code", "elapsed_seconds"), columns)))
    return rows
def terminal(name, maximum):
    expires = time.monotonic() + maximum
    while True:
        rows = [row for row in accounting() if row["label"] == name]
        if len(rows) > 1:
            raise RuntimeError(f"multiple Slurm steps for {name}")
        if rows and rows[0]["state"].split()[0] in {
                "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}:
            return rows[0]
        if time.monotonic() >= expires:
            return None
        time.sleep(1)
def active():
    result = check(["squeue", "--steps", "-h", "-j", parent, "-o", "%i|%j|%N"])
    for line in result.splitlines():
        ident, name, hosts = line.split("|")
        if ident in {parent + ".batch", parent + ".extern"}:
            continue
        members = check(["scontrol", "show", "hostnames", hosts]).splitlines() if hosts and hosts != "(null)" else []
        if name in {label, label + "-pre", label + "-post"} or node in members:
            raise RuntimeError("existing owned step label or active target-node step")
def script_literal(name):
    return repr(base64.b64encode(blobs[name]).decode())
split_pin = claim["files_SHA256"]["split_admission.py"]
module_code = (
    "import base64,hashlib,socket,sys,types;"
    "m=types.ModuleType('split_admission');sys.modules[m.__name__]=m;"
    f"p=base64.b64decode({script_literal('split_admission.py')});"
    f"assert hashlib.sha256(p).hexdigest()=={split_pin!r};"
    "exec(compile(p,'split_admission.py','exec'),m.__dict__);"
)
python = "/usr/local/python3.12/bin/python3"
def snapshot(name):
    cpu_code = ("with socket.socket() as port:\n port.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n port.bind(('127.0.0.1',43886))\n"
                "import json;print(json.dumps(m.snapshot(False)),flush=True)")
    raw = check(["srun", "--jobid="+parent, "--job-name="+name, "--exact", "--cpu-bind=none",
                 "--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--nodelist="+node,
                 "--gres=none", "--mem=8192M", "--immediate=5", "--time=00:01:00",
                 python, "-B", "-u", "-c", module_code + "\n" + cpu_code], timeout=78)
    matches = [line for line in raw.splitlines() if line.startswith("{")]
    if len(matches) != 1:
        raise RuntimeError("missing unique CPU snapshot")
    row = json.loads(matches[0])
    acc = terminal(name, 22)
    if acc is None or (acc["step"], acc["state"], acc["exit_code"]) != (row["step"], "COMPLETED", "0:0"):
        raise RuntimeError("CPU snapshot lacks clean terminal accounting")
    if row["kfd_holder_count"] != 0 or len(row["pci"]) != 8:
        raise RuntimeError("CPU snapshot failed the holder or PCI check")
    return dict(row, accounting_state=acc["state"], exit_code=acc["exit_code"])
start_row = check(["squeue", "-h", "-j", parent, "-o", "%T|%D|%N"]).strip().split("|")
assert start_row[:2] == ["RUNNING", "40"] and len(start_row) == 3
assert node in check(["scontrol", "show", "hostnames", start_row[2]]).splitlines()
names = {label, label + "-pre", label + "-post"}
if names & {row["label"] for row in accounting()}:
    raise RuntimeError("an original step exists for this claim")
active()
from pathlib import Path
assert not Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/i03-deep-c4-alias-http-v1-private").exists()
before = snapshot(label + "-pre")
before_raw = compact(before)
emit("before", snapshot=before)
active()
gpin = claim["files_SHA256"]["pilot_host_gate.py"]
bootstrap = (module_code +
    "g=types.ModuleType('pilot_host_gate');"
    f"p=base64.b64decode({script_literal('pilot_host_gate.py')});"
    f"assert hashlib.sha256(p).hexdigest()=={gpin!r};"
    "exec(compile(p,'pilot_host_gate.py','exec'),g.__dict__);raise SystemExit(g.main())")
service_names = {"alias_policy.py","alias_summary.py","diagnostic_binding.py","input_binding.json","pilot_inputs.py","pilot_observation.py","pilot_run.py","scoring.py","service_config.json","wire_client_r290.py"}
service_bundle = compact({name: base64.b64encode(blobs[name]).decode() for name in sorted(service_names)})
assert len(service_bundle) <= 131072
bundle = {"before": base64.b64encode(before_raw).decode(),
          "inside": base64.b64encode(blobs["pilot_inside.sh"]).decode(),
          "probe": base64.b64encode(service_bundle).decode()}
epoch = time.time()
deadline = epoch + claim["deadline"]["seconds"]
argv = ["srun", *claim["srun_options"], python, "-B", "-u", "-c", bootstrap,
        "--before-sha256", hashlib.sha256(before_raw).hexdigest(),
        "--inside-sha256", claim["files_SHA256"]["pilot_inside.sh"],
        "--probe-sha256", hashlib.sha256(service_bundle).hexdigest(),
        "--deadline-epoch", str(deadline)]
emit("dispatch_intent", epoch=epoch, deadline=deadline, claim_sha256=wire["claim_sha256"], label=label)
with subprocess.Popen(argv, stdin=subprocess.PIPE) as process:
    process.communicate(input=compact(bundle))
    rc = process.returncode
emit("srun_return", exit_code=rc)
acc = terminal(label, 60)
emit("accounting", row=acc)
if acc is None:
    raise SystemExit(70)
post = snapshot(label + "-post")
emit("post", snapshot=post)
raise SystemExit(0 if rc == 0 and acc["state"] == "COMPLETED" and acc["exit_code"] == "0:0" else 1)
