"""Materialize the pinned framework and execute one bounded CPU step."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import time

R = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-qwen-deterministic-ckpt25-values-v1"
PARENT = "1103606"
NODE = "f13r1n10"
W = R / "results" / LABEL
COMPUTE_SHA="025ff50c1ebfc19164f63e4d842ffa6a205be6332d26050f74d0be6fb50ff3c7"


ORIGINAL = {'outcome.json': '71c9ac1cfd9f23fb9067a17b15421f7d4ec0173e30fa4f8f0285e27b1cf249e9', 'native-outcome.json': '0f117ed0928ec7d3aade150c21b4e8ec5b077a87d9dbdaafbf656e95ced2878e', 'continuous/native-outcome.json': '3f499409b39376e87e8506b5cb1c0c9c3f0c989f68935cfc1a6dfabb5efc4685', 'resumed/native-outcome.json': '440cf0fedbd74f3f44e40c24d7a0bb3340e1f0f80b0d9cfd8e55c7713150e71e'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(name, value):
    with (W / name).open("x") as f:
        json.dump(value, f, indent=2)


def query(*argv):
    return subprocess.check_output(argv, timeout=10, text=True).strip()


def accounting():
    return [line.split("|") for line in query("sacct", "-n", "-P", "-j", PARENT,
        "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList").splitlines()
        if "|"+LABEL+"|" in line]


def main():
    epoch = time.time()
    started = time.monotonic()
    payload = json.load(sys.stdin)
    require(set(payload) == {"request","claim","package"}, "Unexpected payload")
    require(not W.exists() and not accounting(), "Existing scope requires reconciliation")
    state, count, hosts = query("squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N").split("|")
    nodes = query("scontrol", "show", "hostnames", hosts).splitlines()
    require(state == "RUNNING" and int(count) == len(set(nodes)) == 40 and NODE in nodes,
            "Frozen CPU node is unavailable")
    for line in query("squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N").splitlines():
        handle, hosts = line.split("|")
        if not handle.endswith((".batch", ".extern")) and hosts not in ("", "(null)"):
            require(NODE not in query("scontrol", "show", "hostnames", hosts).splitlines(),
                    "CPU node has an active step")
    W.mkdir(mode=0o700)
    write("intent.json", dict(epoch=epoch, deadline=epoch+290, request=payload["request"],
        claim=payload["claim"], pid=os.getpid(), parent=PARENT, node=NODE, GPU_cards=0))
    first_root = R/'results/training-qwen-deterministic-math-recovery-v1'
    original = first_root/'continuous'
    resumed = first_root/'resumed'
    for relative, digest in ORIGINAL.items():
        path = first_root/relative
        require(hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'Original deterministic candidate terminal differs')
    rows = query('sacct','-n','-P','-j','1087656','--format=JobIDRaw,State,ExitCode,NodeList').splitlines()
    for expected in ('1087656.426|COMPLETED|0:0|a01r4n16','1087656.427|COMPLETED|0:0|a01r4n16'):
        require(expected in rows, 'Original candidate terminal is not independently complete')
    first_different = {}
    for rank in range(8):
        label = f'rank-{rank:02d}/result.json'
        a = json.loads((original/'train/continuous-tail'/label).read_bytes())
        b = json.loads((resumed/'train/resume-witness'/label).read_bytes())
        assert a['rank']==b['rank']==rank and a['completed_global_step']==b['completed_global_step']==25
        def rows(record, what, key):
            result = [(x['ordinal'], x.get('global_step'), x[key]['SHA256']) for x in record[what]]
            assert len(result)==20 and len({x[0] for x in result})==20
            return result
        assert rows(a,'inputs','full_input')==rows(b,'inputs','full_input')
        losses = list(zip(rows(a,'outputs','loss'), rows(b,'outputs','loss')))
        assert all(x[:2]==y[:2] for x,y in losses)
        first_different[str(rank)] = next((dict(suffix_microbatch=i, ordinal=x[0], global_step=x[1])
                                          for i,(x,y) in enumerate(losses,1) if x[2]!=y[2]),None)
    write('first_loss_difference.json',first_different)
    fix = W/'fix'; fix.mkdir()
    require(set(payload['package'])=={'compute.py'}, 'Unexpected CPU source inventory')
    data=base64.b64decode(payload['package']['compute.py'],validate=True)
    require(hashlib.sha256(data).hexdigest()==COMPUTE_SHA, 'Unexpected CPU math checkpoint reader')
    with (fix/'compute.py').open('xb') as f:f.write(data)
    result=json.loads((first_root/'native-outcome.json').read_bytes())['result']
    pins={}
    for n in ('adapter_model.safetensors','optimizer.pt'):
        pins[n]={f'{role}_{field}':result[f'{role}25'][n][field] for role in ('continuous','resumed') for field in ('bytes','SHA256')}
    with (fix/'PINS.json').open('x',encoding='utf-8') as f:json.dump(pins,f)
    evidence=W/"evidence"
    evidence.mkdir()
    require(time.monotonic() - started < 85 and time.time() < epoch+85, "Admission budget exhausted")
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
        "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 "
        "HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; "
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/swift-runtime/site QWEN_CPU_DEADLINE="
        + shlex.quote(str(epoch+230)) + " QWEN_CPU_EPOCH=" + shlex.quote(str(epoch))
        + "; exec python3 -B -u /fix/compute.py")
    command = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
    binds = [(original, "/original", "ro"), (resumed, "/resumed", "ro"),
             (fix, "/fix", "ro"), (evidence, "/evidence", "rw"),
             (R/"results/training-vlm-swift-offline-runtime-v1/site", "/swift-runtime/site", "ro"),
             (Path("/opt/hyhal"), "/opt/hyhal", "ro")]
    for source, target, mode in binds:
        command += ["--bind", f"{source}:{target}:{mode}"]
    command += [str(R/"containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell]
    step = ["srun", "--jobid="+PARENT, "--job-name="+LABEL, "--exact", "--nodes=1", "--ntasks=1",
        "--nodelist="+NODE, "--cpus-per-task=4", "--mem=8192M", "--gres=none",
        "--immediate=5", "--time=4", *command]
    forced = False
    with (W/"CPU.log").open("xb") as log:
        p = subprocess.Popen(step, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            stat = Path(f"/proc/{p.pid}/stat").read_text().split(") ",1)[1].split()
            write("owned-srun.json", {"pid":p.pid, "pgid":int(stat[2]), "startticks":int(stat[19])})
            p.wait(timeout=max(1,min(epoch+242-time.time(),242-(time.monotonic()-started))))
        finally:
            if p.poll() is None:
                forced = True
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=5)
                except ProcessLookupError:
                    p.wait(timeout=5)
            rows = []
            while time.monotonic()-started < 265 and time.time() < epoch+265:
                rows = accounting()
                if rows and rows[0][2] not in ("RUNNING","PENDING","COMPLETING"):
                    break
                time.sleep(0.5)
            receipt = evidence/"compute.json"
            compute = json.loads(receipt.read_bytes()) if receipt.is_file() else None
            passed = bool(compute and len(rows)==1 and rows[0][0]==compute["step"] and
                          rows[0][2:4]==["COMPLETED","0:0"] and p.returncode==0 and not forced)
            value = dict(passed=passed, rows=rows, rc=p.returncode, forced=forced, compute=compute,
                         log_SHA256=hashlib.sha256((W/"CPU.log").read_bytes()).hexdigest(),
                         request=payload["request"], claim=payload["claim"],first_loss_difference=first_different)
            write("outcome.json",value)
            print(json.dumps({**value, "compute": None if compute is None else {k:compute[k] for k in ("adapter","optimizer_moments_by_serialized_parameter_id","step","GPU_cards")}}),flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
