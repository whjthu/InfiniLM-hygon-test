"""Run new Qwen multi-event72 nativeCPU qualification and one continuous25 task."""

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

import mid_epoch_recovery
import owned_processes
from qwen_domain_recipe import DATASET, DATASET_SHA256, make_plan
from slurm_accounting import is_terminal, wait_terminal
from snapshot import snapshot
from data_view import build as build_view
from prepare_dev import TARGET as DEV_TARGET, PINS as DEV_PINS


RUN = Path(__file__).resolve().parents[1]
REFACTOR = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = REFACTOR / "results"
BASE = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1"
DATA = RESULTS / "training-gemma-usgs-render72-native-CPU-v1"
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
FRAMEWORK = RUN / "framework"
ARM = RUN / "continuous"
HOST_PYTHON = "/usr/local/python3.12/bin/python3"
NODE, PARENT, PORT = "f13r1n10", "1103606", 29941
LABEL = "training-qwen-usgs-multievent72-domain25-v1"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
PACKAGE_SECONDS = 2400
WORKER_SECONDS = 1700


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def port_available():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))
    return True


def asset(path, size, digest):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size == size and sha(path) == digest,
            f"asset identity differs: {path.name}")
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(info, field) == getattr(after, field) for field in fields), "asset changed while reading")


def prepare():
    cfg = read(RUN / "intent.json")
    require(cfg["node"] == NODE and cfg["parent"] == PARENT and cfg["arm"] == "continuous",
            "original resource intent differs")
    require(RUN == RESULTS / LABEL, "wrong independent run root")
    manifest = BASE / "source/FRAMEWORK_FILES.json"
    require(sha(manifest) == "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",
            "original common framework manifest differs")
    files = read(manifest)["files"]
    require(len(files) == 393, "common framework denominator differs")
    for name, pin in files.items():
        path = BASE / "framework" / name
        asset(path, pin["bytes"], pin["SHA256"])
        require(stat.S_IMODE(path.stat().st_mode) == pin["mode"], "common framework mode differs")
    updates = {}  # Continuous-only recipe; original393 mathematical/runtime sources unchanged.
    FRAMEWORK.mkdir(mode=0o700, exist_ok=False)
    for name in files:
        target = FRAMEWORK / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in updates:
            target.write_bytes(updates[name])
            target.chmod(files[name]["mode"])
        else:
            shutil.copy2(BASE / "framework" / name, target)
        expected = hashlib.sha256(updates[name]).hexdigest() if name in updates else files[name]["SHA256"]
        require(sha(target) == expected, "candidate framework file changed")
    original = BASE / "source/original-launch.json"
    binding = BASE / "source/QWEN_TARGET_NATIVE_BINDING.json"
    require(sha(binding) == "9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067",
            "original trainable binding differs")
    initial = read(binding)["initial_state"]
    asset(Path(initial["path"]), initial["bytes"], initial["SHA256"])
    target_data = DATA / 'data/train-swift.jsonl'
    require(sha(target_data)==DATASET_SHA256, 'existing target train72 bytes differ')
    inputs=read(RUN/'src/TRAIN_INPUTS.json')
    media=[x for x in inputs['files'] if x['target_relative'].startswith('data/train/media/')]
    require(len(media)==12, 'exact12 existing target train images required')
    for pin in media:
        asset(DATA/pin['target_relative'],pin['bytes'],pin['SHA256'])
    for name,pin in DEV_PINS.items():
        asset(Path(DEV_TARGET)/name,pin['bytes'],pin['SHA256'])
    dev_receipt=Path(DEV_TARGET)/'stage-complete.json'
    asset(dev_receipt,968,'9411e7c43788145a56e3fe1ca20cd163bf0b0f9329ff3f3642a9a8cffec3dd61')
    dev_stage=read(dev_receipt)
    require(dev_stage['passed'] is True and dev_stage['files']==DEV_PINS and
            dev_stage['labels_read'] is dev_stage['blind_read'] is False,
            'new dev36 input-only stage must complete before training')
    view=build_view(target_data,RUN/'view')
    save(RUN/'view-receipt.json',view)
    model_pins=read(RUN/'src/MODEL_ASSETS.json')['model']
    # Full source identity verification occurs in HOST; only10 small files enter CPU metadata view.
    metadata=RUN/'model-metadata-only'
    metadata.mkdir(mode=0o700)
    count=0
    for name,pin in model_pins.items():
        asset(MODEL/name,pin['bytes'],pin['SHA256'])
        if not name.endswith('.safetensors'):
            shutil.copy2(MODEL/name,metadata/name)
            asset(metadata/name,pin['bytes'],pin['SHA256'])
            count+=1
    require(count==10 and len(list(metadata.iterdir()))==10, 'exact Qwen metadata-only view differs')
    require(shutil.disk_usage(RUN).free >= 16*1024**3, 'insufficient native checkpoint storage')
    save(RUN/'prepare-result.json',dict(passed=True,train_source_rows=72,train_media=12,
         metadata_only_files=10,common_framework_files=393,view=view,
         dev_or_blind_gold_read=False,new_native72_qualified=False))


def prepare_plan():
    receipt=RUN/'native72-result.json'
    require(receipt.is_file(),'new native72 result missing')
    actual=read(receipt);intent=read(RUN/'intent.json')
    require(actual['step']==read(RUN/'native72-accounting.json')['row'][0] and
            actual['request']==intent['request_id'] and actual['epoch']==intent['epoch'] and
            actual['deadline']==intent['deadline'] and
            actual['source_manifest_SHA256']==intent['source_manifest_SHA256'] and
            actual['claim_SHA256']==intent['claim_SHA256'],'native72 request/actual original step differs')
    plan=make_plan(FRAMEWORK,(BASE/'source/original-launch.json').read_bytes(),receipt.read_bytes(),
        PurePosixPath(RUN.as_posix()),PurePosixPath((RUN/'src/qwen_treasury_witness.py').as_posix()),
        PORT,'continuous',PurePosixPath((RUN/'view/training800.jsonl').as_posix()),
        read(RUN/'view-receipt.json')['view']['SHA256'])
    save(RUN/'continuous-plan.json',plan)
    for plugin in plan['required_assets']['plugins']:
        require(Path(plugin).is_file(),'planned common callback missing')
    save(RUN/'witness-config.json',dict(original_binding=str(BASE/'source/QWEN_TARGET_NATIVE_BINDING.json'),
        view_SHA256=read(RUN/'view-receipt.json')['view']['SHA256'],source_unique_rows=72,physical_exposures=800,
        new_native72_result_SHA256=sha(receipt)))


def cpu_native():
    from native72 import check
    intent=read(RUN/'intent.json');epoch=intent['epoch']
    result=check(DATA/'data/train-swift.jsonl',RUN/'model-metadata-only',RUN,
                 min(epoch+PACKAGE_SECONDS-300,time.time()+240),context=intent)
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0


def phase_budget(epoch, now, requested, reserve=0):
    require(all(math.isfinite(x) for x in (epoch, now, requested, reserve))
            and epoch <= now and requested > 0 and reserve >= 0, "invalid original phase clock")
    available = epoch + PACKAGE_SECONDS - now - reserve
    require(available > 0, "original phase epoch exhausted")
    return min(requested, available)


def accounting_budget(epoch, now):
    require(math.isfinite(epoch) and math.isfinite(now) and epoch <= now, "invalid accounting clock")
    return min(40, max(0, epoch + PACKAGE_SECONDS - now - 5))


def command(args):
    epoch = read(RUN / "intent.json")["epoch"]
    return subprocess.check_output(args, text=True, timeout=phase_budget(epoch, time.time(), 14)).strip()


def accounting(label):
    rows = [line.split("|") for line in command(["sacct", "-n", "-P", "-j", PARENT, "-o", COLUMNS]).splitlines()]
    require(all(len(row)==7 for row in rows if any(x.strip() for x in row)), 'unexpected exact7 accounting schema')
    selected = [row for row in rows if len(row)==7 and row[1]==label]
    require(all(re.fullmatch(PARENT+r'\.[0-9]+',row[0]) and row[5]==NODE for row in selected),
            'exact label accounting parent/node differs')
    require(len(selected) <= 1, "duplicate Slurm step identity")
    return selected[0] if selected else None


def admit():
    ownership = command(["scontrol", "show", "job", PARENT, "-o"])
    settings = dict(re.findall(r"(\w+)=([^ ]+)", ownership))
    require(settings["UserId"].endswith("(" + str(os.getuid()) + ")")
            and "gres/dcu=320" in settings["AllocTRES"].split(","), "parent owner or DCU scope differs")
    record = command(["squeue", "-h", "-j", PARENT, "-o", "%T|%D|%N"]).split("|")
    require(record[:2] == ["RUNNING", "40"] and NODE in command(["scontrol", "show", "hostnames", record[2]]).splitlines(),
            "claim node is absent from the active parent")
    shown = command(["scontrol", "-o", "show", "step", PARENT])
    for line in shown.splitlines():
        fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
        if re.fullmatch(PARENT + r"\.\d+", fields.get("StepId", "")):
            require(NODE not in command(["scontrol", "show", "hostnames", fields["NodeList"]]).splitlines(),
                    "claimed node already has a compute step")


def step(name, args, gpu, seconds):
    label = LABEL + "-" + name
    require(accounting(label) is None, "existing step requires reconciliation")
    epoch = read(RUN / "intent.json")["epoch"]
    reserve = 150 if gpu or name == "pre" else 5
    seconds = phase_budget(epoch, time.time(), seconds, reserve)
    slurm_seconds = int(phase_budget(epoch, time.time(), 1800 if gpu else (300 if name=="native72" else 120), reserve))
    require(slurm_seconds >= 1, "insufficient original epoch for new Slurm step")
    slurm_time = f"00:{slurm_seconds // 60:02d}:{slurm_seconds % 60:02d}"
    cpu_count, memory = ((64,224000) if gpu else ((8,16384) if name=='native72' else (2,4096)))
    cmd = ["srun", "--jobid=" + PARENT, "--job-name=" + label, "--nodes=1", "--ntasks=1",
           "--nodelist=" + NODE, "--exact", "--immediate=5", "--cpus-per-task=" + str(cpu_count),
           "--mem=" + str(memory)+"M", "--gres=" + ("dcu:8" if gpu else "none"),
           "--time=" + slurm_time, *args]
    rc = None
    with (RUN / (name + ".log")).open("xb") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            pass
    status = wait_terminal(lambda: accounting(label),
                           time.monotonic() + accounting_budget(epoch, time.time()),
                           interval=2, max_observations=18)
    row = status["row"]
    tres={} if row is None else dict(x.split('=',1) for x in row[6].split(','))
    exact_devices={k:v for k,v in tres.items() if k.startswith('gres/')}
    result = dict(status, rc=rc, passed=bool(rc == 0 and row and row[2:4] == ["COMPLETED", "0:0"]
                  and row[5] == NODE and tres.get('node')=='1' and tres.get('cpu')==str(cpu_count)
                  and exact_devices==({'gres/dcu':'8'} if gpu else {})))
    save(RUN / (name + "-accounting.json"), result)
    return result


def cpu_snapshot():
    data = snapshot(False)
    require(data["kfd_holder_count"] == 0 and data["full_fd_inventory_performed"] is True,
            "host GPU ownership is not empty")
    data["port_available"] = port_available()
    data["owned_remaining"]=0
    for arm in ("continuous",):
        outcome=RUN/arm/"native-outcome.json"
        if outcome.exists():
            held=read(outcome)["owners"]
            current=owned_processes.snapshot()
            data["owned_remaining"]+=sum(int(pid) in current and current[int(pid)]["startticks"]==row["startticks"]
                                        and current[int(pid)]["state"]!="Z" for pid,row in held.items())
    require(data["owned_remaining"]==0,"native owned children remain")
    print(json.dumps(data), flush=True)


CHECKPOINT_NAMES={"trainer_state.json","args.json","adapter_model.safetensors","optimizer.pt","scheduler.pt","adapter_config.json"}
CHECKPOINT_NAMES.update("rng_state_%d.pth"%i for i in range(8))


def checkpoint_inventory(arm,number):
    checkpoint=RUN/arm/"train"/("checkpoint-%d"%number)
    pins={}
    started=time.monotonic()
    for name in sorted(CHECKPOINT_NAMES):
        p=checkpoint/name
        before=p.lstat()
        require(stat.S_ISREG(before.st_mode),"missing or aliased checkpoint member")
        pins[name]={"bytes":before.st_size,"SHA256":sha(p)}
        after=p.lstat()
        require(all(getattr(before,k)==getattr(after,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")),"checkpoint changed")
    state=read(checkpoint/"trainer_state.json")
    require(state["global_step"]==number,"saved trainer global step differs")
    proof=dict(arm=arm,global_step=number,files=pins,member_count=14,
               total_bytes=sum(x["bytes"] for x in pins.values()),
               native_storage_fields=["adapter_model.safetensors","optimizer.pt","scheduler.pt","trainer_state.json","args.json","adapter_config.json","rng_state_0.pth..rng_state_7.pth"],
               accumulation_boundary="after-complete-four-micro-update-witness",
               native_full_restore_checked=False,
               restore_observation_in="resumed/train/treasury-witness/rank-*/restored-before-RNG.json" if arm=="continuous" and number==20 else None,
               nonJSON_tensor_state_independently_loaded_here=False,
               wall_seconds=time.monotonic()-started)
    save(RUN/(arm+"-checkpoint-%d-files.json"%number),proof)
    return pins


def run_arm(arm,deadline):
    started=time.monotonic()
    global ARM
    require(arm in ("continuous","resumed"),"unknown arm")
    ARM=RUN/arm
    ARM.mkdir(mode=0o700,exist_ok=False)
    plan = read(RUN / (arm+"-plan.json"))
    config = RUN / (arm+"-witness-config.json")
    cfg=read(RUN / "witness-config.json")
    if arm=="resumed":
        checkpoint=RUN/"continuous/train/checkpoint-20"
        inventory=read(RUN/"continuous-checkpoint-20-files.json")
        require(inventory["global_step"]==20 and inventory["member_count"]==14,"continuous saved20 inventory differs")
        pins=inventory["files"]
        cfg.update(checkpoint=str(checkpoint),checkpoint_files=pins)
        save(RUN/"checkpoint20-files.json",pins)
    save(config,cfg)
    epoch = read(RUN / "intent.json")["epoch"]
    env = dict(os.environ, **plan["environment_overrides"])
    env.update(MASTER_ADDR="127.0.0.1", QWEN_TREASURY_CONFIG=str(config),
               QWEN_TREASURY_CONFIG_SHA256=sha(config), QWEN_TREASURY_EPOCH=str(epoch),
               QWEN_TREASURY_DEADLINE=str(deadline), QWEN_TREASURY_MAX_SECONDS=str(PACKAGE_SECONDS))
    require(not any(k in env for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")), "worker must start unranked")
    save(ARM / "native-intent.json", dict(plan_SHA256=sha(RUN / (arm+"-plan.json")), epoch=epoch,deadline=deadline,arm=arm,planned_optimizer_updates=25 if arm=="continuous" else 5))
    owned, rc, failed = {}, None, None
    try:
        with (ARM / "native.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-B", *plan["command"]], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            observed = owned_processes.snapshot().get(child.pid)
            if observed:
                owned[child.pid] = {k: observed[k] for k in ("PID", "PPID", "PGID", "startticks", "state")}
            while child.poll() is None:
                owned_processes.refresh(owned, ARM)
                require(time.time() < deadline, "native continuous deadline expired")
                time.sleep(1)
            rc = child.returncode
            quiet_until = time.monotonic() + 5
            while rc == 0 and owned_processes.refresh(owned, ARM) and time.monotonic() < quiet_until:
                time.sleep(.25)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        failed = type(error).__name__ + ": " + str(error)[:220]
    cleaned = owned_processes.cleanup(owned, ARM, grace=5)
    finished = []
    for rank in range(8):
        p = ARM / "train/treasury-witness" / f"rank-{rank:02d}" / "result.json"
        if p.is_file():
            entry = read(p)
            if entry["rank"] == rank and entry["completed_global_step"] == 25:
                finished.append(dict(rank=rank,path=str(p),bytes=p.stat().st_size,SHA256=sha(p)))
    passed = bool(rc == 0 and failed is None and len(finished) == 8 and not cleaned["termination_needed"])
    save(ARM / "native-outcome.json", dict(rc=rc, error=failed, cleanup=cleaned, owners=owned,
                                           finished=finished, passed=passed,arm=arm,
                                           planned_updates=25 if arm=="continuous" else 5,wall_seconds=time.monotonic()-started))
    return 0 if passed else 1


def run_metric(deadline):
    global ARM
    ARM=RUN/'effect'
    ARM.mkdir(mode=0o700)
    for rank in range(8):
        init=read(RUN/'continuous/train/treasury-witness'/f'rank-{rank:02d}'/'initialization.json')
        require(init.get('lora_B_all_zero') is True and init.get('lora_B_tensors')==252,
                'public-base effect comparator requires actually zero original LoRA B on all8 ranks')
    args=[sys.executable,'-B','-u',str(RUN/'src/metric_worker.py'),
          '--data',DEV_TARGET,'--model',str(MODEL),'--adapter',str(RUN/'continuous/train/checkpoint-25'),
          '--inventory',str(RUN/'continuous-checkpoint-25-files.json'),'--output',str(ARM/'generated'),
          '--intent',str(RUN/'intent.json'),'--intent-sha',sha(RUN/'intent.json'),
          '--deadline',str(deadline)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HIP_VISIBLE_DEVICES='0',ROCR_VISIBLE_DEVICES='0')
    owned={};child=None;rc=None;error=None
    try:
        with (ARM/'native.log').open('xb') as log:
            child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   env=env,start_new_session=True)
            current=owned_processes.snapshot().get(child.pid)
            if current:owned[child.pid]={k:current[k] for k in ('PID','PPID','PGID','startticks','state')}
            while child.poll() is None:
                owned_processes.refresh(owned,ARM)
                require(time.time()<deadline,'original metric deadline exhausted')
                time.sleep(.5)
            rc=child.returncode
    except (ValueError,OSError,subprocess.SubprocessError) as exc:
        error=type(exc).__name__+': '+str(exc)[:220]
    finally:
        cleaned=owned_processes.cleanup(owned,ARM,grace=5)
        if child is not None and child.poll() is None:
            child.kill();child.wait(timeout=3)
    path=ARM/'generated/generation-finalized.json'
    complete=read(path) if path.is_file() else None
    passed=bool(rc==0 and error is None and complete and complete['passed'] is True and
        complete['all72sealed_before_gold'] is True and not cleaned['termination_needed'] and
        cleaned['owned_processes_remaining']==0 and
        all(x['completed']==36 and x['missing']==0 for x in complete['arms'].values()))
    save(ARM/'native-outcome.json',dict(passed=passed,rc=rc,error=error,cleanup=cleaned,owners=owned,
        planned_predictions=72,completed=None if complete is None else sum(x['completed'] for x in complete['arms'].values()),
        generation_finalized=None if complete is None else dict(bytes=path.stat().st_size,SHA256=sha(path)),
        allocated_DCU=8,visible_inference_DCU=1,formal_performance=False))
    return 0 if passed else 1


def worker():
    before = read(RUN / "before.json")
    current = snapshot(True)
    require(current["pci"] == before["pci"] and current["boot_id"] == before["boot_id"]
            and current["kfd_holder_count"] == 0 and 0 <= current["monotonic_ns"] - before["monotonic_ns"] <= 45_000_000_000,
            "compute GPU ownership changed")
    port_available()
    hip = ctypes.CDLL("libamdhip64.so")
    n = ctypes.c_int()
    require(hip.hipGetDeviceCount(ctypes.byref(n)) == 0 and n.value == 8, "wrong HIP device count")
    pci = []
    for index in range(8):
        label = ctypes.create_string_buffer(32)
        require(hip.hipDeviceGetPCIBusId(label, len(label), index) == 0, "HIP PCI query failed")
        ident = label.value.decode().lower()
        pci.append("0000:" + ident if len(ident) == 7 else ident)
    require(sorted(pci) == current["pci"] and len(os.sched_getaffinity(0)) >= 64, "actual lease differs")
    save(RUN / "assigned.json", dict(current, visible_pci=pci, cpus=len(os.sched_getaffinity(0))))
    epoch=read(RUN/"intent.json")["epoch"]
    deadline=min(epoch+PACKAGE_SECONDS-150,time.time()+WORKER_SECONDS)
    for arm in ("continuous",):
        require(time.time()<deadline-30,"insufficient shared native budget for next arm")
        if run_arm(arm,deadline)!=0:
            return 1
        for number in ((10,20,25) if arm=="continuous" else (25,)):
            require(time.time()<deadline,"shared native epoch exhausted before checkpoint inventory")
            checkpoint_inventory(arm,number)
    return run_metric(deadline)


def controller():
    expected = read(RUN / "intent.json")
    require(RUN.name == LABEL and RUN.parent == RESULTS, "wrong run root")
    budget = int(expected["epoch"] + PACKAGE_SECONDS - time.time())
    require(0 < budget <= PACKAGE_SECONDS, "expired original launch epoch")
    def expired(*_):
        raise TimeoutError("original host controller deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(budget)
    admitted = False
    stages = {}
    failure = None
    try:
        admit()
        prepare()
        admit()
        shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
                 "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH=" +
                 shlex.quote("/vision:/minicpm-transformers:/swift-runtime/site:" + str(FRAMEWORK) + ":" + str(RUN / "src")) +
                 "; exec python3 -B -u " + shlex.quote(str(Path(__file__).resolve())) + " worker")
        runtime = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
        for source, target, mode in (
                (RUN, RUN, "rw"), (BASE / "source", BASE / "source", "ro"), (MODEL, MODEL, "ro"),
                (RESULTS / "training-qwen-initial-trainables-v1", RESULTS / "training-qwen-initial-trainables-v1", "ro"),
                (RESULTS / "training-vlm-original-assets-v1", RESULTS / "training-vlm-original-assets-v1", "ro"),
                (RESULTS / "training-vlm-swift-offline-runtime-v1", "/swift-runtime", "ro"),
                (RESULTS / "training-qwen-vision014-CPU-v2/vision", "/vision", "ro"),
                (RESULTS / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers", "ro"),
                (DATA / 'data', DATA / 'data', 'ro'), (Path(DEV_TARGET),Path(DEV_TARGET),'ro'), ("/opt/hyhal", "/opt/hyhal", "ro")):
            runtime.extend(["--bind", f"{source}:{target}:{mode}"])
        runtime.extend([str(REFACTOR / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell])
        cpu_runtime=list(runtime)
        cpu_runtime.remove('--rocm')
        for hidden in (f'{MODEL}:{MODEL}:ro',
                       f"{RESULTS/'training-qwen-initial-trainables-v1'}:{RESULTS/'training-qwen-initial-trainables-v1'}:ro"):
            require(cpu_runtime.count(hidden)==1,'exact CPU-only no-weight bind anchor differs')
            position=cpu_runtime.index(hidden)
            require(cpu_runtime[position-1]=='--bind','CPU bind flag differs')
            del cpu_runtime[position-1:position+1]
        cpu_shell=shell.rsplit(' worker',1)[0]+' native72'
        cpu_shell=cpu_shell.replace('exec python3',
            'export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; exec python3')
        cpu_runtime[-1]=cpu_shell
        stages['native72']=step('native72',['env','CUDA_VISIBLE_DEVICES=-1','HIP_VISIBLE_DEVICES=-1',
            'ROCR_VISIBLE_DEVICES=-1',*cpu_runtime],False,285)
        require(stages['native72']['passed'],'new native72 CPU qualification failed; no GPU invocation permitted')
        prepare_plan()
        admit()
        snapshot_cmd = [HOST_PYTHON, "-B", str(Path(__file__).resolve()), "snapshot"]
        stages["pre"] = step("pre", snapshot_cmd, False, 60)
        require(stages["pre"]["passed"], "original pre-snapshot failed")
        before = json.loads((RUN / "pre.log").read_bytes())
        save(RUN / "before.json", before)
        admitted = True
        stages["gpu"] = step("gpu", runtime, True, 1800)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        failure = type(error).__name__ + ": " + str(error)[:280]
    if (admitted and time.time() < expected["epoch"] + PACKAGE_SECONDS - 6
            and is_terminal(accounting(LABEL + "-gpu"))):
        stages["post"] = step("post", snapshot_cmd, False, 75)
        if stages["post"]["passed"]:
            after = json.loads((RUN / "post.log").read_bytes())
            require(after["pci"] == before["pci"] and after["boot_id"] == before["boot_id"],
                    "post snapshot identities differ")
            save(RUN / "after.json", after)
    passed = bool(failure is None and all(stages.get(k, {}).get("passed") for k in ("native72", "pre", "gpu", "post"))
                  and all((RUN/a/"native-outcome.json").exists() and read(RUN/a/"native-outcome.json")["passed"] for a in ("continuous","effect")))
    save(RUN / "controller-outcome.json", dict(stages=stages, failure=failure, passed=passed, finished=time.time(),planned_optimizer_invocations=25,planned_updates_by_arm={"continuous":25},unique_train_examples=72,physical_exposures=800,formal_acceptance=False))
    return 0 if passed else 1


if __name__ == "__main__":
    entry = {"host": controller, "worker": worker, "snapshot": cpu_snapshot, "native72": cpu_native}[sys.argv[1]]
    raise SystemExit(entry())
