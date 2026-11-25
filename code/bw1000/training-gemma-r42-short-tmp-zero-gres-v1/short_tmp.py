"""Exercise the deployed Dataset.map Manager using a short private mount."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import sys
import tempfile
import time


ALIAS = Path("/tmp/g42")
NODE = "f13r1n12"
PARENT = "1103606"
ARROW_SHA = "1fe17db3c76d1fac5472b305c538e26cc11e1e995f667bdd6868ac60596dcde1"
POOL_SHA = "286b1899798e5c3dd01d131ddda151617a63e3b6e24d2470b73514f9dd259bc5"
SITE = Path("/swift-runtime/site")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def exact_mount(alias, inode, scope):
    info = alias.lstat()
    need(alias.resolve(strict=True) == alias and stat.S_ISDIR(info.st_mode)
         and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700
         and info.st_ino == inode, "G42_EXACT_PRIVATE_SHORT_MOUNT")
    need({member.name for member in alias.iterdir()} == {".scope"}, "G42_INITIAL_EMPTY_SCOPE")
    marker = alias/".scope"
    m = marker.lstat()
    need(stat.S_ISREG(m.st_mode) and m.st_uid == os.getuid() and stat.S_IMODE(m.st_mode) == 0o600
         and m.st_size == 33 and marker.read_bytes() == scope.encode("ascii")+b"\n",
         "G42_EXACT_PRIVATE_SCOPE_MARKER")


def provider_file(path, expected):
    before = path.lstat()
    need(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
         and 0 < before.st_size <= 512*1024, "G42_EXACT_PROVIDER_FILE")
    with path.open("rb") as f:
        opened = os.fstat(f.fileno())
        raw = f.read(512*1024+1)
        after = os.fstat(f.fileno())
    bind = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(bind(before)==bind(opened)==bind(after)==bind(path.lstat()) and len(raw)==before.st_size,
         "G42_STABLE_PROVIDER_FILE")
    digest = sha256(raw).hexdigest()
    need(expected is None or digest == expected, "G42_EXACT_DEPLOYED_PROVIDER_SHA")
    return digest


def monitor(alias, log, phase):
    # Python audit hooks run just before socket operations. A connect event also
    # checks that the pathname has already become a real socket inode.
    def hook(event, args):
        if event not in ("socket.bind", "socket.connect"):
            return
        sock, address = args
        need(sock.family == socket.AF_UNIX and type(address) is str and "\x00" not in address,
             "G42_UNRELATED_SOCKET_REFUSED")
        target = Path(address)
        need(target.is_absolute() and target.is_relative_to(alias) and target != alias
             and ".." not in target.parts and len(os.fsencode(address)) <= 107,
             "G42_SOCKET_OUTSIDE_SHORT_PRIVATE_MOUNT")
        parent = target.parent
        need(parent.resolve(strict=True) == parent, "G42_SOCKET_PARENT_NOT_CANONICAL")
        for directory in (parent, *parent.parents):
            if not directory.is_relative_to(alias):
                break
            info = directory.lstat()
            need(info.st_uid==os.getuid() and stat.S_ISDIR(info.st_mode) and
                 stat.S_IMODE(info.st_mode)==0o700, "G42_SOCKET_PARENT_NOT_PRIVATE")
        bound = event == "socket.connect"
        if bound:
            info = target.lstat()
            need(stat.S_ISSOCK(info.st_mode) and info.st_uid==os.getuid(), "G42_SOCKET_INODE_MISSING")
        raw=(json.dumps({"phase":phase[0],"event":event,"bound_inode_at_connect":bound,
                          "pathname_bytes":len(os.fsencode(address))},separators=(",",":"))+"\n").encode()
        fd=os.open(log,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            need(os.fstat(fd).st_size < 48*1024, "G42_SOCKET_LOG_BOUND")
            need(os.write(fd,raw)==len(raw), "G42_SOCKET_LOG_WRITE")
        finally:
            os.close(fd)
    sys.addaudithook(hook)


def fixture_transform(row):
    return {"converted": row["ordinal"] + 10}


def step_identity(epoch):
    elapsed = time.time()-epoch
    need(math.isfinite(epoch) and 0 <= elapsed < 120, "G42_120S_EPOCH")
    env = os.environ
    numeric = env.get("SLURM_STEP_ID", "")
    need(env.get("SLURM_JOB_ID")==PARENT and re.fullmatch(r"[0-9]+",numeric)
         and socket.gethostname().split(".")[0]==NODE and env.get("SLURM_CPUS_PER_TASK")=="4"
         and len(os.sched_getaffinity(0)) >= 4, "G42_ACTUAL_CPU_STEP")
    groups = Path("/proc/self/cgroup").read_text("ascii").splitlines()
    need(any({"job_"+PARENT,"step_"+numeric} <= set(line.split(":",2)[-1].split("/")) for line in groups),
         "G42_EXACT_NUMERIC_STEP_CGROUP")
    need(env.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
         and all(env.get(key,"0") in ("","0","(null)") for key in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
         and not any(key in env.get("SLURM_TRES_PER_TASK","").lower() for key in ("gres/","gpu","dcu"))
         and all(env.get(key)=="-1" for key in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")),
         "G42_NO_STEP_GPU")
    step = PARENT+"."+numeric
    need(step not in ("1103606.3434","1103606.3435","1103606.3436"), "G42_NO_ORIGINAL_REPLAY")
    return step


def inspect_events(log):
    need(log.stat().st_size <= 48*1024, "G42_SOCKET_AUDIT_EXTENT")
    rows = [json.loads(line) for line in log.read_bytes().splitlines()]
    by_phase = {}
    for name in ("Dataset.map", "Provider Manager"):
        chosen = [row for row in rows if row["phase"]==name]
        by_phase[name] = {"binds":sum(row["event"]=="socket.bind" for row in chosen),
                          "connects_to_live_socket":sum(row["bound_inode_at_connect"] for row in chosen),
                          "maximum_path_bytes":max((row["pathname_bytes"] for row in chosen),default=None)}
    return by_phase


def exercise(epoch, scope, inode, result):
    need(re.fullmatch(r"[0-9a-f]{32}",scope) and inode>0, "G42_EXACT_NEW_CPU_SCOPE")
    result.update(step=step_identity(epoch), epoch=epoch)
    need(sys.dont_write_bytecode and sys.pycache_prefix is None
         and all(os.environ.get(name)==str(ALIAS) for name in ("TMPDIR","TMP","TEMP"))
         and tempfile.tempdir is None and tempfile.gettempdir()==str(ALIAS), "G42_SHORT_PATH_BEFORE_PROVIDERS")
    exact_mount(ALIAS, inode, scope)
    log = ALIAS/".ipc"
    fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    os.close(fd)
    phase = ["Dataset.map"]
    result["phase"] = phase[0]
    monitor(ALIAS, log, phase)
    mapped = source = None
    provider = None
    try:
        import datasets.arrow_dataset as arrow
        import datasets.utils.py_utils as pool
        import multiprocess as provider
        need(Path(arrow.__file__)==SITE/"datasets/arrow_dataset.py"
             and Path(pool.__file__)==SITE/"datasets/utils/py_utils.py", "G42_ACTUAL_DEPLOYED_DATASETS_PATH")
        result["arrow_SHA256"] = provider_file(Path(arrow.__file__),ARROW_SHA)
        result["pool_SHA256"] = provider_file(Path(pool.__file__),POOL_SHA)
        need(getattr(arrow,"Pool",None) is provider.Pool or
             any(value is provider for value in vars(arrow).values()), "G42_DATASETS_POOL_PROVIDER_BINDING")
        # Package source may live in the image or a mounted site. Record its
        # actual loaded bytes without using a guessed version or local module.
        result["actual_manager_provider_SHA256"] = provider_file(Path(provider.__file__),None)
        source = arrow.Dataset.from_dict({"ordinal": [0,1]})
        mapped = source.map(fixture_transform, num_proc=1, keep_in_memory=True, load_from_cache_file=False)
        need(list(mapped["converted"])==[10,11], "G42_NATIVE_DATASET_MAP_VALUES")
        result["actual_dataset_map_fixture_rows"] = 2
        if not inspect_events(log)["Dataset.map"]["connects_to_live_socket"]:
            phase[0] = result["phase"] = "Provider Manager"
            # Datasets iflatmap_unordered invokes pool._ctx.Manager(). The
            # same provider context receives an actual queue round trip.
            with provider.get_context().Manager() as manager:
                queue = manager.Queue()
                queue.put(42)
                need(queue.get(timeout=5)==42, "G42_REAL_PROVIDER_MANAGER_QUEUE")
                del queue
    finally:
        del mapped, source
        events=inspect_events(log)
        result["IPC"]=events
        if provider is not None:
            result["active_provider_children_after_exit"] = len(provider.active_children())
        log.unlink()
        result["private_mount_clean_after"] = {entry.name for entry in ALIAS.iterdir()}=={".scope"}
    need(any(row["binds"] and row["connects_to_live_socket"] for row in result["IPC"].values()),
         "G42_REAL_PRIVATE_SOCKET_LIFECYCLE")
    need(result["private_mount_clean_after"] is True and result.get("active_provider_children_after_exit")==0
         and time.time()<epoch+120, "G42_CLEANUP_AND_BUDGET")
    result["phase"]="COMPLETE"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch",required=True,type=float)
    parser.add_argument("--scope",required=True)
    parser.add_argument("--scratch-inode",required=True,type=int)
    args=parser.parse_args()
    result={"request":args.scope,"passed":False,"target_GPU":0,"native_optimizer_updates":0,
            "actual_training_rows_opened":0,"actual_weight_files_opened":0}
    def timeout(*_):
        raise TimeoutError("G42_120S_EPOCH")
    signal.signal(signal.SIGALRM,timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(120,args.epoch+120-time.time())))
        exercise(args.epoch,args.scope,args.scratch_inode,result)
        result["passed"]=True
    except Exception as error:
        code=str(error) if type(error) in (ValueError,TimeoutError) else ""
        result.update(error_class=type(error).__name__,
                      predicate=code if re.fullmatch(r"G42_[A-Z0-9_]+",code) else None,
                      errno=error.errno if isinstance(error,OSError) else None)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(result, sort_keys=True, separators=(",",":")),flush=True)
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
