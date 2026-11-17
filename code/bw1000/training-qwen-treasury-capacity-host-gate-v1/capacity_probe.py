"""Metadata only: selected filesystem capacity/inodes/cache paths; no writes or native imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014"
ENV_PATHS = ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR",
             "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR", "MIOPEN_CACHE_DIR", "AMD_COMGR_CACHE_DIR",
             "LLVM_PROFILE_FILE", "APPTAINER_TMPDIR", "APPTAINER_CACHEDIR")
CONFIGS = ("/etc/apptainer/apptainer.conf", "/public/software/apps/apptainer/1.3.4/etc/apptainer/apptainer.conf")


def safe_path(value):
    return isinstance(value,str) and 0<len(value)<=1024 and value.startswith("/") and "\x00" not in value and "://" not in value and "@" not in value and re.search(r"(?i)(?:password|token|secret|key)\s*[:=]",value) is None


def unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m:chr(int(m.group(1),8)),value)


def mounts():
    p=Path("/proc/self/mountinfo")
    if not p.is_file() or p.stat().st_size>1024**2:
        return [],dict(error="MOUNTINFO_UNAVAILABLE")
    b=p.read_bytes()
    if len(b)>1024**2:
        return [],dict(error="MOUNTINFO_OVERSIZED")
    rows=[]
    for index,line in enumerate(b.decode("utf8",errors="replace").splitlines()):
        if index>=4096:
            return rows,dict(SHA256=hashlib.sha256(b).hexdigest(),truncated=True)
        left,sep,right=line.partition(" - ")
        a,z=left.split(),right.split()
        if not sep or len(a)<6 or len(z)<2:
            continue
        if not re.fullmatch(r"[0-9]+:[0-9]+",a[2]):
            continue
        rows.append(dict(mount_point=unescape(a[4]),fs_type=z[0],major_minor=a[2]))
    return rows,dict(SHA256=hashlib.sha256(b).hexdigest(),truncated=False)


def inspect_path(requested, mount_rows):
    result=dict(requested=requested,canary_written=False)
    if not safe_path(requested):
        return dict(result,error="INVALID_NONABSOLUTE_OR_NONPATH_VALUE")
    try:
        p=Path(requested)
        result["requested_exists"]=p.exists()
        resolved=p.resolve(strict=False)
        result["resolved_path"]=str(resolved)
        current=resolved
        for _ in range(32):
            if current.exists():
                break
            if current==current.parent:
                raise ValueError("no ancestor")
            current=current.parent
        else:
            raise ValueError("ancestor bound")
        v=os.statvfs(current);block=v.f_frsize or v.f_bsize
        matches=[x for x in mount_rows if str(current)==x["mount_point"] or str(current).startswith(x["mount_point"].rstrip("/")+"/")]
        mount=max(matches,key=lambda x:len(x["mount_point"])) if matches else None
        result.update(stat_path=str(current),used_existing_ancestor=str(current)!=str(resolved),mount=mount,
                      total_bytes=v.f_blocks*block,free_bytes=v.f_bfree*block,available_to_uid_bytes=v.f_bavail*block,
                      total_inodes=v.f_files,free_inodes=v.f_ffree,available_inodes=v.f_favail,
                      statvfs_flags=v.f_flag,fragment_bytes=block)
    except (OSError,RuntimeError,ValueError) as error:
        result["error_class"]=type(error).__name__
    return result


def config_metadata():
    results=[]
    for name in CONFIGS:
        p=Path(name);record=dict(path=name,exists=p.exists(),active_config_proved=False)
        try:
            if p.exists():
                info=p.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size>256*1024:
                    record["error"]="LINKED_OR_OVERSIZED_CONFIG"
                else:
                    b=p.read_bytes();record.update(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
                    lines=[line.strip() for line in b.decode("utf8",errors="replace").splitlines()
                           if re.fullmatch(r"\s*sessiondir max size\s*=\s*[0-9]+\s*",line)]
                    record["sessiondir_max_size_values"]=[int(line.rsplit("=",1)[1]) for line in lines]
                    record["configuration_units_independently_verified"]=False
        except OSError as error:
            record["error_class"]=type(error).__name__
        results.append(record)
    return results


def probe(arm, context, deadline):
    start=time.monotonic()
    if "torch" in sys.modules or "ctypes" in sys.modules:
        raise ValueError("unexpected native-capable import before metadata probe")
    env={}
    candidates=[("fixed",p) for p in ("/","/tmp","/dev/shm",ROOT+"/results/training-qwen-treasury-explicit-dev3-gpu-v3/arms/"+arm,
                 ROOT+"/results/training-qwen-treasury-capacity-host-gate-v1/arms/"+arm)]
    for key in ENV_PATHS:
        value=os.environ.get(key)
        if value is None:
            env[key]=dict(set=False)
        elif safe_path(value):
            env[key]=dict(set=True,path=value);candidates.append(("environment:"+key,value))
        else:
            env[key]=dict(set=True,path_rejected=True,value_SHA256=hashlib.sha256(value.encode()).hexdigest())
    home=os.environ.get("HOME")
    if safe_path(home):
        for suffix in (".cache",".cache/torch_extensions",".triton/cache",".cache/miopen"):
            candidates.append(("CANDIDATE_DEFAULT_NOT_LIBRARY_CONFIRMED",str(Path(home)/suffix)))
    mount_rows,mountinfo=mounts()
    paths=[];seen=set()
    for basis,path in candidates:
        if time.time()>=deadline:
            raise TimeoutError("original metadata probe deadline expired")
        if path in seen:
            continue
        seen.add(path);item=inspect_path(path,mount_rows);item["basis"]=basis;paths.append(item)
    df_paths=sorted({x["stat_path"] for x in paths if "stat_path" in x})
    df=dict(command=["/usr/bin/df","-i","-P","--",*df_paths],actual_paths=len(df_paths))
    try:
        remaining=min(3,deadline-time.time())
        if remaining<=0:
            raise TimeoutError("no original df budget")
        command=subprocess.run(df["command"],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=remaining,check=False)
        df.update(rc=command.returncode,stdout_SHA256=hashlib.sha256(command.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(command.stderr).hexdigest(),stderr_bytes=len(command.stderr))
        entries=[]
        for line in command.stdout.decode("utf8",errors="replace").splitlines()[1:]:
            parts=line.split()
            if len(parts)>=6:
                entries.append(dict(total_inodes=parts[1],used_inodes=parts[2],free_inodes=parts[3],used_percent=parts[4],mount_point=" ".join(parts[5:])))
        df["entries"]=entries
    except (OSError,TimeoutError,subprocess.SubprocessError) as error:
        df["error_class"]=type(error).__name__
    return dict(kind="training_qwen_capacity_metadata",arm=arm,context=context,completed=True,wall_seconds=time.monotonic()-start,
        node=os.uname().nodename.split(".")[0],uid=os.getuid(),euid=os.geteuid(),gid=os.getgid(),cwd=os.getcwd(),sys_flags=repr(sys.flags),
        environment_path_whitelist=env,paths=paths,mountinfo=mountinfo,df_inodes=df,apptainer_config_metadata=config_metadata(),
        Torch_imported=False,SDK_explicit_calls=0,LLVM_explicit_calls=0,model_or_weight_reads=0,canary_written=False,
        default_cache_paths_are_candidates_only=True,exact_ENOSPC_writer_path_proved=False,GPU_GO=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arm",required=True,choices=("base","old_domain","domain"));p.add_argument("--context",required=True,choices=("host","container"));p.add_argument("--deadline",required=True,type=float)
    args=p.parse_args();print(json.dumps(probe(args.arm,args.context,args.deadline),allow_nan=False))
