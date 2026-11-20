"""Default-OFF cache setup and tiny controlled write checks, only inside a new bound RUN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-cache-host-main-candidate-v2")
CACHE_DIRS={"TMPDIR":"tmp","XDG_CACHE_HOME":"xdg","TORCH_EXTENSIONS_DIR":"torch_extensions","TRITON_CACHE_DIR":"triton",
            "TORCHINDUCTOR_CACHE_DIR":"torchinductor","AMD_COMGR_CACHE_DIR":"comgr","MIOPEN_USER_DB_PATH":"miopen",
            "HOME":"home","HF_HOME":"hf_home","HF_MODULES_CACHE":"hf_modules"}


def require(ok,why):
    if not ok:raise ValueError(why)


def prepare(arm,enabled):
    require(enabled is True and arm in ("base","old_domain","domain"),"explicit HOST cache optin required")
    os.umask(0o077)
    run=ROOT/"arms"/arm
    require(run.is_dir() and not run.is_symlink() and run.resolve(strict=True)==run,"owned bound RUN missing or linked")
    require(not (run/"cache").exists() and not (run/"owned-cache-readiness.json").exists() and not (run/"host-main").exists(),"existing owned namespace needs reconciliation; no replay")
    cache=run/"cache";cache.mkdir(mode=0o700)
    payload=(b"owned-cache-host-main-write-check\n"*128)[:4096]
    # Fixed <=4KiB per directory, <=40KiB total. This does not prove userquota/JIT capacity.
    records={}
    for key,name in CACHE_DIRS.items():
        path=cache/name;path.mkdir(mode=0o700)
        require(Path(os.environ.get(key,""))==path,"cache ENV must already be set by child shell")
        target=path/".owned-cache-canary-v1"
        with target.open("x+b") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno());identity=os.fstat(stream.fileno())
            stream.seek(0);actual=stream.read()
            require(actual==payload,"owned cache write mismatch")
        now=target.lstat();require(stat.S_ISREG(now.st_mode) and (now.st_dev,now.st_ino)==(identity.st_dev,identity.st_ino),"owned canary identity changed")
        target.unlink()  # Only this just-created exact owned canary; never sweep/cache cleanup.
        info=os.statvfs(path)
        records[key]=dict(path=str(path),bytes_written=len(payload),SHA256=hashlib.sha256(actual).hexdigest(),own_canary_removed=True,
                          available_to_uid_bytes=info.f_bavail*(info.f_frsize or info.f_bsize),free_inodes=info.f_favail)
    (run/"host-main").mkdir(mode=0o700)
    receipt=dict(arm=arm,all_cache_canaries_verified=True,total_canary_bytes=len(payload)*len(records),records=records,
                 HOME_HF_HOME_HF_MODULES_CACHE_created_empty_owned=True,original_home_or_credentials_copied=False,
                 tested_bytes_only=True,user_quota_and_arbitrary_JIT_capacity_qualified=False,shared_tmp_or_global_config_changed=False,GPU_GO=False)
    with (run/"owned-cache-readiness.json").open("x",encoding="utf8") as stream:json.dump(receipt,stream,indent=2);stream.write("\n")
    print(json.dumps(dict(arm=arm,prepared=True,total_canary_bytes=receipt["total_canary_bytes"],GPU_GO=False)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arm",required=True,choices=("base","old_domain","domain"));p.add_argument("--host-main-preflight",action="store_true",default=False)
    a=p.parse_args();prepare(a.arm,a.host_main_preflight)
