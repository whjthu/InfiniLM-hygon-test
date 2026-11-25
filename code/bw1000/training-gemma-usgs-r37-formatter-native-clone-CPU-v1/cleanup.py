"""Release only this probe's provider handles and exact private scratch."""

import errno
import gc
import os
from pathlib import Path
import shutil
import stat
import sys
import time


def provider_children():
    children={}
    for name in ("multiprocessing","multiprocess"):
        provider=sys.modules.get(name)
        if provider is not None:
            for handle in provider.active_children():
                if type(handle.pid) is int and handle.pid>0:
                    children.setdefault(handle.pid,handle)
    return children


def owned_children(baseline,step,process,check,report):
    report.update(joined=0,terminated=0,still_live=0,unproven=0)
    owner=process(os.getpid(),step)
    owned={pid:h for pid,h in provider_children().items() if pid not in baseline}
    for pid,handle in owned.items():
        check()
        actor=process(pid,step)
        valid=(owner and actor and owner["same_slurm_step"] and actor["same_slurm_step"]
               and actor["ppid"]==os.getpid() and actor["uid"]==owner["uid"] and actor["namespace_inode"]==owner["namespace_inode"])
        if not valid:
            report["unproven"]+=1
            continue
        handle.join(timeout=.2)
        if handle.is_alive():
            check()
            # Recheck exact identity after joining. Termination is restricted to
            # a same-step, same-UID child from this request's actual provider.
            if process(pid,step)!=actor:
                report["unproven"]+=1
                continue
            handle.terminate()
            report["terminated"]+=1
            handle.join(timeout=.4)
        if handle.is_alive():
            report["still_live"]+=1
        else:
            handle.close()
            report["joined"]+=1
    return not(report["still_live"] or report["unproven"])


def contained(value,temp):
    path=Path(value.removesuffix(" (deleted)"))
    return path.is_absolute() and path.is_relative_to(temp)


def references(temp):
    info={"state":"OBSERVED","mapped_regions":0,"open_FDs":0}
    try:
        with Path("/proc/self/maps").open("r",encoding="utf-8",errors="surrogateescape") as raw:
            for n,line in enumerate(raw):
                if n>=32768:
                    raise ValueError("PROC_MAPS_BOUND")
                fields=line.rstrip("\n").split(maxsplit=5)
                if len(fields)==6 and contained(fields[-1],temp):
                    info["mapped_regions"]+=1
        for n,entry in enumerate(Path("/proc/self/fd").iterdir()):
            if n>=8192:
                raise ValueError("PROC_FD_BOUND")
            try:
                if contained(os.readlink(entry),temp):
                    info["open_FDs"]+=1
            except FileNotFoundError:
                continue
    except (OSError,ValueError):
        info["state"]="UNAVAILABLE"
    return info


def failure_symbol(error):
    number=getattr(error,"errno",None)
    return errno.errorcode.get(number,"ERROR_WITHOUT_POSIX_ERRNO")


def remove_exact(temp,scratch,baseline,dataset_refs,step,process,check,report):
    report.update(closed_python_handles=True,gc_collected=gc.collect(),children={},attempts=[],removed=False)
    check()
    if (temp is None or temp.parent!=scratch or not temp.name.startswith("native-") or scratch.resolve(strict=True)!=scratch
        or temp.resolve(strict=True)!=temp or not stat.S_ISDIR(temp.lstat().st_mode)):
        report["state"]="EXACT_PRIVATE_SCRATCH_IDENTITY_UNPROVEN"
        return
    if not owned_children(baseline,step,process,check,report["children"]):
        report["state"]="OWNED_NATIVE_CHILDREN_NOT_RELEASED"
        return
    info=references(temp)
    report["dataset_objects_still_live"]=sum(ref() is not None for ref in dataset_refs)
    report["scratch_mmap_and_FDs"]=info
    if report["dataset_objects_still_live"] or info["state"]!="OBSERVED" or info["mapped_regions"] or info["open_FDs"]:
        report["state"]="SCRATCH_MMAP_OR_FD_RELEASE_UNPROVEN"
        return
    for attempt in range(2):
        check()
        if not os.path.lexists(temp):
            break
        try:
            shutil.rmtree(temp)
            report["attempts"].append("REMOVED")
            break
        except OSError as error:
            report["attempts"].append(failure_symbol(error))
            if error.errno!=errno.ENOTEMPTY or attempt:
                break
            check()
            time.sleep(.15)
    report["removed"]=not os.path.lexists(temp) and not any(scratch.iterdir())
    report["state"]="COMPLETE" if report["removed"] else "REQUEST_SCRATCH_REMAINS"
