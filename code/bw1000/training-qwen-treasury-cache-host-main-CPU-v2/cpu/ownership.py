"""Read-only current-server PCI/KFD/same-UID FD ownership; no cmdline/environ/native calls."""
import os
from pathlib import Path
import re
import socket
import time

PROC=Path("/proc")
DRM=Path("/sys/class/drm")
KFD=Path("/sys/class/kfd/kfd/proc")
MAX_PIDS=8192
MAX_FDS=65536
PCI=re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")


def require(ok,reason):
    if not ok:raise ValueError(reason)


def process_identity(pid):
    path=PROC/str(pid)
    owner=path.stat().st_uid
    raw=(path/"stat").read_text()
    suffix=raw.rsplit(") ",1)[1].split()
    # The comm field is ignored and never returned; only state/PGID/startticks identity.
    return dict(PID=int(pid),UID=owner,state=suffix[0],PGID=int(suffix[2]),startticks=int(suffix[19]))


def validate_snapshot(value,parent,node):
    require(value["complete_inventory"] is True and not value["errors"],"ownership inventory incomplete/unknown")
    require(value["node"]==node and value["step"].startswith(parent+".") and re.fullmatch(re.escape(parent)+r"\.[0-9]+",value["step"]),"CPUstep/node identity differs")
    require(type(value["UID"]) is int and value["stage_process"]["UID"]==value["UID"]
            and value["stage_process"]["PID"]>0 and value["stage_process"]["startticks"]>0,"stage PID/UID identity differs")
    pci=value["pci"]
    require(isinstance(pci,list) and len(pci)==len(set(pci))==8 and pci==sorted(pci)
            and all(PCI.fullmatch(p) for p in pci),"current eight distinct PCI differs")
    require(value["kernel_KFD_holders"]==[] and value["same_UID_device_FD_holders"]==[]
            and value["KFD_holder_count"]==0 and value["same_UID_FD_inventory_complete"] is True,"current device holder remains or unknown")
    require(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",value["boot_id"])
            and type(value["monotonic_ns"]) is int and value["monotonic_ns"]>0,"boot/clock anchor differs")
    return True


def capture(parent,node,deadline):
    value=dict(kind="current_server_readonly_ownership",node=socket.gethostname().split(".")[0],
       step=os.environ.get("SLURM_JOB_ID","")+"."+os.environ.get("SLURM_STEP_ID",""),UID=os.getuid(),
       observed_epoch=time.time(),monotonic_ns=time.monotonic_ns(),complete_inventory=False,
       same_UID_FD_inventory_complete=False,pci=None,boot_id=None,stage_process=None,
       kernel_KFD_holders=None,same_UID_device_FD_holders=None,KFD_holder_count=None,
       same_UID_processes_checked=0,FD_entries_checked=0,process_exit_races=0,errors=[],
       proc_cmdline_or_environ_read=False,busy_qualification=False)
    def budget():require(time.time()<deadline,"original stage ownership budget expired")
    try:
        budget();require(value["node"]==node and os.environ.get("SLURM_JOB_ID")==parent,"wrong current server")
        value["stage_process"]=process_identity(os.getpid())
        value["boot_id"]=(PROC/"sys/kernel/random/boot_id").read_text().strip()
        pci=[]
        for entry in DRM.iterdir():
            budget()
            if re.fullmatch(r"renderD[0-9]+",entry.name):
                identity=Path(os.readlink(entry/"device")).name.lower();require(PCI.fullmatch(identity),"DRM PCI identity malformed");pci.append(identity)
        value["pci"]=sorted(pci)
        holders=[]
        for entry in KFD.iterdir():
            budget();require(entry.name.isdecimal(),"KFD process schema malformed");holders.append(int(entry.name))
        value["kernel_KFD_holders"]=sorted(holders)
        same_uid=[];seen=0;FD_total=0
        for process in PROC.iterdir():
            budget()
            if not process.name.isdecimal():continue
            seen+=1;require(seen<=MAX_PIDS,"bounded PID inventory exceeded")
            try:
                if process.stat().st_uid!=value["UID"]:continue
                before=process_identity(process.name);require(before["UID"]==value["UID"],"process owner changed")
                held=False
                for fd in (process/"fd").iterdir():
                    budget();FD_total+=1;value["FD_entries_checked"]=FD_total;require(FD_total<=MAX_FDS,"bounded sameUID FD inventory exceeded")
                    try:target=os.readlink(fd)
                    except FileNotFoundError:continue
                    if target=="/dev/kfd" or target.startswith("/dev/dri/"):held=True
                after=process_identity(process.name)
                require((before["UID"],before["startticks"])==(after["UID"],after["startticks"]),"PID identity changed during FD inventory")
                value["same_UID_processes_checked"]+=1
                if held:same_uid.append({k:before[k] for k in ("PID","UID","startticks","PGID")})
            except FileNotFoundError:
                value["process_exit_races"]+=1
                continue
        value["FD_entries_checked"]=FD_total
        value["same_UID_device_FD_holders"]=same_uid
        value["KFD_holder_count"]=len(set(holders)|{p["PID"] for p in same_uid})
        value["same_UID_FD_inventory_complete"]=True;value["complete_inventory"]=True
        validate_snapshot(value,parent,node)
        value["qualified"]=True
    except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
        value["qualified"]=False;value["complete_inventory"]=False;value["errors"].append(type(error).__name__)
    return value


def container_exit_identity(handle,deadline):
    if time.time()>=deadline:return dict(qualified=False,original_stage_deadline_expired=True)
    if handle is None:return dict(qualified=True,container_not_started=True)
    if "startticks" not in handle:return dict(qualified=False,exited_before_startticks_observation=bool(handle.get("exited_before_identity_observation")),identity_unknown=True)
    try:
        actual=process_identity(handle["PID"])
    except FileNotFoundError:return dict(qualified=True,original_owned_PID_absent=True)
    except (OSError,ValueError,IndexError) as error:return dict(qualified=False,error_class=type(error).__name__)
    same=actual["startticks"]==handle["startticks"]
    return dict(qualified=not same or actual["state"]=="Z",original_owned_PID_live=bool(same and actual["state"]!="Z"),
                observed_PID=actual["PID"],observed_UID=actual["UID"],observed_startticks=actual["startticks"],observed_state=actual["state"])


def validate_pair(before,after,parent,node,container_exit):
    validate_snapshot(before,parent,node);validate_snapshot(after,parent,node)
    require(before["step"]==after["step"] and before["boot_id"]==after["boot_id"] and before["pci"]==after["pci"],"pre/post CPUstep/boot/PCI changed")
    require(before["stage_process"]["PID"]==after["stage_process"]["PID"] and before["stage_process"]["startticks"]==after["stage_process"]["startticks"]
            and before["UID"]==after["UID"] and after["monotonic_ns"]>=before["monotonic_ns"],"pre/post stage process/clock changed")
    require(container_exit["qualified"] is True,"owned container PID still live or unknown")
    return True
