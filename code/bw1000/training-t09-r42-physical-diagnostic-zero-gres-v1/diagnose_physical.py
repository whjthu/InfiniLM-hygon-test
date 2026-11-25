"""Classify the original T09 physical gate's read-only failure."""

import argparse
import errno
import json
import os
from pathlib import Path
import re
import socket
import sys
import time


PARENT="1103606"
NODE="f13r3n13"
DRM=Path("/sys/class/drm")
KFD=Path("/sys/class/kfd/kfd")
PROC=Path("/proc")
PORT=29642
PCI=re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")


def need(test):
    if not test:
        raise ValueError("UNEXPECTED_SHAPE")


class Classifier:
    def __init__(self):
        self.current=None
        self.results={}

    def operation(self,name,call,*args,**kwargs):
        self.current=name
        return call(*args,**kwargs)

    def phase(self, name, task):
        self.current=name
        try:
            value=task()
        except (OSError,ValueError,KeyError,RuntimeError) as error:
            number=getattr(error,"errno",None)
            self.results[name]=dict(passed=False,operation=self.current,error_class=type(error).__name__,
                                    errno=number,errno_name=errno.errorcode.get(number))
            return None
        self.results[name]=dict(passed=True)
        return value


def bounded(reader,limit):
    result=[]
    for row in reader:
        need(len(result)<limit)
        result.append(row)
    return result


def pci_from_render(c,render,category):
    link=c.operation(category,os.readlink,DRM/render/"device")
    name=Path(link).name.lower()
    need(PCI.fullmatch(name))
    return name


def cgroup(c):
    numeric=os.environ.get("SLURM_STEP_ID","")
    own=PARENT+"."+numeric
    need(os.environ.get("SLURM_JOB_ID")==PARENT and re.fullmatch(r"[0-9]+",numeric)
         and own not in {"1103606.1611","1103606.1612","1103606.1614","1103606.3452"}
         and socket.gethostname().split(".")[0]==NODE and os.environ.get("SLURM_CPUS_PER_TASK")=="1"
         and len(os.sched_getaffinity(0))>=1 and os.environ.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
         and all(os.environ.get(k,"0") in ("","0","(null)") for k in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
         and all(os.environ.get(k)=="-1" for k in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")))
    rows=c.operation("SELF_CGROUP_READ",(PROC/"self/cgroup").read_text,encoding="ascii").splitlines()
    need(any({"job_"+PARENT,"step_"+numeric}<=set(row.split(":",2)[-1].split("/")) for row in rows))
    return own


def drm(c):
    files=c.operation("DRM_SYSFS_ENUMERATE",lambda:bounded(DRM.iterdir(),256))
    cards=sorted(pci_from_render(c,p.name,"DRM_RENDER_DEVICE_READLINK") for p in files
                 if re.fullmatch(r"renderD[0-9]+",p.name))
    need(len(cards)==len(set(cards))==8)
    return cards


def kfd_processes(c):
    paths=c.operation("KFD_PROC_SYSFS_ENUMERATE",lambda:bounded((KFD/"proc").iterdir(),8192))
    holders={entry.name for entry in paths}
    need(all(value.isdecimal() for value in holders))
    c.results["global_kfd_holder_count"]=len(holders)
    return len(holders)


def same_uid_fds(c):
    uid=os.getuid()
    procs=c.operation("PROC_ROOT_ENUMERATE",lambda:bounded(PROC.iterdir(),65536))
    counts={"same_uid_checked":0,"descriptors_readlink_attempted":0,"disappeared":0}
    c.results["same_uid_fd_progress"]=counts
    holders=set()
    for proc in procs:
        if not proc.name.isdecimal():
            continue
        try:
            info=c.operation("PROC_PID_UID_STAT",proc.stat)
            if info.st_uid!=uid:
                continue
            counts["same_uid_checked"]+=1
            entries=c.operation("SAME_UID_FD_ENUMERATE",lambda:bounded((proc/"fd").iterdir(),16384))
            for fd in entries:
                try:
                    counts["descriptors_readlink_attempted"]+=1
                    target=c.operation("SAME_UID_FD_READLINK",os.readlink,fd)
                    if target=="/dev/kfd" or target.startswith("/dev/dri/"):
                        holders.add(proc.name)
                except FileNotFoundError:
                    counts["disappeared"]+=1
        except FileNotFoundError:
            counts["disappeared"]+=1
    return len(holders)


def kfd_topology(c):
    nodes=c.operation("KFD_TOPOLOGY_ENUMERATE",lambda:bounded((KFD/"topology/nodes").iterdir(),256))
    cards={}
    for node in nodes:
        gpu_id=int(c.operation("KFD_TOPOLOGY_GPU_ID_READ",(node/"gpu_id").read_text))
        if gpu_id==0:
            continue
        raw=c.operation("KFD_TOPOLOGY_PROPERTIES_READ",(node/"properties").read_text)
        props=dict(row.split(maxsplit=1) for row in raw.splitlines())
        address=pci_from_render(c,"renderD"+str(int(props["drm_render_minor"])),"KFD_DR_MAPPED_DEVICE_READLINK")
        domain,bus,device=address.split(":")
        slot,function=device.split(".")
        location=int(bus,16)<<8|int(slot,16)<<3|int(function,16)
        need(int(domain,16)==int(props["domain"]) and location==int(props["location_id"]))
        need(address not in cards and gpu_id>0 and gpu_id not in cards.values())
        cards[address]=gpu_id
    need(len(cards)==8)
    return sorted(cards)


def loopback(c):
    with c.operation("LOOPBACK_SOCKET_CREATE",socket.socket,socket.AF_INET,socket.SOCK_STREAM) as listener:
        c.operation("LOOPBACK_PORT_BIND",listener.bind,("127.0.0.1",PORT))
    return True


def run(request,epoch):
    need(re.fullmatch(r"[0-9a-f]{32}",request) and 0<=time.time()-epoch<90)
    need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
    c=Classifier()
    result=dict(request=request,epoch=epoch,diagnostic_only=True,prior_3452_replayed=False,
                target_GPU=0,optimizer_updates=0,phases=c.results)
    step=c.phase("NUMERIC_CPU_STEP",lambda:cgroup(c))
    if step is None:
        return result
    result.update(step=step,node=NODE)
    cards=c.phase("DRM",lambda:drm(c))
    kfd=c.phase("KFD_PROC",lambda:kfd_processes(c))
    fds=c.phase("SAME_UID_FDS",lambda:same_uid_fds(c))
    topology=c.phase("KFD_TOPOLOGY",lambda:kfd_topology(c))
    port=c.phase("LOOPBACK",lambda:loopback(c))
    c.results["DRM"]["board_count"]=len(cards) if cards is not None else None
    c.results["KFD_PROC"]["holder_count"]=kfd
    c.results["SAME_UID_FDS"]["holder_count"]=fds
    c.results["KFD_TOPOLOGY"]["board_count"]=len(topology) if topology is not None else None
    c.results["LOOPBACK"]["port_available"]=port
    result["strict_read_conditions_observed"]=(cards is not None and cards==topology and kfd==0 and fds==0
                                               and port is True and time.time()<epoch+90)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch",type=float,required=True)
    p.add_argument("--request",required=True)
    opts=p.parse_args()
    try:
        result=run(opts.request,opts.epoch)
    except (OSError,ValueError) as e:
        result={"diagnostic_only":True,"error_class":type(e).__name__,"strict_read_conditions_observed":False}
    print(json.dumps(result,sort_keys=True,separators=(",",":")),flush=True)
    raise SystemExit(0 if result.get("strict_read_conditions_observed") is True else 1)
