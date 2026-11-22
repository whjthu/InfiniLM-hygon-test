"""Run exactly one reusable 16-owner model context cohort inside the GPU step."""

import base64
import ctypes as C
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time

from contract import (CONSTRUCT, HERE, INSIDE, LOAD, NODES, PHASES, PORTS, WORK, budget, need,
                      proc_start, put, read_json, step_identity, validate_claim, verify_stage)
from node import read_lease


class Probe:
    def __init__(self, device):
        self.lib = C.CDLL("/opt/dtk/cuda/cuda/lib64/libcudart.so.12")
        self.lib.cudaSetDevice.argtypes, self.lib.cudaSetDevice.restype = [C.c_int], C.c_int
        self.lib.cudaMemGetInfo.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
        self.lib.cudaMemGetInfo.restype = C.c_int
        self.device = device
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.errors = []
        self.minimum, self.count, self.last, self.max_gap = None, 0, None, 0.
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def memory(self, label):
        free, total = C.c_size_t(), C.c_size_t()
        need(self.lib.cudaMemGetInfo(C.byref(free),C.byref(total)) == 0 and total.value == 68702699520,
             "original device memory query: " + label)
        return free.value, total.value

    def sample(self):
        try:
            need(self.lib.cudaSetDevice(self.device) == 0, "sampler device assignment")
            while not self.stop.is_set():
                free,_ = self.memory("5ms driver sampler")
                now = time.monotonic()
                with self.lock:
                    self.minimum = free if self.minimum is None else min(free,self.minimum)
                    if self.last is not None:
                        self.max_gap = max(self.max_gap,now-self.last)
                    self.last = now
                    self.count += 1
                self.stop.wait(.005)
        except Exception as exc:
            with self.lock:
                self.errors.append(str(exc)[:160])

    def summary(self):
        with self.lock:
            return dict(samples=self.count,min_observed=self.minimum,errors=list(self.errors),
                        max_gap_s=self.max_gap,target_s=.005)

    def health(self):
        result = self.summary()
        need(not result["errors"] and (result["min_observed"] is None or result["min_observed"] >= 8*1024**3),
             "8GiB asynchronous driver sample failed")

    def close(self):
        self.stop.set()
        self.thread.join(2)
        need(not self.thread.is_alive(), "sampler did not stop")
        self.health()


class Frames:
    def __init__(self, epoch, aborted=lambda:False):
        from ready_socket_parent import canonical, receive_json, send_json
        self.epoch, self.aborted, self.canonical = epoch, aborted, canonical
        self._receive, self._send = receive_json, send_json

    def stop_requested(self):
        return self.aborted()

    def limit(self, phase):
        cap = CONSTRUCT if phase == PHASES[0] else LOAD if phase == PHASES[1] else WORK
        return time.monotonic()+budget(self.epoch,cap)

    def send(self,sock,record,phase):
        self._send(sock,record,self,lambda:False,self.limit(phase))

    def receive(self,sock,phase):
        return self._receive(sock,self,lambda:False,self.limit(phase))


def identity(intent):
    return dict(request=intent["claim"]["request"],label=intent["claim"]["label"],step=step_identity(),
                epoch=intent["epoch"],source=intent["claim"]["source_SHA256"],claim_SHA256=intent["claim_SHA256"])


def process(node):
    return dict(node=NODES[node],PID=os.getpid(),startticks=proc_start(os.getpid()),uid=os.getuid())


def verify_records(records,node,phase,run):
    need(type(records) is list and len(records) == 8, "complete eight-owner ready set required")
    for dev,row in enumerate(records):
        need(row["run"] == run and row["phase"] == phase and row["rank"] == node*8+dev
             and row["node"] == NODES[node] and row["device"] == dev and row["uid"] == os.getuid()
             and type(row["PID"]) is int and row["PID"] > 0 and type(row["startticks"]) is int and row["startticks"] > 0,
             "wrong owner identity, phase or run")
    need(len({r["PID"] for r in records}) == 8, "duplicate owner process")


def exchange(conn,frames,node,phase,run,records):
    verify_records(records,node,phase,run)
    common = dict(run=run,phase=phase)
    if node == 1:
        nonce = secrets.token_hex(32)
        frames.send(conn,dict(common,kind="challenge",nonce=nonce),phase)
        answer = frames.receive(conn,phase)
        remote = answer["records"]
        need(answer == dict(common,kind="records",nonce=nonce,records=remote), "foreign challenged ready answer")
        verify_records(remote,0,phase,run)
        joined = remote+records
        digest = hashlib.sha256(frames.canonical(joined)).hexdigest()
        frames.send(conn,dict(common,kind="verified",nonce=nonce,records=joined,digest=digest),phase)
        need(frames.receive(conn,phase) == dict(common,kind="ACK",nonce=nonce,digest=digest), "bad ready acknowledgement")
    else:
        challenge = frames.receive(conn,phase)
        nonce = challenge["nonce"]
        need(re.fullmatch("[0-9a-f]{64}",nonce) and challenge == dict(common,kind="challenge",nonce=nonce),
             "foreign peer challenge")
        frames.send(conn,dict(common,kind="records",nonce=nonce,records=records),phase)
        result = frames.receive(conn,phase)
        joined = result["records"]
        need(len(joined) == 16 and joined[:8] == records, "wrong complete ready denominator")
        verify_records(joined[8:],1,phase,run)
        digest = hashlib.sha256(frames.canonical(joined)).hexdigest()
        need(result == dict(common,kind="verified",nonce=nonce,records=joined,digest=digest), "wrong complete ready digest")
        frames.send(conn,dict(common,kind="ACK",nonce=nonce,digest=digest),phase)
    return digest


def actual_uids(out,run,owner,native,deadline):
    from bindings import unique_ids
    path = out/"uids.json"
    if owner["rank"] == 0:
        values = unique_ids(native)
        put(path,dict(run=run,owner=owner,ids=[base64.b64encode(v).decode("ascii") for v in values]))
    while True:
        deadline("original two communicator UIDs")
        try:
            value = read_json(path,16384)
            first = read_json(out/"leader-0.json")["spawned"][0]
        except (FileNotFoundError,json.JSONDecodeError,IndexError):
            time.sleep(.05)
            continue
        require_fields = ("rank","node","PID","startticks","uid")
        need(value["run"] == run and all(value["owner"][k] == first[k] for k in require_fields),
             "foreign communicator producer")
        world,tp = (base64.b64decode(v,validate=True) for v in value["ids"])
        size = int(native.api["id_size"]())
        need(0 < size <= 4096 and len(world) == len(tp) == size and world != tp, "communicator width or identity differs")
        return world,tp


def owner_main(out, node, dev, fd, intent, run):
    from bindings import load_native
    from run_pair import run_owner
    rank = node*8+dev
    channel = socket.socket(fileno=fd)
    owner = dict(process(node),rank=rank,device=dev)
    report_path = out/f"owner-{rank:02d}.json"
    sampler = None
    report = {"state":"OWNER_STARTED"}
    phases = iter(PHASES)

    def emit(item):
        nonlocal report
        report = item
        put(report_path,dict(identity=owner,run=run,pair=item),replace=True)

    def deadline(label):
        cap = WORK
        if label == "before construct" or label.startswith("original two"):
            cap = CONSTRUCT
        if label.startswith(("typed ","safe_open ")):
            cap = LOAD
        budget(intent["epoch"],cap)
        if sampler:
            sampler.health()

    frames = Frames(intent["epoch"])

    def gate(phase):
        need(next(phases) == phase, "local owner ready phase order changed")
        frames.send(channel,dict(owner,run=run,phase=phase),phase)
        answer = frames.receive(channel,phase)
        need(answer["run"] == run and answer["phase"] == phase and re.fullmatch("[0-9a-f]{64}",answer["digest"]),
             "foreign owner gate")

    def mark(phase, **data):
        budget(intent["epoch"],LOAD)
        if phase in {"load.progress","finish.return"}:
            put(out/f"load-{rank:02d}.json",dict(run=run,identity=owner,phase=phase,**data),replace=True)

    try:
        emit(report)
        leases = verify_lease_pins(out,intent,run)
        pci_map = read_json(out/f"leader-{node}.json")["logical_PCI"]
        need(len(pci_map) == len(set(pci_map)) == 8 and set(pci_map) == set(leases[NODES[node]]["PCI_to_KFD"]),
             "owner logical PCI mapping differs from the leased KFD receipt")
        common = sorted(os.sched_getaffinity(0))
        need(len(common) >= 64, "original 64-CPU node block absent")
        mask = set(common[dev*8:dev*8+4])
        os.sched_setaffinity(0,mask)
        need(os.sched_getaffinity(0) == mask, "original four-CPU owner mask differs")
        import torch
        from safetensors import safe_open
        torch.set_num_threads(1)
        torch.cuda.set_device(dev)
        sampler = Probe(dev)
        sampler.thread.start()
        native = load_native(HERE/"1071.json",HERE/"typed_binding.py")
        world,tp = actual_uids(out,run,owner,native,deadline)
        run_owner(node_index=node,device=dev,expected_pci=pci_map[dev],world_uid=world,tp_uid=tp,
                  torch=torch,safe_open=safe_open,native=native,input_binding=HERE/"INPUT_BINDING.json",
                  loader_path=HERE/"checkpoint.py",gate=gate,memory=sampler.memory,memory_summary=sampler.summary,
                  deadline=deadline,emit=emit,mark=mark)
    except Exception as exc:
        emit(dict(report,state="FAILED",worker_error=str(exc)[:500]))
        raise
    finally:
        channel.close()
        if sampler:
            sampler.close()


def connect(stage, frames, ips, listener, resources):
    limit = frames.limit(PHASES[0])
    while time.monotonic() < limit and not frames.stop_requested():
        if stage == 1:
            listener.settimeout(.25)
            try:
                conn,peer = listener.accept()
            except socket.timeout:
                continue
            resources.callback(conn.close)
            need(peer[0] == ips[0], "unexpected peer address")
            return conn
        conn = socket.socket()
        try:
            conn.bind((ips[0],0))
            conn.settimeout(min(3,max(.001,limit-time.monotonic())))
            conn.connect((ips[1],PORTS[1]))
        except ConnectionRefusedError:
            conn.close()
            time.sleep(.1)
            continue
        except Exception:
            conn.close()
            raise
        resources.callback(conn.close)
        return conn
    raise TimeoutError("new exact peer connection deadline")


def terminate(children, epoch):
    owned = [p for p in children if p.poll() is None]
    for p in owned:
        p.terminate()
    end = min(time.monotonic()+4,time.monotonic()+max(0,epoch+INSIDE-time.time()))
    for p in owned:
        try:
            p.wait(timeout=max(0,end-time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
    return [p.pid for p in owned]


def close_peer(conn,node,epoch):
    if node == 1:
        limit = time.monotonic()+min(5,budget(epoch,INSIDE))
        while time.monotonic() < limit:
            conn.settimeout(min(.25,max(.001,limit-time.monotonic())))
            try:
                data = conn.recv(1)
            except socket.timeout:
                continue
            need(data == b"", "unexpected bytes after final ready acknowledgement")
            break
        else:
            raise TimeoutError("ready peer did not close")
    conn.close()


def leader(out,node,intent,run):
    from bindings import NATIVE, verify_node_provider_bytes
    from run_pair import physical_pci
    epoch = intent["epoch"]
    ips = intent["claim"]["data_IPs"]
    children, locals_ = [], []
    frames = Frames(epoch,lambda:any(p.poll() not in (None,0) for p in children))
    report = dict(run=run,identity=process(node),spawned=[],rounds=[],state="STARTED")
    receipt = out/f"leader-{node}.json"
    put(receipt,report)
    with ExitStack() as resources:
        try:
            device = verify_lease_pins(out,intent,run)[NODES[node]]
            ldd = subprocess.check_output(["/usr/bin/ldd","-r",str(NATIVE)],text=True,
                                          timeout=min(30,budget(epoch,CONSTRUCT)))
            need(len(ldd) < 2**20 and "not found" not in ldd and "undefined symbol" not in ldd,
                 "unresolved native provider")
            paths = set(re.findall(r"(?:=>\s+|^\s*)(/[^\s()]+)",ldd,re.M))
            report["providers"] = verify_node_provider_bytes(paths,lambda _:budget(epoch,CONSTRUCT))
            report["logical_PCI"] = [physical_pci(dev) for dev in range(8)]
            need(len(set(report["logical_PCI"])) == 8 and set(report["logical_PCI"]) == set(device["PCI_to_KFD"]),
                 "logical device PCI differs from freshly leased hardware")
            report["leased_PCI_to_KFD"] = device["PCI_to_KFD"]
            listener = None
            if node == 1:
                listener = resources.enter_context(socket.socket())
                listener.bind((ips[1],PORTS[1]))
                listener.listen(1)
            put(receipt,report,replace=True)
            for dev in range(8):
                local,remote = socket.socketpair()
                resources.callback(local.close)
                with remote,(out/f"owner-{node*8+dev:02d}.log").open("xb") as log:
                    child = subprocess.Popen([sys.executable,"-B",str(HERE/"worker.py"),"owner",str(out),str(node),
                                              str(dev),str(remote.fileno())],pass_fds=(remote.fileno(),),
                                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    children.append(child)
                report["spawned"].append(dict(rank=node*8+dev,node=NODES[node],PID=child.pid,
                                              startticks=proc_start(child.pid),uid=os.getuid()))
                locals_.append(local)
                put(receipt,report,replace=True)
            conn = connect(node,frames,ips,listener,resources)
            for phase in PHASES:
                records = [frames.receive(local,phase) for local in locals_]
                for actual,saved in zip(records,report["spawned"]):
                    need(all(actual[key] == value for key,value in saved.items()), "unexpected local ready process")
                digest = exchange(conn,frames,node,phase,run,records)
                for local in locals_:
                    frames.send(local,dict(run=run,phase=phase,digest=digest),phase)
                report["rounds"].append(dict(phase=phase,digest=digest))
                put(receipt,report,replace=True)
            close_peer(conn,node,epoch)
            need([p.wait(timeout=budget(epoch,INSIDE)) for p in children] == [0]*8, "one or more owners failed")
            report["state"] = "COMPLETE"
        except Exception as exc:
            report.update(state="FAILED", error=str(exc)[:500], forced_PIDs=terminate(children,epoch))
            raise
        finally:
            put(receipt,report,replace=True)


def verify_lease_pins(out,intent,run):
    leases = {name:read_lease(out,name,run["step"]) for name in NODES}
    need(all(leases[name]["PCI"] == intent["claim"]["PCI_by_node"][name] for name in NODES),
         "leased DRM differs from approved per-card pin")
    return leases


def main(mode, root, *args):
    out = Path(root)
    intent = read_json(out/"intent.json")
    run = identity(intent)
    validate_claim(intent["claim"],run["source"])
    if mode == "node":
        verify_stage(HERE,run["source"])
    sys.path.insert(0,str(HERE/"pair"))
    node = int(os.environ["SLURM_PROCID"])
    need(node in (0,1) and socket.gethostname().split(".")[0] == NODES[node], "unexpected Slurm node/task rank")
    if mode == "node":
        leader(out,node,intent,run)
    else:
        need(mode == "owner" and len(args) == 3 and node == int(args[0]), "owner launcher identity differs")
        owner_main(out,node,int(args[1]),int(args[2]),intent,run)


if __name__ == "__main__":
    main(*sys.argv[1:])
