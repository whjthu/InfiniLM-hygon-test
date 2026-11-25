"""Observe the exact socket audit boundary without permitting a connection."""

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
from threading import local


FILENAME="PRIVATE_IPC_STATE.json"
CAP=32768


def encoded(data):
    return (json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(value):
    return sha256(value).hexdigest()


def enum_value(value,definition):
    try:
        return definition(value).name
    except (ValueError,TypeError,AttributeError):
        return "UNRECOGNIZED"


def process(pid,step,proc=Path("/proc")):
    try:
        target=proc/str(pid)
        pieces=(target/"stat").read_text("ascii").rsplit(")",1)[1].split()
        group=(target/"cgroup").read_bytes()
        parent,numeric=step.split(".")
        membership=any({"job_"+parent,"step_"+numeric}<=set(line.split(":",2)[-1].strip("/").split("/"))
                       for line in group.decode("ascii").splitlines())
        return {"pid":pid,"ppid":int(pieces[1]),"startticks":int(pieces[19]),"uid":target.stat().st_uid,
                "namespace_inode":(target/"ns/net").stat().st_ino,"cgroup_SHA256":digest(group),"same_slurm_step":membership}
    except (OSError,ValueError,IndexError,UnicodeError):
        return None


def callsite(frame):
    values=[]
    while frame is not None and len(values)<6:
        module=frame.f_globals.get("__name__","")
        fn=frame.f_code.co_name
        if isinstance(module,str) and re.fullmatch(r"[A-Za-z0-9_.]{1,100}",module):
            values.append({"module":module,"function":fn if re.fullmatch(r"[A-Za-z0-9_<>]{1,100}",fn) else "OTHER",
                           "line":frame.f_lineno})
        frame=frame.f_back
    return values


def private_dir(entry,uid):
    return stat.S_ISDIR(entry.st_mode) and entry.st_uid==uid and entry.st_mode&0o077==0


def scratch_root(mount,request_dir,tmpdir,uid):
    """Bind the actual mounted path, exclusive child and effective TMPDIR."""
    try:
        temp=Path(tmpdir)
        if (not mount.is_absolute() or mount.resolve(strict=True)!=mount
            or request_dir.parent!=mount or request_dir.resolve(strict=True)!=request_dir
            or temp.parent!=request_dir or temp.resolve(strict=True)!=temp
            or not all(private_dir(path.lstat(),uid) for path in (mount,request_dir,temp))):
            return None
        return mount
    except (OSError,RuntimeError,TypeError,ValueError):
        return None


def pathname_endpoint(value,root,uid,*,lookup=None):
    meta={"category":"UNKNOWN","identity":None}
    if root is None or uid is None or not value or "\x00" in value:
        return meta
    path=Path(value)
    if ".." in path.parts:
        return {**meta,"category":"UNSAFE_PATH_COMPONENT"}
    if not path.is_absolute():
        path=Path.cwd()/path
    if path==root or not path.is_relative_to(root):
        return {**meta,"category":"OUTSIDE_REQUEST_SCRATCH"}
    def get(entry):
        return entry.lstat() if lookup is None else lookup(entry)
    try:
        ancestors=tuple(reversed(path.parents))
        for directory in ancestors:
            info=get(directory)
            if stat.S_ISLNK(info.st_mode):
                return {**meta,"category":"UNSAFE_PATH_COMPONENT"}
            if not stat.S_ISDIR(info.st_mode):
                return {**meta,"category":"UNKNOWN"}
            if directory.is_relative_to(root) and not private_dir(info,uid):
                return {**meta,"category":"NONPRIVATE_REQUEST_PARENT"}
        info=get(path)
    except (OSError,ValueError):
        return meta
    if stat.S_ISLNK(info.st_mode):
        return {**meta,"category":"UNSAFE_PATH_COMPONENT"}
    identity={"uid":info.st_uid,"device":info.st_dev,"inode":info.st_ino,"mode":info.st_mode}
    if not stat.S_ISSOCK(info.st_mode):
        return {"category":"NONSOCKET_REQUEST_ENDPOINT","identity":identity}
    if info.st_uid!=uid:
        return {"category":"OTHER_UID_REQUEST_ENDPOINT","identity":identity}
    return {"category":"PRIVATE_REQUEST_SOCKET","identity":identity}


def endpoint_match(now,previous):
    first,second=now["identity"],previous["identity"]
    if first is None or second is None or now["category"]!="PRIVATE_REQUEST_SOCKET" or previous["category"]!="PRIVATE_REQUEST_SOCKET":
        return "UNKNOWN"
    identity=("uid","device","inode")
    return "MATCH" if all(first[k]==second[k] for k in identity) else "DIFFERENT"


class Admission:
    """One live local Manager endpoint per phase, within the one worker epoch."""

    PHASES=("swift_unshuffled_call","swift_shuffled_call")
    POLICY="REQUEST_PRIVATE_UNIX_EXACT_PHASE_ENDPOINT"
    MAX_INITIATORS=8
    MAX_BIND_CANDIDATES=8
    LAST_COUNTER=4294967295

    def __init__(self,path,owner,step,scratch,*,phase,deadline,actor=process,lookup=None):
        self.path=path
        self.lock=path.with_suffix(".lock")
        self.next=path.with_suffix(".next")
        self.owner,self.step,self.scratch=owner,step,scratch
        self.phase,self.deadline,self.actor,self.lookup=phase,deadline,actor,lookup
        self.pid=os.getpid()
        self.pending=local()
        self.local_errors=0
        self.root=self.current()
        with self.lock.open("xb"):
            pass
        empty=lambda:{"endpoint":None,"candidates":[],"initiators":{},"issued":0,"returned":0,
                      "failed":0,"denied":0,"chain_SHA256":digest(b""),"rejection":None}
        self.write({"schema":"R34_BOUNDED_PRIVATE_IPC_STATE_V1","request":owner,"step":step,"root":self.root,
                    "active":None,"next_phase":0,"failure":None,"phases":{key:empty() for key in self.PHASES}})

    def current(self):
        actor=self.actor(os.getpid(),self.step)
        if not actor or not actor["same_slurm_step"] or actor["uid"]!=os.getuid():
            raise ValueError("LIVE_INITIATOR_STEP_UNPROVEN")
        return actor

    @staticmethod
    def process_key(actor):
        return tuple(actor[key] for key in ("pid","startticks","uid","namespace_inode","cgroup_SHA256"))

    def read(self):
        with self.path.open("rb") as stream:
            raw=stream.read(CAP+1)
        if len(raw)>CAP:
            raise ValueError("PRIVATE_STATE_BOUND")
        return raw

    def write(self,value):
        data=encoded(value)
        if len(data)>CAP:
            raise ValueError("PRIVATE_STATE_BOUND")
        fd=os.open(self.next,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.next,self.path)
        directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @contextmanager
    def locked(self):
        import fcntl
        with self.lock.open("r+b",buffering=0) as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            value=json.loads(self.read())
            if (value["request"],value["step"],value["root"])!=(self.owner,self.step,self.root):
                raise ValueError("PRIVATE_STATE_WRONG_SCOPE")
            yield value

    def root_is_live(self):
        actor=self.actor(self.pid,self.step)
        return bool(actor and actor["same_slurm_step"] and self.process_key(actor)==self.process_key(self.root))

    def live(self,entry):
        if not entry:
            return False
        old=entry["process"]
        actor=self.actor(old["pid"],self.step)
        if not actor or not actor["same_slurm_step"] or self.process_key(actor)!=self.process_key(old):
            return False
        fd=Path("/proc")/str(old["pid"])/"fd"/str(entry["fd"])
        try:
            return os.readlink(fd)==f"socket:[{entry['socket_inode']}]"
        except FileNotFoundError:
            return False

    @contextmanager
    def loader_phase(self,phase):
        self.deadline()
        with self.locked() as value:
            if (os.getpid()!=self.pid or not self.root_is_live() or value["active"] is not None or value["failure"]
                or value["next_phase"]>=len(self.PHASES) or self.PHASES[value["next_phase"]]!=phase
                or any(self.live(row["endpoint"]) for row in value["phases"].values())):
                raise ValueError("EXACT_NEXT_LOADER_PHASE_UNPROVEN")
            value["active"]=phase
            self.write(value)
        try:
            yield
        finally:
            with self.locked() as value:
                if value["active"]!=phase:
                    raise ValueError("SHARED_LOADER_PHASE_CHANGED")
                value["active"]=None
                value["next_phase"]+=1
                self.write(value)

    @staticmethod
    def pathname(sock,value):
        if (enum_value(getattr(sock,"family",None),socket.AddressFamily)!="AF_UNIX"
            or enum_value(getattr(sock,"type",None),socket.SocketKind)!="SOCK_STREAM" or type(value) not in (str,bytes)):
            return None
        path=os.fsdecode(value)
        if not path or "\x00" in path or not Path(path).is_absolute() or len(os.fsencode(path))>107:
            return None
        return path

    def snapshot(self,actor,sock,path):
        endpoint=pathname_endpoint(path,self.scratch,actor["uid"],lookup=self.lookup)
        if endpoint["category"]!="PRIVATE_REQUEST_SOCKET":
            return None
        fd=sock.fileno()
        return {"process":actor,"fd":fd,"socket_inode":os.fstat(fd).st_ino,
                "file":endpoint,"address_SHA256":digest(os.fsencode(path))}

    def register(self,sock,listening):
        phase=self.phase()
        if phase not in self.PHASES:
            return
        path=self.pathname(sock,sock.getsockname())
        if path is None:
            return
        with self.locked() as value:
            if value["active"]!=phase or not self.root_is_live():
                return
            actor=self.current()
            entry=self.snapshot(actor,sock,path)
            if entry is None:
                return
            if not self.live(entry) or actor["namespace_inode"]!=self.root["namespace_inode"]:
                raise ValueError("PRIVATE_LISTENER_PROCESS_UNPROVEN")
            row=value["phases"][phase]
            candidates=row["candidates"]
            if not listening:
                if entry not in candidates:
                    if len(candidates)>=self.MAX_BIND_CANDIDATES:
                        value["failure"]="PRIVATE_BIND_CANDIDATE_LIMIT"
                        self.write(value)
                        raise ValueError("PRIVATE_BIND_CANDIDATE_LIMIT")
                    candidates.append(entry)
                    self.write(value)
                return
            other=[old["endpoint"] for key,old in value["phases"].items() if key!=phase and old["endpoint"]]
            if (entry not in candidates or row["endpoint"] not in (None,entry)
                or any(self.live(old) or old["address_SHA256"]==entry["address_SHA256"]
                       or endpoint_match(old["file"],entry["file"])=="MATCH" for old in other)):
                value["failure"]="PRIVATE_LISTEN_REPLACEMENT_OR_CROSS_PHASE_REUSE"
                self.write(value)
                raise ValueError("PRIVATE_LISTEN_REPLACEMENT_OR_CROSS_PHASE_REUSE")
            row["endpoint"]=entry
            self.write(value)

    def reserve(self,sock,address,site):
        self.deadline()
        phase=self.phase()
        if phase not in self.PHASES:
            return None,"OUTSIDE_SWIFT_LOADER"
        if [(x["module"],x["function"]) for x in site[:2]]!=[
            ("multiprocess.connection","SocketClient"),("multiprocess.connection","Client")]:
            return None,"UNPROVEN_MULTIPROCESS_CALLSITE"
        path=self.pathname(sock,address)
        if path is None:
            return None,"NOT_PRIVATE_UNIX_STREAM_PATHNAME"
        with self.locked() as value:
            if value["active"]!=phase or not self.root_is_live() or value["failure"]:
                return None,"SHARED_LOADER_PHASE_UNPROVEN"
            actor=self.current()
            if actor["namespace_inode"]!=self.root["namespace_inode"]:
                return None,"LIVE_INITIATOR_STEP_UNPROVEN"
            row=value["phases"][phase]
            endpoint=row["endpoint"]
            if (not endpoint or endpoint["address_SHA256"]!=digest(os.fsencode(path))
                or not self.live(endpoint) or endpoint not in row["candidates"]):
                return None,"PRIVATE_REGISTERED_LISTENER_UNPROVEN"
            current=pathname_endpoint(path,self.scratch,actor["uid"],lookup=self.lookup)
            if endpoint_match(endpoint["file"],current)!="MATCH":
                return None,"PRIVATE_PATHNAME_IDENTITY_CHANGED"
            key=digest(encoded(self.process_key(actor)))
            processes=row["initiators"]
            if ((key not in processes and len(processes)>=self.MAX_INITIATORS)
                or processes.get(key) or row["failed"] or row["issued"]>=self.LAST_COUNTER):
                return None,"PROCESS_SCOPE_OR_PENDING_CALL_LIMIT"
            self.deadline()
            serial=row["issued"]+1
            processes[key]=serial
            row["issued"]=serial
            row["chain_SHA256"]=digest(encoded((row["chain_SHA256"],"RESERVE",key,serial,endpoint["address_SHA256"])))
            self.write(value)
            return (phase,key,serial),"DURABLE_NATIVE_CONNECT_RESERVED"

    def finish(self,token,returned):
        phase,key,serial=token
        with self.locked() as value:
            row=value["phases"][phase]
            if row["initiators"][key]!=serial:
                raise ValueError("EXACT_PENDING_NATIVE_CONNECT_MISSING")
            row["initiators"][key]=0
            row["returned" if returned else "failed"]+=1
            row["chain_SHA256"]=digest(encoded((row["chain_SHA256"],"RETURNED" if returned else "FAILED",key,serial)))
            if not returned:
                value["failure"]="AUTHORIZED_NATIVE_CONNECT_FAILED"
            self.write(value)

    def refused(self,reason):
        phase=self.phase()
        if phase not in self.PHASES:
            return
        with self.locked() as value:
            row=value["phases"][phase]
            row["denied"]=min(self.LAST_COUNTER,row["denied"]+1)
            row["rejection"]=reason
            self.write(value)

    def guard(self,event,args,deny):
        if event!="socket.connect":
            return deny(event,args)
        pending=getattr(self.pending,"value",None)
        reason="NOT_WRAPPED_NATIVE_CONNECT"
        try:
            if pending is not None and len(args)==2 and args[0] is pending[0] and args[1]==pending[1] and pending[3] is None:
                token,reason=self.reserve(args[0],args[1],pending[2])
                if token is not None:
                    pending[3]=token
                    return None
        except (OSError,AttributeError,ValueError,TypeError,IndexError,KeyError,RuntimeError,ImportError):
            reason="EVIDENCE_OR_DEADLINE_UNAVAILABLE"
        try:
            self.refused(reason)
        except (OSError,ValueError,TypeError,KeyError,ImportError):
            self.local_errors+=1
        return deny(event,args)

    @contextmanager
    def track_local_listeners(self):
        original=(socket.socket.bind,socket.socket.listen,socket.socket.connect)
        def bind(sock,*args,**kwargs):
            result=original[0](sock,*args,**kwargs)
            self.register(sock,False)
            return result
        def listen(sock,*args,**kwargs):
            result=original[1](sock,*args,**kwargs)
            self.register(sock,True)
            return result
        def connect(sock,value):
            previous=getattr(self.pending,"value",None)
            attempt=[sock,value,callsite(sys._getframe(1)),None]
            self.pending.value=attempt
            try:
                try:
                    result=original[2](sock,value)
                except BaseException:
                    if attempt[3] is not None:
                        self.finish(attempt[3],False)
                    raise
                if attempt[3] is not None:
                    self.finish(attempt[3],True)
                return result
            finally:
                self.pending.value=previous
        socket.socket.bind,socket.socket.listen,socket.socket.connect=bind,listen,connect
        try:
            yield
        finally:
            socket.socket.bind,socket.socket.listen,socket.socket.connect=original

    def summary(self):
        raw=self.read()
        value=json.loads(raw)
        phases={key:{"registered_endpoints":int(row["endpoint"] is not None),"initiator_identities":len(row["initiators"]),
                     "permits":row["issued"],"native_returned":row["returned"],"native_failed":row["failed"],
                     "pending":sum(n!=0 for n in row["initiators"].values()),"denied":row["denied"],
                     "chain_SHA256":row["chain_SHA256"],"last_rejection":row["rejection"]} for key,row in value["phases"].items()}
        return {"policy":self.POLICY,"phase_metrics":phases,"phases_closed":value["next_phase"],
                "state_failure":value["failure"],"private_witness_SHA256":digest(raw),"private_witness_bytes":len(raw),
                "raw_addresses_forwarded":False,"owner_diagnostic_failures":self.local_errors}
