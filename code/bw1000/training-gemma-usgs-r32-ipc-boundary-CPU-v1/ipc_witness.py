"""Observe the exact socket audit boundary without permitting a connection."""

from base64 import b64encode
from contextlib import contextmanager
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys


FILENAME="PRIVATE_IPC_WITNESS.jsonl"
CAP=32768
CALLERS={"multiprocessing.connection":"MULTIPROCESSING_CONNECTION","multiprocessing.managers":"MULTIPROCESSING_MANAGER",
         "socket":"PYTHON_SOCKET","datasets":"DATASETS","swift":"SWIFT","modelscope":"MODELSCOPE","multiprocess":"MULTIPROCESS"}


def encoded(data):
    return (json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(value):
    return sha256(value).hexdigest()


def enum_value(value,definition):
    try:
        return definition(value).name
    except (ValueError,TypeError,AttributeError):
        return "UNRECOGNIZED"


def address(value,family):
    if isinstance(value,bytes):
        kind="UNIX_ABSTRACT_BYTES" if family=="AF_UNIX" and value.startswith(b"\0") else "BYTES"
        return kind,{"base64":b64encode(value[:1024]).decode(),"extent":len(value),"truncated":len(value)>1024}
    if type(value) is str:
        kind=("UNIX_ABSTRACT_TEXT" if value.startswith("\0") else "UNIX_PATHNAME") if family=="AF_UNIX" else "TEXT"
        return kind,{"text":value[:1024],"extent":len(value),"truncated":len(value)>1024}
    if isinstance(value,tuple) and len(value) in (2,4) and type(value[0]) is str and all(type(n) is int for n in value[1:]):
        host=value[0]
        try:
            ip=ipaddress.ip_address(host.partition("%")[0])
            kind="INET_LOOPBACK" if ip.is_loopback else "INET_UNSPECIFIED" if ip.is_unspecified else "INET_OTHER"
        except ValueError:
            kind="INET_HOSTNAME_OR_NONLITERAL"
        return kind,{"host":host[:1024],"extent":len(host),"numbers":list(value[1:]),"truncated":len(host)>1024}
    if value is None:
        return "MISSING",None
    return "UNSUPPORTED_SHAPE",None


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


def caller_enum(source):
    for item in source:
        module=item["module"]
        for name,kind in CALLERS.items():
            if module==name or module.startswith(name+"."):
                return kind
    return "OTHER"


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


class Recorder:
    def __init__(self,path,owner,step,scratch=None,*,actor=process,lookup=None):
        self.path=path
        self.owner=owner
        self.step=step
        self.actor=actor
        self.scratch=scratch
        self.lookup=lookup
        self.denied=0
        self.failures=0
        self.last=None
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        os.close(fd)

    def read(self):
        with self.path.open("rb") as stream:
            raw=stream.read(CAP+1)
        if len(raw)>CAP:
            raise ValueError("R32_IPC_WITNESS_EXTENT")
        return raw

    def journal(self,row):
        import fcntl
        data=encoded(row)
        if len(data)>4096:
            raise ValueError("R32_IPC_EVENT_EXTENT")
        with self.path.open("ab",buffering=0) as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            if os.fstat(stream.fileno()).st_size+len(data)>CAP:
                raise ValueError("R32_IPC_WITNESS_EXTENT")
            stream.write(data)
            os.fsync(stream.fileno())

    def listener(self,candidate,here):
        if not here or not here["same_slurm_step"]:
            return None
        rows=[json.loads(line) for line in self.read().splitlines()]
        for row in reversed(rows):
            if (row["event"]!="LISTEN_SUCCEEDED" or row["address_SHA256"]!=candidate["address_SHA256"]
                or row["family"]!=candidate["family"] or row["request"]!=self.owner):
                continue
            peer=row["process"]
            if not peer or not peer["same_slurm_step"] or peer["uid"]!=here["uid"] or peer["namespace_inode"]!=here["namespace_inode"]:
                continue
            active=self.actor(peer["pid"],self.step)
            if (not active or not active["same_slurm_step"] or active["startticks"]!=peer["startticks"]
                or active["namespace_inode"]!=peer["namespace_inode"] or active["uid"]!=peer["uid"]):
                continue
            try:
                # The original successful local listen is correlated only when the
                # same current process still holds the same socket fd/inode.
                fd=Path("/proc")/str(peer["pid"])/"fd"/str(row["fd"])
                if os.readlink(fd)==f"socket:[{row['socket_inode']}]":
                    bind=next((entry for entry in reversed(rows) if entry["event"]=="BIND_SUCCEEDED"
                              and all(entry[key]==row[key] for key in ("request","step","family","process","fd","socket_inode","address_SHA256"))),None)
                    return {"process":peer,"listen_endpoint":endpoint_match(candidate["filesystem_endpoint"],row["filesystem_endpoint"]),
                            "bind_endpoint":endpoint_match(candidate["filesystem_endpoint"],bind["filesystem_endpoint"]) if bind else "UNKNOWN"}
            except OSError:
                continue
        return None

    def record(self,event,sock,value,frame):
        family=enum_value(getattr(sock,"family",None),socket.AddressFamily)
        kind,raw=address(value,family)
        provenance=self.actor(os.getpid(),self.step)
        source=callsite(frame)
        endpoint={"category":"NOT_PATHNAME","identity":None}
        if family=="AF_UNIX" and (kind=="UNIX_PATHNAME" or (kind=="BYTES" and isinstance(value,bytes))):
            endpoint=pathname_endpoint(os.fsdecode(value),self.scratch,provenance["uid"] if provenance else None,lookup=self.lookup)
        fd=inode=None
        try:
            fd=sock.fileno()
            inode=os.fstat(fd).st_ino
        except (AttributeError,OSError,TypeError):
            pass
        row={"event":event,"request":self.owner,"step":self.step,"family":family,
             "socket_type":enum_value(getattr(sock,"type",None),socket.SocketKind),
             "address_kind":kind,"address_SHA256":digest(encoded(raw)) if raw is not None else None,
             "raw_address_local_only":raw,"process":provenance,"callsite":source,"fd":fd,"socket_inode":inode,
             "filesystem_endpoint":endpoint}
        if event=="CONNECT_DENIED":
            matched=None if raw is None or raw.get("truncated") else self.listener(row,provenance)
            row["peer_observation"]=matched
            self.last=connect_detail(row)
        self.journal(row)

    def guard(self,event,args,deny):
        """A diagnostic error still returns to the exact original deny function."""
        if event=="socket.connect":
            self.denied+=1
            try:
                self.record("CONNECT_DENIED",args[0],args[1],sys._getframe(1))
            except (OSError,AttributeError,ValueError,TypeError,IndexError,KeyError):
                self.failures+=1
            finally:
                return deny(event,args)
        return deny(event,args)

    @contextmanager
    def track_local_listeners(self):
        original_bind,original_listen=socket.socket.bind,socket.socket.listen
        recorder=self
        def track(sock,event):
            try:
                recorder.record(event,sock,sock.getsockname(),sys._getframe(2))
            except (OSError,AttributeError,ValueError,TypeError,IndexError):
                recorder.failures+=1
        def bind(sock,*args,**kwargs):
            result=original_bind(sock,*args,**kwargs)
            track(sock,"BIND_SUCCEEDED")
            return result
        def listen(sock,*args,**kwargs):
            result=original_listen(sock,*args,**kwargs)
            track(sock,"LISTEN_SUCCEEDED")
            return result
        socket.socket.bind,socket.socket.listen=bind,listen
        try:
            yield
        finally:
            socket.socket.bind,socket.socket.listen=original_bind,original_listen

    def summary(self):
        raw=self.read()
        rows=[json.loads(line) for line in raw.splitlines()]
        denied=[row for row in rows if row["event"]=="CONNECT_DENIED"]
        result=self.last
        if result is None and denied:
            result=connect_detail(denied[-1])
        return {"policy":"OBSERVE_AND_DENY_ALL_ORIGINAL_CONNECTS","denied_connects_observed":max(self.denied,len(denied)),
                "binds_observed":sum(x["event"]=="BIND_SUCCEEDED" for x in rows),
                "listens_observed":sum(x["event"]=="LISTEN_SUCCEEDED" for x in rows),
                "diagnostic_failures_in_owner":self.failures,"last_observed_attempt":result,
                "private_witness_SHA256":digest(raw),"private_witness_bytes":len(raw),"raw_addresses_forwarded":False}


def connect_detail(row):
    current=row["process"]
    peer=row["peer_observation"]
    endpoint=row["filesystem_endpoint"]
    return {"family":row["family"],"socket_type":row["socket_type"],"address_kind":row["address_kind"],
            "address_SHA256":row["address_SHA256"],"caller_category":caller_enum(row["callsite"]),
            "callsite_SHA256":digest(encoded(row["callsite"])),
            "initiator":"SAME_REQUEST_SLURM_STEP_VERIFIED" if current and current["same_slurm_step"] else "UNCONFIRMED",
            "initiator_identity_SHA256":digest(encoded(current)) if current else None,
            "peer":"CORRELATED_LIVE_SAME_REQUEST_LISTENER" if peer else "UNCONFIRMED",
            "peer_identity_SHA256":digest(encoded(peer["process"])) if peer else None,
            "filesystem_endpoint":endpoint["category"],
            "filesystem_endpoint_SHA256":digest(encoded(endpoint["identity"])) if endpoint["identity"] else None,
            "same_as_successful_bind_endpoint":peer["bind_endpoint"] if peer else "UNKNOWN",
            "same_as_successful_listen_endpoint":peer["listen_endpoint"] if peer else "UNKNOWN",
            "connection_opened":False}
