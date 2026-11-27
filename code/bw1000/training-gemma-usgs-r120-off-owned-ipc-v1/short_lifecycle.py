"""Observe only top-level metadata in this owner's verified short mount."""
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import time


def observation(path, uid, *, max_members=128):
    root = path.lstat()
    if (path.resolve(strict=True) != path or not stat.S_ISDIR(root.st_mode)
            or stat.S_IMODE(root.st_mode) != 0o700 or root.st_uid != uid):
        raise ValueError("R46_SHORT_IPC_OBSERVATION_ROOT")
    root_key = root.st_dev, root.st_ino
    result = {}
    for child in path.iterdir():
        if len(result) >= max_members:
            raise ValueError("R46_SHORT_IPC_OBSERVATION_BOUND")
        try:
            meta = child.lstat()
        except FileNotFoundError:
            continue
        category = ("SCOPE" if child.name == ".scope" else
                    "PYMP" if re.fullmatch(r"pymp-[a-z0-9_]{8}", child.name) else
                    "TORCHINDUCTOR" if child.name.startswith("torchinductor_") else "OTHER")
        kind = ("DIRECTORY" if stat.S_ISDIR(meta.st_mode) else "SYMLINK" if stat.S_ISLNK(meta.st_mode)
                else "SOCKET" if stat.S_ISSOCK(meta.st_mode) else "REGULAR" if stat.S_ISREG(meta.st_mode)
                else "OTHER")
        key = sha256(os.fsencode(child.name)).hexdigest()
        identity = sha256(f"{meta.st_dev}:{meta.st_ino}:{meta.st_uid}".encode()).hexdigest()
        result[key] = dict(category=category, kind=kind, same_UID=meta.st_uid == uid,
                           same_mount=meta.st_dev == root.st_dev, private_mode=oct(stat.S_IMODE(meta.st_mode)),
                           identity_SHA256=identity)
    current = path.lstat()
    if root_key != (current.st_dev, current.st_ino):
        raise ValueError("R46_SHORT_IPC_OBSERVATION_ROOT_CHANGED")
    return result


class Lifecycle:
    def __init__(self, path, uid, *, archive_path):
        self.path, self.uid = Path(path), uid
        self.started = time.monotonic()
        self.current = {}
        self.total_samples = 0
        self.errors = []
        self.events = []
        self.archive = EventArchive(archive_path)
        self.truncated_events = 0
        self.checkpoints = {}

    def sample(self, stage):
        self.total_samples += 1
        try:
            now = observation(self.path, self.uid)
        except (OSError, ValueError) as exc:
            if len(self.errors) < 16:
                self.errors.append(dict(stage=stage, error_class=type(exc).__name__,
                                        errno=getattr(exc,"errno",None)))
            return
        elapsed_ms = int((time.monotonic()-self.started)*1000)
        for key in sorted(self.current.keys() | now.keys()):
            previous, current = self.current.get(key), now.get(key)
            if previous == current:
                continue
            event = "ADDED" if previous is None else "DISAPPEARED" if current is None else "CHANGED"
            record=dict(stage=stage,elapsed_ms=elapsed_ms,event=event,basename_SHA256=key,current=current,previous=previous)
            try:
                if not self.archive.append(record): self.truncated_events += 1
            except OSError as exc:
                self.truncated_events += 1
                if len(self.errors)<16: self.errors.append(dict(stage=stage,error_class=type(exc).__name__,errno=exc.errno))
            if len(self.events)<128: self.events.append(record)
        self.current = now
        if stage != "RUNNING":
            self.checkpoints[stage] = now

    def evidence(self):
        return dict(schema="GEMMA_R120_REQUEST_SHORT_DIRECTORY_LIFECYCLE_V2",samples=self.total_samples,
                    event_archive=self.archive.seal(),inline_omitted_events=max(0,self.archive.count-len(self.events)),
                    observed_only=True,creator_attribution=False,children_traversed=False,raw_paths=False,
                    errors=self.errors,events=self.events,truncated_events=self.truncated_events,
                    checkpoints=self.checkpoints)


def receipt_domains(*, exact_rank_results, returncode, failure, cleanup_errors, held, tree, retirement):
    training = len(exact_rank_results) == 8 and returncode == 0 and failure is None
    owner = (not cleanup_errors and held.get("reaped") is True and held.get("termination_needed") is False
             and tree.get("owned_processes_remaining") == 0 and tree.get("termination_needed") is False
             and retirement.get("state") == "MARKER_ONLY")
    return dict(native_two_updates_witnessed=training, owner_cleanup_passed=owner, passed=training and owner)


# Added bounded evidence and natural-only retirement to the original observer.

import json


class EventArchive:
    MAX_BYTES = 4 << 20
    MAX_EVENTS = 16384

    def __init__(self,path):
        self.path=Path(path)
        self.stream=None
        self.count=0
        self.height=0
        self.hash=sha256()
        self.sealed=None

    def append(self,event):
        data=(json.dumps(dict(sequence=self.count+1,**event),sort_keys=True,allow_nan=False)+"\n").encode()
        if self.count>=self.MAX_EVENTS or self.height+len(data)>self.MAX_BYTES:
            return False
        if self.stream is None:
            self.stream=self.path.open("xb")
        if self.stream.write(data)!=len(data):
            raise OSError("IPC_EVENT_ARCHIVE_SHORT_WRITE")
        self.hash.update(data)
        self.count+=1
        self.height+=len(data)
        return True

    def seal(self):
        if self.sealed is not None:
            return self.sealed
        if self.stream is None:
            self.stream=self.path.open("xb")
        with self.stream as stream:
            stream.flush()
            os.fsync(stream.fileno())
        self.sealed=dict(relative=self.path.name,count=self.count,bytes=self.height,SHA256=self.hash.hexdigest())
        return self.sealed


def _require(ok,code):
    if not ok:
        raise ValueError(code)


def mount_witness(root,request):
    """Read authenticated current root and marker without following their final links."""
    root=Path(root)
    uid=os.getuid()
    _require(root.resolve(strict=True)==root,"IPC_CANONICAL_ROOT")
    flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC
    fd=os.open(root,flags|os.O_DIRECTORY)
    try:
        directory=os.fstat(fd)
        _require(stat.S_ISDIR(directory.st_mode) and directory.st_uid==uid
                 and stat.S_IMODE(directory.st_mode)==0o700,"IPC_ROOT_PRIVATE_MODE")
        marker=os.open(".scope",flags,dir_fd=fd)
        with os.fdopen(marker,"rb") as stream:
            info=os.fstat(stream.fileno())
            contents=stream.read(34)
            _require(stat.S_ISREG(info.st_mode) and info.st_uid==uid and info.st_dev==directory.st_dev
                     and stat.S_IMODE(info.st_mode)==0o600 and info.st_nlink==1
                     and info.st_size==33 and contents==request.encode("ascii")+b"\n",
                     "IPC_MARKER_IDENTITY_OR_BYTES")
            def key(value):
                return (value.st_dev,value.st_ino,value.st_uid,value.st_mode,value.st_nlink,value.st_size,
                        value.st_mtime_ns,value.st_ctime_ns)
            _require(key(info)==key(os.fstat(stream.fileno()))==
                     key(os.stat(".scope",dir_fd=fd,follow_symlinks=False)),
                     "IPC_MARKER_CHANGED_DURING_READ")
        after=root.lstat()
        _require((after.st_dev,after.st_ino,after.st_uid,after.st_mode)==
                 (directory.st_dev,directory.st_ino,directory.st_uid,directory.st_mode),"IPC_ROOT_CHANGED")
        return dict(root=dict(ino=directory.st_ino,uid=directory.st_uid,dev=directory.st_dev,mode=0o700),
                    marker=dict(ino=info.st_ino,uid=info.st_uid,dev=info.st_dev,mode=0o600,
                                mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,
                                bytes=len(contents),SHA256=sha256(contents).hexdigest()))
    finally:
        os.close(fd)


def compare_mount(creation,host,guest):
    _require(creation["request"] and creation["marker_only_initial"] is True,"IPC_CREATION_EVIDENCE_REQUIRED")
    basis=creation["identity"]
    for part in ("root","marker"):
        expected={k:v for k,v in basis[part].items() if k!="dev"}
        for actual in (host,guest):
            _require({k:v for k,v in actual[part].items() if k!="dev"}==expected,"IPC_CREATED_ROOT_OR_MARKER_CHANGED")
    return dict(observed_creator="SUPERVISOR_HOST_CREATE_ONCE_ROOT_AND_SCOPE",
                mount_namespace=dict(host_device_rebased=host["root"]["dev"]!=basis["root"]["dev"],
                                     guest_device_rebased=guest["root"]["dev"]!=basis["root"]["dev"]),
                creation_identity_SHA256=sha256(json.dumps(basis,sort_keys=True).encode()).hexdigest())


def residuals(path,uid,*,limit=128,child_limit=64):
    """Bounded nofollow observation. Names and contents are withheld, including NFS-like files."""
    p=Path(path)
    meta=p.lstat()
    initial=(meta.st_dev,meta.st_ino,meta.st_uid,meta.st_mode)
    found={}

    def attributes(name, item, directory):
        kind=("DIRECTORY" if stat.S_ISDIR(item.st_mode) else
              "SOCKET" if stat.S_ISSOCK(item.st_mode) else
              "SYMLINK" if stat.S_ISLNK(item.st_mode) else
              "REGULAR" if stat.S_ISREG(item.st_mode) else "OTHER")
        category=("PYMP" if re.fullmatch(r"pymp-[a-z0-9_]{8}",name) else
                  "TORCHINDUCTOR" if name.startswith("torchinductor_") else
                  "NFS_GENERATED_NAME_LIKE" if re.fullmatch(r"\.nfs[0-9a-fA-F]+",name) else
                  "MP_LISTENER_NAME_LIKE" if re.fullmatch(r"listener-[a-z0-9_]{8}",name) else "OTHER")
        identity=sha256(f"{item.st_dev}:{item.st_ino}:{item.st_uid}".encode()).hexdigest()
        return dict(category=category,kind=kind,same_UID=item.st_uid==uid,same_mount=item.st_dev==directory.st_dev,
                    mode=oct(stat.S_IMODE(item.st_mode)),identity_SHA256=identity,creator_verified=False)

    with os.scandir(p) as entries:
        for entry in entries:
            if entry.name==".scope":
                continue
            _require(len(found)<limit,"IPC_RESIDUAL_TOP_BOUND")
            child=entry.stat(follow_symlinks=False)
            tag=attributes(entry.name,child,meta)
            if tag["kind"]=="DIRECTORY" and tag["same_UID"] and tag["same_mount"]:
                flags=(os.O_RDONLY|os.O_NOFOLLOW|os.O_DIRECTORY|os.O_CLOEXEC)
                fd=os.open(entry.path,flags)
                try:
                    opened=os.fstat(fd)
                    _require((opened.st_dev,opened.st_ino,opened.st_uid,opened.st_mode)==
                             (child.st_dev,child.st_ino,child.st_uid,child.st_mode),"IPC_RESIDUAL_CHILD_REPLACED")
                    counts={}
                    members=0
                    with os.scandir(fd) as inner:
                        for item in inner:
                            members+=1
                            _require(members<=child_limit,"IPC_RESIDUAL_CHILD_BOUND")
                            value=attributes(item.name,item.stat(follow_symlinks=False),opened)
                            category=value["category"]+"/"+value["kind"]
                            counts[category]=counts.get(category,0)+1
                    tag["direct_members"]=members
                    tag["direct_members_categories"]=counts
                finally:
                    os.close(fd)
            found[sha256(os.fsencode(entry.name)).hexdigest()]=tag
    after=p.lstat()
    _require(initial==(after.st_dev,after.st_ino,after.st_uid,after.st_mode),"IPC_RESIDUAL_ROOT_REPLACED")
    return found


def retire_natural(host,guest,creation,request,*,initially_marker_only,children_reaped,report):
    """Observe the original cleanup terminal shape and keep every unproved residual."""
    report.update(initially_marker_only=initially_marker_only,children_reaped=children_reaped,
                  state="REFUSED",before=None,after=None,removed=0)
    _require(initially_marker_only is True and children_reaped is True and creation["request"]==request,
             "IPC_RETIREMENT_LINEAGE_OR_CHILDREN_UNPROVEN")
    before_host,before_guest=mount_witness(host,request),mount_witness(guest,request)
    report["custody"]=compare_mount(creation,before_host,before_guest)
    left=residuals(host,os.getuid())
    right=residuals(guest,os.getuid())
    # Compare hashed names and mode/type only. Device numbering can differ across mounts.
    def fields(value):
        return {key:(item["category"],item["kind"],item["mode"],item.get("direct_members"))
                for key,item in value.items()}
    report["before"]={"residuals":right,"host_guest_match":fields(left)==fields(right)}
    _require(report["before"]["host_guest_match"],"IPC_HOST_GUEST_RESIDUE_DIFFERS")
    _require(not left and not right,"IPC_UNPROVEN_RESIDUE_RETAINED")
    after_host,after_guest=mount_witness(host,request),mount_witness(guest,request)
    compare_mount(creation,after_host,after_guest)
    _require(not residuals(host,os.getuid()) and not residuals(guest,os.getuid()),"IPC_RESIDUE_APPEARED_DURING_OBSERVATION")
    report.update(after=dict(marker_valid=True,candidate_directories=0,unexpected=0),state="MARKER_ONLY")
