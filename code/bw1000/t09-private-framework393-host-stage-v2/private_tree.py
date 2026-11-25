"""Stage only the frozen T09 framework file inventory in a unique private scope."""
import argparse
from contextlib import contextmanager, ExitStack
from hashlib import sha1, sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from framework_binding import COMMIT, MANIFEST_SHA256, TREE, verify_framework


PROJECT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
DEPLOYMENT=PROJECT/"results/training-qwen-a68952c-swift-initial-CPU-v1"
SHARED=DEPLOYMENT/"framework"
MANIFEST=DEPLOYMENT/"source/FRAMEWORK_FILES.json"
INVOCATIONS=PROJECT/"incoming/t09-private-framework393-ingress-v1"
REQUEST="b1a692f520b94308a36818d2e1db252"
LABEL='t09-private-framework393-host-stage-v2'
DENOMINATOR=393
V1_FAILED_STAGE='14b65046c3d9d6fa31ce4b31ea8a4f682f7e504760648272a4798ae3450dadc6'
O_PATH_QUALIFICATION='d7c3b50854a15b0c069c75d782e2404477e0cc58cbc17b667cfe22d7a04712bf'
MANIFEST_BYTES=112315
MAX_MEMBER=16<<20
FAILED_3524="0506aaa86f28ee9b7d2d173866c4b025d51f758dcaba3f707e0ed9f7b889e0e5"
SHARED_READBACK="53cec3389ec9982a3691bf3ddf51e1e7d9636a4a2f603acca076ac801a13f7f6"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return sha256(raw).hexdigest()


def encoding(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def stamp(meta):
    return meta.st_dev,meta.st_ino,meta.st_mode,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns


def relative_parts(name):
    need(type(name) is str and 0<len(name)<=512 and "\\" not in name and "\0" not in name,"BAD_RELATIVE_PATH")
    path=PurePosixPath(name)
    need(path.parts and not path.is_absolute() and path.as_posix()==name and all(part not in (".","..") for part in path.parts),
         "BAD_RELATIVE_PATH")
    return path.parts


class DirectoryIOError(OSError):
    def __init__(self, error, root_category, operation, part_index):
        super().__init__(error.errno, "")
        self.source_error_class=type(error).__name__
        self.root_category=root_category
        self.path_operation=operation
        self.part_index=part_index


def open_dir(name, flags, *, category, operation, index, dir_fd=None):
    try:
        return os.open(name,flags,dir_fd=dir_fd)
    except OSError as error:
        raise DirectoryIOError(error,category,operation,index) from None


def search_flags():
    need(hasattr(os,"O_PATH"),"LINUX_O_PATH_REQUIRED")
    return os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC


def leaf_flags():
    return os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC


@contextmanager
def directory_fd(root, category):
    need(root.is_absolute() and ".." not in root.parts and root.anchor=="/","POSIX_ABSOLUTE_CANONICAL_ROOT")
    flags=search_flags()
    with ExitStack() as stack:
        fd=open_dir("/",flags,category=category,operation="SEARCH_ROOT",index=0)
        stack.callback(os.close,fd)
        for index,part in enumerate(root.parts[1:],1):
            fd=open_dir(part,flags,category=category,operation="SEARCH_ANCESTOR",index=index,dir_fd=fd)
            stack.callback(os.close,fd)
        yield fd


def error_projection(error):
    result=dict(error_class=getattr(error,"source_error_class",type(error).__name__),errno=getattr(error,"errno",None),
                failure_code=str(error) if type(error) is ValueError and re.fullmatch("[A-Z0-9_]{1,80}",str(error)) else None)
    if isinstance(error,DirectoryIOError):
        result.update(root_category=error.root_category,path_operation=error.path_operation,part_index=error.part_index)
    return result


class Audit:
    def __init__(self, stream):
        self.stream=stream
        self.sequence=0

    def record(self, report, event):
        self.sequence+=1
        need(self.sequence<=20,"BOUNDED_STAGE_AUDIT_EVENTS")
        fields=("phase","scope_search_attempted","scope_creation_attempted","scope_created","shared_read_attempted","copy_attempted",
                "publication_attempted","published","copied","verified_original","planned","passed","error_class",
                "errno","failure_code","root_category","path_operation","part_index","persisted_scope_receipt",
                "shared_inventory","persistence_error")
        row={name:report[name] for name in fields if name in report}
        row.update(request=REQUEST,sequence=self.sequence,event=event)
        self.stream.write(encoding(row))
        self.stream.flush()
        os.fsync(self.stream.fileno())


@contextmanager
def open_audit(parent):
    # The audit lives beside the scope in the known owner-private parent.
    # A direct O_PATH open can establish and bind it without listing ancestors.
    need(parent.resolve(strict=True)==parent,"EXACT_CANONICAL_AUDIT_PARENT")
    fd=open_dir(parent,search_flags(),category="PRIVATE_PARENT",operation="OPEN_AUDIT_PARENT",index=None)
    try:
        meta=os.fstat(fd)
        named=parent.lstat()
        need(stat.S_ISDIR(meta.st_mode) and meta.st_uid==os.getuid() and stat.S_IMODE(meta.st_mode)==0o700
             and (meta.st_dev,meta.st_ino)==(named.st_dev,named.st_ino),"OWNER_PRIVATE_AUDIT_PARENT")
        flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC
        handle=os.open(REQUEST+".stage-audit.jsonl",flags,0o600,dir_fd=fd)
        with os.fdopen(handle,"wb") as stream:
            audit=Audit(stream)
            audit.record(dict(phase="AUDIT_OPEN",planned=DENOMINATOR,copied=0,verified_original=0,
                      scope_creation_attempted=False,shared_read_attempted=False,copy_attempted=False,
                      publication_attempted=False,published=False,passed=False),"STARTED_EXCLUSIVE")
            try:
                sync=open_dir(".",leaf_flags(),category="PRIVATE_PARENT",operation="SYNC_AUDIT_PARENT_LEAF",
                              index=None,dir_fd=fd)
                try:
                    os.fsync(sync)
                finally:
                    os.close(sync)
            except OSError as error:
                audit.record(dict(phase="AUDIT_OPEN",planned=DENOMINATOR,copied=0,verified_original=0,
                                  scope_creation_attempted=False,**error_projection(error)),"FAILED")
                raise
            yield audit
    finally:
        os.close(fd)


def read_file(base_fd, parts, extent, mode=None, *, category):
    with ExitStack() as owns:
        fd=base_fd
        for index,piece in enumerate(parts[:-1]):
            fd=open_dir(piece,search_flags(),category=category,operation="SEARCH_MEMBER_ANCESTOR",index=index,dir_fd=fd)
            owns.callback(os.close,fd)
        name=parts[-1]
        try:
            before=os.stat(name,dir_fd=fd,follow_symlinks=False)
        except OSError as exc:
            raise DirectoryIOError(exc,category,"STAT_REGULAR_MEMBER",len(parts)-1) from None
        need(stat.S_ISREG(before.st_mode) and before.st_size==extent and
             (mode is None or stat.S_IMODE(before.st_mode)==mode),"MEMBER_TYPE_MODE_OR_EXTENT")
        flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC
        try:
            file=os.open(name,flags,dir_fd=fd)
        except OSError as exc:
            raise DirectoryIOError(exc,category,"OPEN_REGULAR_MEMBER_READONLY",len(parts)-1) from None
        with os.fdopen(file,"rb") as source:
            opened=os.fstat(source.fileno())
            need(stamp(before)==stamp(opened),"MEMBER_OPEN_IDENTITY_CHANGED")
            raw=source.read(extent+1)
            after=os.fstat(source.fileno())
        current=os.stat(name,dir_fd=fd,follow_symlinks=False)
        need(len(raw)==extent and stamp(before)==stamp(after)==stamp(current),"MEMBER_READ_IDENTITY_CHANGED")
        return raw


def validate_manifest(raw, *, manifest_sha=MANIFEST_SHA256, extent=MANIFEST_BYTES, denominator=DENOMINATOR):
    need(len(raw)==extent and sha(raw)==manifest_sha,"FROZEN_MANIFEST_BYTES")
    value=json.loads(raw)
    need(value["framework_commit"]==COMMIT and value["framework_tree"]==TREE
         and len(value["files"])==denominator,"FROZEN_MANIFEST_REVISION_DENOMINATOR")
    files=value["files"]
    for name, item in files.items():
        relative_parts(name)
        need(type(item["bytes"]) is int and 0<=item["bytes"]<=MAX_MEMBER and type(item["mode"]) is int
             and item["mode"] in (0o644,0o755) and re.fullmatch("[0-9a-f]{64}",item["SHA256"])
             and re.fullmatch("[0-9a-f]{40}",item["git_blob"]),"FROZEN_MANIFEST_MEMBER_SCHEMA")
    return files


def inspect_tree(root, files, *, category):
    regular=set()
    other=0
    symlinks=0
    dir_count=0

    def scan(search_fd, prefix):
        nonlocal other,symlinks,dir_count
        dir_count+=1
        need(dir_count<=1024,"FRAMEWORK_DIRECTORY_COUNT")
        fd=open_dir(".",leaf_flags(),category=category,operation="ENUMERATE_VERIFIED_LEAF",
                     index=len(prefix),dir_fd=search_fd)
        try:
            try:
                scan_entries(fd,prefix)
            except DirectoryIOError:
                raise
            except OSError as error:
                raise DirectoryIOError(error,category,"ENUMERATE_VERIFIED_LEAF",len(prefix)) from None
        finally:
            os.close(fd)

    def scan_entries(fd,prefix):
        nonlocal other,symlinks
        with os.scandir(fd) as entries:
            for file in entries:
                rel="/".join((*prefix,file.name))
                mode=file.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    symlinks+=1
                elif stat.S_ISDIR(mode):
                    child=open_dir(file.name,search_flags(),category=category,operation="SEARCH_TREE_MEMBER",
                                   index=len(prefix)+1,dir_fd=fd)
                    try:
                        scan(child,(*prefix,file.name))
                    finally:
                        os.close(child)
                elif stat.S_ISREG(mode):
                    regular.add(rel)
                else:
                    other+=1
                need(len(regular)+symlinks+other<=2048,"FRAMEWORK_MEMBER_COUNT")
    with directory_fd(root,category) as fd:
        scan(fd,())
    missing=sorted(set(files)-regular)
    extras=sorted(regular-set(files))
    pyc=[item for item in extras if Path(item).parent.name=="__pycache__"
         and re.fullmatch(r".+\.cpython-[0-9]{2,4}(?:\.opt-[0-9]+)?\.pyc",Path(item).name)]
    return dict(manifest_members=len(files),regular_members=len(regular),missing_members=len(missing),
                missing_set_SHA256=sha(encoding(missing)),extra_members=len(extras),
                extra_set_SHA256=sha(encoding(extras)),ordinary_extra_pyc=len(pyc),
                unexpected_extra_members=len(extras)-len(pyc),symlinks=symlinks,special_members=other)


def write_exclusive(path,raw,mode=0o600):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,mode)
    with os.fdopen(fd,"wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fchmod(stream.fileno(),mode)
        os.fsync(stream.fileno())


def fsync_directories(root):
    dirs=[Path(folder) for folder,_,_ in os.walk(root,topdown=False,followlinks=False)]
    for path in dirs:
        fd=os.open(path,leaf_flags())
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def verify_member(raw,pin):
    need(sha(raw)==pin["SHA256"],"MEMBER_SHA256")
    header=f"blob {len(raw)}".encode()+b"\0"
    need(sha1(header+raw).hexdigest()==pin["git_blob"],"MEMBER_GIT_BLOB")


def stage(shared, manifest, result, audit, *, manifest_sha=MANIFEST_SHA256,extent=MANIFEST_BYTES,
          denominator=DENOMINATOR, verify=verify_framework):
    report=dict(request=REQUEST,phase="ALLOCATE_PRIVATE_SCOPE",passed=False,planned=denominator,copied=0,
                verified_original=0,source_mutated=False,historical_3524_extras_known=False,
                current_shared_readback_SHA256=SHARED_READBACK,previous_3524_public_SHA256=FAILED_3524,
                previous_v1_failed_stage_SHA256=V1_FAILED_STAGE,
                actual_login_o_path_qualification_SHA256=O_PATH_QUALIFICATION,
                scope_search_attempted=True,scope_creation_attempted=False,scope_created=False,
                shared_read_attempted=False,copy_attempted=False,
                publication_attempted=False,published=False,persisted_scope_receipt=False)

    def mark(phase,event="ENTER",**changes):
        report.update(phase=phase,**changes)
        audit.record(report,event)

    try:
        audit.record(report,"ENTER")
        with directory_fd(result.parent,"PRIVATE_PARENT") as parent:
            mark("CREATE_SCOPE",scope_creation_attempted=True)
            os.mkdir(result.name,mode=0o700,dir_fd=parent)
        report["scope_created"]=True
        pending=result/".framework.pending"
        published=result/"framework"
        mark("MANIFEST",shared_read_attempted=True)
        with directory_fd(manifest.parent,"SHARED_MANIFEST_PARENT") as fd:
            raw=read_file(fd,(manifest.name,),extent,category="SHARED_MANIFEST_FILE")
        pins=validate_manifest(raw,manifest_sha=manifest_sha,extent=extent,denominator=denominator)
        mark("SHARED_INVENTORY")
        with directory_fd(shared,"SHARED_FRAMEWORK") as source:
            inv=inspect_tree(shared,pins,category="SHARED_FRAMEWORK")
            report["shared_inventory"]=inv
            need(not (inv["missing_members"] or inv["unexpected_extra_members"] or inv["symlinks"] or inv["special_members"]),
                 "SHARED_PINNED_TREE_MISSING_OR_UNCLASSIFIED")
            pending.mkdir(mode=0o700)
            mark("COPY_EXACT_MEMBER",copy_attempted=True)
            for name,pin in sorted(pins.items()):
                report["member_name_SHA256"]=sha(name.encode())
                content=read_file(source,relative_parts(name),pin["bytes"],pin["mode"],
                                  category="SHARED_FRAMEWORK_MEMBER")
                verify_member(content,pin)
                target=pending/name
                target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                write_exclusive(target,content,pin["mode"])
                report["copied"]+=1
        report.pop("member_name_SHA256",None)
        mark("ORIGINAL_VERIFIER")
        control=result/"FRAMEWORK_FILES.json"
        write_exclusive(control,raw)
        result_native=verify(pending,control)
        staged=inspect_tree(pending,pins,category="PRIVATE_FRAMEWORK")
        need(staged["regular_members"]==denominator and not any(staged[key] for key in (
             "missing_members","extra_members","symlinks","special_members")),"STAGED_TREE_EXACT_INVENTORY")
        report["verified_original"]=denominator
        mark("ATOMIC_FINAL_PUBLICATION",publication_attempted=True)
        need(not published.exists() and not published.is_symlink(),"NEVER_REPLACE_EXISTING_PRIVATE_FRAMEWORK")
        fsync_directories(pending)
        pending.rename(published)
        report.update(published=True,passed=True,phase="COMPLETE",manifest_SHA256=manifest_sha,
                      staged_inventory=staged,original_verifier=result_native,atomic_private_tree=True)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        report.update(error_projection(exc))
    finally:
        if report["scope_created"]:
            try:
                report["persisted_scope_receipt"]=True
                write_exclusive(result/"stage-result.json",encoding(report))
                fsync_directories(result)
            except (OSError,ValueError) as exc:
                report["persisted_scope_receipt"]=False
                report["passed"]=False
                report["persistence_error"]=error_projection(exc)
        audit.record(report,"COMPLETE" if report["passed"] else "FAILED")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,help="Existing dedicated absolute incoming parent")
    args=parser.parse_args()
    parent=INVOCATIONS if args.parent is None else args.parent
    need(parent==INVOCATIONS,"T09_CANONICAL_REMOTE_PRIVATE_PARENT")
    try:
        with open_audit(parent) as audit:
            report=stage(SHARED,MANIFEST,parent/REQUEST,audit)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        report=dict(request=REQUEST,phase="AUDIT_OPEN_OR_STAGE_OUTCOME",passed=False,planned=DENOMINATOR,
                    scope_creation_attempted=None,shared_read_attempted=None,readback_required=True,**error_projection(exc))
    print(json.dumps(report,sort_keys=True))
    return 0 if report["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
