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
REQUEST="3d616fad903f459e9bfb2c56d1a6314e"
DENOMINATOR=393
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


def secure_flags():
    return os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC


@contextmanager
def directory_fd(root):
    need(root.is_absolute() and ".." not in root.parts and root.anchor=="/","POSIX_ABSOLUTE_CANONICAL_ROOT")
    with ExitStack() as stack:
        fd=os.open("/",secure_flags())
        stack.callback(os.close,fd)
        for part in root.parts[1:]:
            fd=os.open(part,secure_flags(),dir_fd=fd)
            stack.callback(os.close,fd)
        yield fd


def read_file(base_fd, parts, extent, mode=None):
    with ExitStack() as owns:
        fd=base_fd
        for piece in parts[:-1]:
            fd=os.open(piece,secure_flags(),dir_fd=fd)
            owns.callback(os.close,fd)
        name=parts[-1]
        before=os.stat(name,dir_fd=fd,follow_symlinks=False)
        need(stat.S_ISREG(before.st_mode) and before.st_size==extent and
             (mode is None or stat.S_IMODE(before.st_mode)==mode),"MEMBER_TYPE_MODE_OR_EXTENT")
        flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC
        file=os.open(name,flags,dir_fd=fd)
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


def inspect_tree(root, files):
    regular=set()
    other=0
    symlinks=0
    dir_count=0

    def scan(fd, prefix):
        nonlocal other,symlinks,dir_count
        dir_count+=1
        need(dir_count<=1024,"FRAMEWORK_DIRECTORY_COUNT")
        with os.scandir(fd) as entries:
            for file in entries:
                rel="/".join((*prefix,file.name))
                mode=file.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    symlinks+=1
                elif stat.S_ISDIR(mode):
                    child=os.open(file.name,secure_flags(),dir_fd=fd)
                    try:
                        scan(child,(*prefix,file.name))
                    finally:
                        os.close(child)
                elif stat.S_ISREG(mode):
                    regular.add(rel)
                else:
                    other+=1
                need(len(regular)+symlinks+other<=2048,"FRAMEWORK_MEMBER_COUNT")
    with directory_fd(root) as fd:
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
        fd=os.open(path,secure_flags())
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def verify_member(raw,pin):
    need(sha(raw)==pin["SHA256"],"MEMBER_SHA256")
    header=f"blob {len(raw)}".encode()+b"\0"
    need(sha1(header+raw).hexdigest()==pin["git_blob"],"MEMBER_GIT_BLOB")


def stage(shared, manifest, result, *, manifest_sha=MANIFEST_SHA256,extent=MANIFEST_BYTES,
          denominator=DENOMINATOR, verify=verify_framework):
    # Ownership is acquired by mkdir. Final tree is renamed in this private
    # namespace only after every member and the original verifier qualify.
    with directory_fd(result.parent) as parent:
        os.mkdir(result.name,mode=0o700,dir_fd=parent)
    report=dict(request=REQUEST,phase="MANIFEST",passed=False,planned=denominator,copied=0,
                verified_original=0,source_mutated=False,historical_3524_extras_known=False,
                current_shared_readback_SHA256=SHARED_READBACK,previous_3524_public_SHA256=FAILED_3524)
    try:
        pending=result/".framework.pending"
        published=result/"framework"
        with directory_fd(manifest.parent) as fd:
            raw=read_file(fd,(manifest.name,),extent)
        pins=validate_manifest(raw,manifest_sha=manifest_sha,extent=extent,denominator=denominator)
        report["phase"]="SHARED_INVENTORY"
        with directory_fd(shared) as source:
            inv=inspect_tree(shared,pins)
            report["shared_inventory"]=inv
            need(not (inv["missing_members"] or inv["unexpected_extra_members"] or inv["symlinks"] or inv["special_members"]),
                 "SHARED_PINNED_TREE_MISSING_OR_UNCLASSIFIED")
            pending.mkdir(mode=0o700)
            report["phase"]="COPY_EXACT_MEMBER"
            for name,pin in sorted(pins.items()):
                report["member_name_SHA256"]=sha(name.encode())
                content=read_file(source,relative_parts(name),pin["bytes"],pin["mode"])
                verify_member(content,pin)
                target=pending/name
                target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                write_exclusive(target,content,pin["mode"])
                report["copied"]+=1
        report.pop("member_name_SHA256",None)
        report["phase"]="ORIGINAL_VERIFIER"
        control=result/"FRAMEWORK_FILES.json"
        write_exclusive(control,raw)
        result_native=verify(pending,control)
        staged=inspect_tree(pending,pins)
        need(staged["regular_members"]==denominator and not any(staged[key] for key in (
             "missing_members","extra_members","symlinks","special_members")),"STAGED_TREE_EXACT_INVENTORY")
        need(not published.exists() and not published.is_symlink(),"NEVER_REPLACE_EXISTING_PRIVATE_FRAMEWORK")
        report["phase"]="ATOMIC_FINAL_PUBLICATION"
        fsync_directories(pending)
        pending.rename(published)
        report.update(passed=True,phase="COMPLETE",verified_original=denominator,manifest_SHA256=manifest_sha,
                      staged_inventory=staged,original_verifier=result_native,atomic_private_tree=True)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        report.update(error_class=type(exc).__name__,errno=getattr(exc,"errno",None),
                      failure_code=str(exc) if type(exc) is ValueError and re.fullmatch("[A-Z0-9_]{1,80}",str(exc)) else None)
    finally:
        write_exclusive(result/"stage-result.json",encoding(report))
        fsync_directories(result)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,help="Existing dedicated absolute incoming parent")
    args=parser.parse_args()
    parent=INVOCATIONS if args.parent is None else args.parent
    need(parent==INVOCATIONS,"T09_CANONICAL_REMOTE_PRIVATE_PARENT")
    report=stage(SHARED,MANIFEST,parent/REQUEST)
    print(json.dumps(report,sort_keys=True))
    return 0 if report["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
