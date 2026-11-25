"""Bind a fresh read-only CPU ingress scope to the sealed T09 HOST stage."""
import json
from hashlib import sha1,sha256
from pathlib import Path,PurePosixPath
from contextlib import contextmanager,ExitStack
import os
import re
import stat

from framework_binding import COMMIT,MANIFEST_SHA256,TREE

PROJECT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
DEPLOYMENT=PROJECT/"results/training-qwen-a68952c-swift-initial-CPU-v1"
SHARED=DEPLOYMENT/"framework"
MANIFEST=DEPLOYMENT/"source/FRAMEWORK_FILES.json"
INVOCATIONS=PROJECT/"incoming/t09-private-framework393-ingress-v1"
STAGE_REQUEST="b1a692f520b94308a36818d2e1db252"
STAGE=INVOCATIONS/STAGE_REQUEST
STAGE_SHA="df12914b78960076cd134fecaff9abe44559b3a48cd4a44df74219ae7c64bc39"
AUDIT=INVOCATIONS/(STAGE_REQUEST+".stage-audit.jsonl")
AUDIT_SHA="10c4cf55fe87e1324674d450834e4a34bdf3ab8a8c08c490f43e7bf8f36864e3"
ARCHIVE_SHA="dd8ea6bf05bf4ce4862a095756a5e03759b864c80a2e5f8015ef5a237971b0fb"
REQUEST="a662d07645914562bbf3f56c9dd16f19"
LABEL="t09-private-framework393-cpu-a662d0764591"
PARENT,NODE="1103606","f13r3n13"
DENOMINATOR=393
MANIFEST_BYTES=112315
MAX_MEMBER=16<<20
BIND_ENV_KEYS=("APPTAINER_BIND","APPTAINER_BINDPATH","SINGULARITY_BIND","SINGULARITY_BINDPATH")


def exact_host_mount_context(value):
    flags=value.get("environment_bind_override_present")
    return (value.get("host_getcwd")=="/" and type(flags) is dict
            and set(flags)==set(BIND_ENV_KEYS) and all(type(item) is bool for item in flags.values()))

def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return sha256(raw).hexdigest()


def encoding(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


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
