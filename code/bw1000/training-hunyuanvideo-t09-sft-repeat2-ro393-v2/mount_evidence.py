"""Identify the effective read-only Linux mount for each opened T09 object."""
from contextlib import ExitStack,contextmanager
from hashlib import sha256
import os
from pathlib import Path
import re
import stat

from framework_mount import FRAMEWORK as SHARED,FRAMEWORK_MANIFEST as MANIFEST,directory_fd


CAP=4<<20
ALIAS=Path("/t09-stage-ro")
OBJECTS=(("framework_directory",SHARED,ALIAS/"framework",True),
         ("framework_manifest",MANIFEST,ALIAS/"FRAMEWORK_FILES.json",False))


def need(ok, code):
    if not ok:
        raise ValueError(code)


def read_proc(path,cap=CAP):
    with Path(path).open("rb") as file:
        raw=file.read(cap+1)
    need(0<len(raw)<=cap,"BOUNDED_MOUNT_PROC_METADATA")
    return raw.decode("utf8")


def mount_path(field):
    return re.sub(r"\\(040|011|012|134)",lambda m:chr(int(m[1],8)),field)


def mount_rows(raw):
    records=[]
    for line in raw.splitlines():
        head,sep,tail=line.partition(" - ")
        cols=head.split()
        need(sep==" - " and len(cols)>=6 and len(tail.split())>=3 and cols[0].isdigit()
             and cols[1].isdigit(),"EXACT_PROC_MOUNTINFO_LAYOUT")
        root=mount_path(cols[3])
        point=mount_path(cols[4])
        need(root.startswith("/") and point.startswith("/"),"ABSOLUTE_EFFECTIVE_MOUNTINFO_PATHS")
        records.append(dict(id=int(cols[0]),parent=int(cols[1]),root=root,mountpoint=point,options=cols[5].split(",")))
    need(0<len(records)<=16384,"BOUNDED_MOUNTINFO_ROWS")
    return records


def effective_id(raw):
    ids=[m[1] for line in raw.splitlines() if (m:=re.fullmatch(r"mnt_id:\s*([0-9]+)",line))]
    need(len(ids)==1 and int(ids[0])>0,"ONE_EFFECTIVE_FD_MNT_ID")
    return int(ids[0])


@contextmanager
def open_readonly(path,is_dir,category):
    flag=os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK
    if is_dir:
        flag|=os.O_DIRECTORY
    with directory_fd(path.parent,category) as parent:
        fd=os.open(path.name,flag,dir_fd=parent)
        try:
            yield fd
        finally:
            os.close(fd)


def metadata_error(error):
    return dict(error_class=getattr(error,"source_error_class",type(error).__name__),
                errno=getattr(error,"errno",None),
                error_code=str(error) if type(error) is ValueError and re.fullmatch("[A-Z0-9_]{1,80}",str(error)) else None)


def project_fd(fd,raw_fdinfo,rows,expected,is_dir,*,fstat=None,fstatvfs=None):
    fstat=os.fstat if fstat is None else fstat
    fstatvfs=os.fstatvfs if fstatvfs is None else fstatvfs
    out=dict(opened=True,effective_mount_id=None,exact_effective_mount_records=None,
             effective_parent_mount_id=None,actual_mount_root_SHA256=None,actual_mountpoint_SHA256=None,
             expected_mountpoint_records=None,additional_records_at_mountpoint=None,
             effective_mountpoint_matches=None,effective_mount_option_ro=None,statvfs_readonly=None,
             object_type_matches=None,identity=None,passed=False)
    if rows is not None:
        out["expected_mountpoint_records"]=sum(r["mountpoint"]==str(expected) for r in rows)
        out["additional_records_at_mountpoint"]=max(0,out["expected_mountpoint_records"]-1)
    try:
        metadata=fstat(fd)
        out["identity"]={"device":metadata.st_dev,"inode":metadata.st_ino}
        out["object_type_matches"]=(stat.S_ISDIR(metadata.st_mode) if is_dir else stat.S_ISREG(metadata.st_mode))
    except (OSError,ValueError) as error:
        out["fstat_error"]=metadata_error(error)
    try:
        out["statvfs_readonly"]=bool(fstatvfs(fd).f_flag & os.ST_RDONLY)
    except (OSError,ValueError) as error:
        out["fstatvfs_error"]=metadata_error(error)
    try:
        number=effective_id(raw_fdinfo(fd))
        out["effective_mount_id"]=number
        if rows is not None:
            matches=[r for r in rows if r["id"]==number]
            out["exact_effective_mount_records"]=len(matches)
            if len(matches)==1:
                row=matches[0]
                out["effective_mountpoint_matches"]=row["mountpoint"]==str(expected)
                out["effective_parent_mount_id"]=row["parent"]
                out["actual_mount_root_SHA256"]=sha256(row["root"].encode()).hexdigest()
                out["actual_mountpoint_SHA256"]=sha256(row["mountpoint"].encode()).hexdigest()
                out["actual_mountpoint_kind"]=("EXPECTED_POINT" if out["effective_mountpoint_matches"] else
                    "EXPECTED_ANCESTOR" if expected.is_relative_to(Path(row["mountpoint"])) else "OTHER_POINT")
                out["effective_mount_option_ro"]="ro" in row["options"] and "rw" not in row["options"]
    except (OSError,ValueError) as error:
        out["fdinfo_error"]=metadata_error(error)
    out["passed"]=(out["exact_effective_mount_records"]==1 and out["effective_mountpoint_matches"] is True
         and out["effective_mount_option_ro"] is True and out["statvfs_readonly"] is True
         and out["object_type_matches"] is True)
    return out


def check_objects(*, opener=open_readonly,proc=read_proc,objects=OBJECTS,fstat=None,fstatvfs=None,
                  alias_root=ALIAS):
    results={}
    mout=dict(passed=False,objects=results,planned_objects=2,verified_objects=0,write_probe_performed=False,
              duplicate_rows_alone_block=False,effective_source="fdinfo.mnt_id")
    with ExitStack() as owners:
        fds={}
        for category,canonical,private,is_dir in objects:
            rolled=results[category]={"roles":{},"same_device":None,"same_inode":None,"passed":False}
            for role,path,expected in (("canonical",canonical,canonical),("private_alias",private,alias_root)):
                try:
                    fds[category,role]=owners.enter_context(opener(path,is_dir,category.upper()+"_"+role.upper()))
                except (OSError,ValueError) as error:
                    rolled["roles"][role]=dict(opened=False,passed=False,open_error=metadata_error(error))
        mountinfo=None
        try:
            mountinfo=mount_rows(proc("/proc/self/mountinfo"))
        except (OSError,ValueError) as error:
            mout["mountinfo_error"]=metadata_error(error)
        for category,canonical,private,is_dir in objects:
            rolled=results[category]
            for role,expected in (("canonical",canonical),("private_alias",alias_root)):
                key=category,role
                if key in fds:
                    rolled["roles"][role]=project_fd(fds[key],lambda fd:proc(f"/proc/self/fdinfo/{fd}",16384),
                                                    mountinfo,expected,is_dir,fstat=fstat,fstatvfs=fstatvfs)
            real,shadow=(rolled["roles"][k] for k in ("canonical","private_alias"))
            if real.get("identity") is not None and shadow.get("identity") is not None:
                rolled["same_device"]=real["identity"]["device"]==shadow["identity"]["device"]
                rolled["same_inode"]=real["identity"]["inode"]==shadow["identity"]["inode"]
            rolled["passed"]=real["passed"] and shadow["passed"] and rolled["same_device"] is True and rolled["same_inode"] is True
            mout["verified_objects"]+=int(rolled["passed"])
    mout["passed"]=mout["verified_objects"]==mout["planned_objects"]==len(results)
    return mout
