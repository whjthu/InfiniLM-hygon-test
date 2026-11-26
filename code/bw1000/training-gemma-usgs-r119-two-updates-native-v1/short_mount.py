"""Bind the new owner's short private pathname without changing Swift."""

import os
from pathlib import Path
import stat
import tempfile

from contract import REQUEST, ROOT, need


SHORT = Path("/tmp/g119")
SOURCE = ROOT/"short-ipc"


def host_create_once():
    SOURCE.mkdir(mode=0o700)
    path=SOURCE/".scope"
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream:
        stream.write(REQUEST.encode("ascii")+b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    for directory in (SOURCE, ROOT):
        handle=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def verify_inside(*, check_temp_environment, require_marker_only=True):
    host=SOURCE.lstat()
    mounted=SHORT.lstat()
    need(SOURCE.resolve(strict=True)==SOURCE and SHORT.resolve(strict=True)==SHORT
         and host.st_ino==mounted.st_ino,
         "G42_CANONICAL_REQUEST_MOUNT_IDENTITY")
    for item in (host,mounted):
        need(stat.S_ISDIR(item.st_mode) and item.st_uid==os.getuid()
             and stat.S_IMODE(item.st_mode)==0o700, "G42_PRIVATE_REQUEST_MOUNT_MODE")
    marker=SHORT/".scope"
    info=marker.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o600
         and info.st_size==33 and marker.read_bytes()==REQUEST.encode("ascii")+b"\n",
         "G42_REQUEST_MOUNT_OWNER_MARKER")
    if check_temp_environment:
        need(all(os.environ.get(k)==str(SHORT) for k in ("TMPDIR","TMP","TEMP"))
             and tempfile.gettempdir()==str(SHORT), "G42_ENV_BINDING_BEFORE_NATIVE_IMPORT")
    if require_marker_only:
        need({p.name for p in SHORT.iterdir()}=={".scope"},"G42_PRIVATE_TEMP_NOT_EMPTY")
