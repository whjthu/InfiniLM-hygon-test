"""Validated append-only continuation of an exact immutable file prefix."""
import hashlib,os
def prefix_hash(path,offset,expected):
    if offset<0:raise ValueError("negative offset")
    h=hashlib.sha256()
    with path.open("rb") as f:
        left=offset
        while left:
            raw=f.read(min(left,1024*1024))
            if not raw:raise EOFError("prefix is shorter than the registered offset")
            h.update(raw);left-=len(raw)
    if h.hexdigest()!=expected:raise ValueError("prefix digest mismatch")
    return h
def append_verified(path,stream,offset,total,prefix_sha,full_sha,progress):
    if offset<0 or total<=offset:raise ValueError("invalid continuation range")
    before=path.stat()
    if before.st_size!=offset or path.is_symlink():raise ValueError("partial size or ownership shape changed")
    h=prefix_hash(path,offset,prefix_sha)
    with path.open("r+b") as f:
        actual=os.fstat(f.fileno())
        if (actual.st_dev,actual.st_ino,actual.st_size,actual.st_mtime_ns)!=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns):
            raise ValueError("partial changed between verification and append")
        f.seek(offset);left=total-offset;copied=0
        while left:
            raw=stream.read(min(left,1024*1024))
            if not raw:raise EOFError("continuation ended before registered total")
            f.write(raw);h.update(raw);left-=len(raw);copied+=len(raw)
            progress(offset+copied,h.hexdigest())
        f.flush();os.fsync(f.fileno())
    if stream.read(1):raise ValueError("excess bytes after registered total")
    if h.hexdigest()!=full_sha:raise ValueError("whole-file digest mismatch")
    return {"bytes":total,"SHA256":h.hexdigest(),"continued_bytes":total-offset,"retained_prefix_bytes":offset}
