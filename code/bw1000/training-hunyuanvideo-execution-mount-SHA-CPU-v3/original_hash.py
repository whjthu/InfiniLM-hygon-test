"""Exact two purestdlib functions from published T09v2 checker; no imports of providers."""
import hashlib,math,os,time

def file_fingerprint(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def hash_on_execution_mount(path, stamp, deadline):
    budget = deadline - time.time()
    if not math.isfinite(budget) or budget <= 0:
        raise TimeoutError("Weight verification deadline exhausted")
    monotonic_end = time.monotonic() + budget

    def remaining():
        if time.time() >= deadline or time.monotonic() >= monotonic_end:
            raise TimeoutError("Weight verification deadline exhausted")

    digest = hashlib.sha256()
    size = 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if file_fingerprint(os.fstat(stream.fileno())) != stamp:
            raise ValueError("Weight changed before verification")
        while size < stamp[2]:
            remaining()
            chunk = stream.read(min(8 * 1024**2, stamp[2] - size))
            if not chunk:
                raise ValueError("Weight ended before audited extent")
            size += len(chunk)
            digest.update(chunk)
        remaining()
        if stream.read(1) or file_fingerprint(os.fstat(stream.fileno())) != stamp:
            raise ValueError("Weight changed during verification")
    if file_fingerprint(path.lstat()) != stamp:
        raise ValueError("Weight changed after verification")
    return digest.hexdigest(), size
