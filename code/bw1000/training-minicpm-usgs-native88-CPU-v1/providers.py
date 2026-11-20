"""Bind actual Swift/HF Python providers to already cached official wheels."""
import hashlib
import inspect
from pathlib import Path
import zipfile

WHEELS={
    "swift":dict(root="/swift-runtime/site",path="/wheels/swift/ms_swift-4.4.2-py3-none-any.whl",
                 bytes=1239486,SHA256="869b573de4a24129140a2b03b8c031b3c4024bba11ab768a9477f8d1dada87aa"),
    "transformers":dict(root="/minicpm-transformers",path="/wheels/transformers/transformers-5.12.1-py3-none-any.whl",
                        bytes=11150587,SHA256="2a5e109d2021265df7098ffbb738295acaf5ad256f12cbc586db2ea4dcbb1a8a"),
}


def verify_wheels():
    result={}
    for name,pin in WHEELS.items():
        p=Path(pin["path"])
        before=p.stat()
        raw=p.read_bytes()
        after=p.stat()
        if p.is_symlink() or len(raw)!=pin["bytes"] or hashlib.sha256(raw).hexdigest()!=pin["SHA256"]:
            raise ValueError("Original cached wheel differs")
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError("Original cached wheel changed")
        result[name]=dict(bytes=len(raw),SHA256=pin["SHA256"])
    return result


def file_identity(filename):
    p=Path(filename).resolve(strict=True)
    raw=p.read_bytes()
    if len(raw)>256*1024:
        raise ValueError("Actual provider exceeds bounded source scope")
    record=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    for package,pin in WHEELS.items():
        root=Path(pin["root"]).resolve(strict=True)
        if p.is_relative_to(root):
            member=p.relative_to(root).as_posix()
            with zipfile.ZipFile(pin["path"]) as wheel:
                info=wheel.getinfo(member)
                if info.file_size>256*1024 or wheel.read(info)!=raw:
                    raise ValueError("Actual native provider differs from cached official wheel")
            record.update(package=package,wheel_member=member,official_wheel_bytes_equal=True)
            return record
    # A Torch @contextmanager wrapper can own the outer __code__. This is
    # recorded explicitly, never confused with the unwrapped Swift implementation.
    if p.name=="_contextlib.py" and p.parent.name=="utils" and p.parent.parent.name=="torch":
        record.update(package="torch",official_wheel_bytes_equal=None,
                      origin="Existing qualified DTK runtime; actual source SHA recorded, no new wheel claim")
        return record
    raise ValueError("Actual provider is outside declared runtime packages")


def function_chain(function):
    function=getattr(function,"__func__",function)
    result=[]
    seen=set()
    while True:
        if id(function) in seen or len(result)>=5 or not hasattr(function,"__code__"):
            raise ValueError("Unknown or cyclic native method wrapper chain")
        seen.add(id(function))
        row=file_identity(function.__code__.co_filename)
        row.update(qualname=function.__qualname__,code_filename=function.__code__.co_filename,
                   inspect_sourcefile=inspect.getsourcefile(function),wrapped=hasattr(function,"__wrapped__"))
        result.append(row)
        if not hasattr(function,"__wrapped__"):
            if row["package"] not in WHEELS:
                raise ValueError("Native implementation is not a frozen public provider")
            return result
        function=function.__wrapped__
