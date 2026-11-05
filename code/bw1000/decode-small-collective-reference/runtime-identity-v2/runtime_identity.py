"""Bind mapped runtime code to pinned files across the DTK overlay namespace."""
import ctypes
from hashlib import sha256
import os
from pathlib import Path
import re


def verify_runtime_maps(libraries, maps):
    families = {name.split(".so", 1)[0] for name in libraries}
    mapped = {family: [] for family in families}
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5].removesuffix(" (deleted)"))
        family = path.name.split(".so", 1)[0]
        if family not in families:
            continue
        if fields[5].endswith(" (deleted)"):
            raise ValueError("runtime mapping was deleted")
        begin, end = (int(value, 16) for value in fields[0].split("-"))
        mapped[family].append(dict(path=path, begin=begin, end=end, permissions=fields[1],
            offset=int(fields[2], 16), device=tuple(int(v, 16) for v in fields[3].split(":")),
            inode=int(fields[4]), range=fields[0]))
    result = {}
    for name, pin in sorted(libraries.items()):
        source = Path(pin["path"])
        if not source.is_absolute() or not re.fullmatch("[a-f0-9]{64}", pin["sha256"]):
            raise ValueError("invalid pinned runtime identity")
        expected = source.resolve(strict=True)
        info = expected.stat()
        namespace_device = (os.major(info.st_dev), os.minor(info.st_dev))
        entries = mapped[name.split(".so", 1)[0]]
        if not entries:
            raise ValueError("required runtime is not mapped: " + name)
        devices = {row["device"] for row in entries}
        if len(devices) != 1:
            raise ValueError("multiple backing devices for runtime: " + name)
        for row in entries:
            if row["path"].resolve(strict=True) != expected or not row["path"].samefile(expected):
                raise ValueError("foreign runtime mapping path: " + name)
            if row["inode"] != info.st_ino:
                raise ValueError("runtime mapping inode differs: " + name)
        device = next(iter(devices))
        overlay = device != namespace_device
        if overlay and not (expected.is_relative_to("/opt/dtk") and device[0] == namespace_device[0] == 0):
            raise ValueError("unqualified runtime device identity difference: " + name)

        digest = sha256()
        with expected.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size):
                raise ValueError("runtime file changed during open: " + name)
            for chunk in iter(lambda: stream.read(1048576), b""):
                digest.update(chunk)
        if digest.hexdigest() != pin["sha256"]:
            raise ValueError("runtime file SHA256 differs: " + name)
        executable = []
        for row in entries:
            if "x" not in row["permissions"]:
                continue
            if "r" not in row["permissions"]:
                raise ValueError("runtime executable mapping is unreadable")
            count = min(row["end"] - row["begin"], info.st_size - row["offset"])
            if not 0 < count <= 64 * 1024**2:
                raise ValueError("runtime executable mapping size is outside the bounded check")
            with expected.open("rb") as stream:
                stream.seek(row["offset"])
                original = stream.read(count)
            actual = ctypes.string_at(row["begin"], count)
            if len(original) != count or actual != original:
                raise ValueError("mapped runtime executable code differs from pinned file: " + name)
            link = os.readlink(Path("/proc/self/map_files") / row["range"])
            if link != str(expected):
                raise ValueError("mapped runtime backing link differs: " + name)
            executable.append(dict(file_offset=row["offset"], bytes=count,
                                   mapped_and_file_SHA256=sha256(actual).hexdigest()))
        if not executable:
            raise ValueError("runtime lacks a verified executable mapping: " + name)
        final = expected.stat()
        if (final.st_dev, final.st_ino, final.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise ValueError("runtime file changed during mapping verification")
        result[name] = dict(receipt_path=str(source), path=str(expected), sha256=pin["sha256"],
            device=list(device), namespace_stat_device=list(namespace_device), inode=info.st_ino,
            overlay_device_difference=overlay, executable_mappings=executable)
    return result
