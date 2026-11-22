"""The same first-forward argument projection on the two platforms."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat


FINGERPRINT_SHA = "816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49"
REFERENCE_SHA = "a52d848f052dfc0ccb42335a7ffa6a1982723fc2a82064fc612d7e1bfca1d959"
REFERENCE_REQUEST = "031e04f24efa429595364e6f0da1e5b6"
REQUEST = "792589c80fa9486dbeb2c42e262332ee"
SCHEMA = "internvl_first_micro_argument_fields_v1"
WORLD_SIZE = 8
LIMIT = 64 * 1024
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,127}")
SHA256 = re.compile(r"[0-9a-f]{64}")
DTYPE = re.compile(r"torch\.[a-z0-9_]{1,40}")
CONTAINERS = {"dict", "tuple", "list"}
SCALARS = {"NoneType", "str", "bool", "int", "float"}
PRE_FORWARD_COUNTERS = {
    "model_forward_started": False,
    "optimizer_updates": 0,
    "raw_tensor_bytes_written": 0,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return type(value) is str and SHA256.fullmatch(value) is not None


def plain_json(raw):
    def unique(items):
        result = dict(items)
        require(len(result) == len(items), "Duplicate JSON key")
        return result
    return json.loads(raw, object_pairs_hook=unique)


def stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_small(path, expected=None, size=None, cap=LIMIT):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts
            and not any(item.is_symlink() for item in (path, *path.parents)),
            "Linked or relative diagnostic path")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap,
            "Diagnostic file extent differs")
    with path.open("rb") as stream:
        opened = stamp(os.fstat(stream.fileno()))
        # Windows reports different ctime meanings via path lookup and the open handle.
        compared = slice(None, 4) if os.name == "nt" else slice(None)
        require(opened[compared] == stamp(before)[compared], "File opened after replacement")
        raw = stream.read(cap + 1)
        require(stamp(os.fstat(stream.fileno())) == opened, "File changed during read")
    require(len(raw) == before.st_size and stamp(path.lstat()) == stamp(before)
            and (size is None or len(raw) == size)
            and (expected is None or sha(raw) == expected), "Diagnostic file identity differs")
    return raw


def write_once(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    require(len(raw) <= LIMIT, "Diagnostic output exceeds metadata limit")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": len(raw), "SHA256": sha(raw)}


def first_reference(path):
    raw = read_small(path, expected=REFERENCE_SHA, size=25357)
    original = plain_json(raw)
    require(original["schema"] == "native_first_two_inputs_v1" and original["case"] == "T03.sft"
            and original["reference_request_id"] == REFERENCE_REQUEST
            and original["fingerprint_source_SHA256"] == FINGERPRINT_SHA
            and original["world_size"] == WORLD_SIZE and original["accumulation"] == 4
            and original["projection"] == ["args", "kwargs"] and len(original["rows"]) == 64,
            "Original aggregate reference scope differs")
    rows = {}
    for index, row in enumerate(original["rows"]):
        rank, offset = divmod(index, 8)
        require(row["rank"] == rank and row["micro_ordinal"] == offset + 1
                and row["global_step"] == offset // 4 and valid_sha(row["input_fingerprint_SHA256"])
                and valid_sha(row["file_SHA256"]), "Original aggregate reference roster differs")
        if offset == 0:
            rows[rank] = row
    return rows


def describe(args, kwargs, torch, fingerprint, deadline):
    """Metadata for each address and the unchanged canonical digest of the entire input."""
    require(type(args) in (tuple, list) and type(kwargs) is dict, "Forward argument schema differs")
    projection = {"args": args, "kwargs": kwargs}
    rows = []

    def visit(value, path):
        deadline()
        require(len(rows) < 256 and len(path) <= 10, "Field inventory exceeds scope")
        kind = type(value).__name__
        row = {"path": path, "kind": kind}
        rows.append(row)
        if type(value) is dict:
            require(all(type(key) is str and IDENTIFIER.fullmatch(key) for key in value),
                    "Unexpected forward argument field name")
            row["children"] = len(value)
            for key in sorted(value):
                visit(value[key], [*path, key])
        elif type(value) in (list, tuple):
            row["children"] = len(value)
            for index, item in enumerate(value):
                visit(item, [*path, index])
        else:
            if isinstance(value, torch.Tensor):
                row.update(kind="tensor", dtype=str(value.dtype), shape=list(value.shape))
            else:
                require(kind in SCALARS, "Unsupported forward argument field type")
            row["SHA256"] = fingerprint(value)["SHA256"]

    visit(projection, [])
    aggregate = fingerprint(projection)["SHA256"]
    deadline()
    return aggregate, rows


def document(side, rank, aggregate, expected, rows):
    require(side in {"A100", "HIP"} and type(rank) is int and 0 <= rank < WORLD_SIZE
            and valid_sha(aggregate) and valid_sha(expected), "First-slot identity differs")
    return {
        "schema": SCHEMA, "request": REQUEST, "side": side, "rank": rank, "world_size": WORLD_SIZE,
        "global_step": 0, "micro_ordinal": 1, "fingerprint_source_SHA256": FINGERPRINT_SHA,
        "original_aggregate_manifest_SHA256": REFERENCE_SHA, "input_SHA256": aggregate,
        "expected_input_SHA256": expected, "input_matches_original": aggregate == expected,
        "fields": rows, **PRE_FORWARD_COUNTERS,
    }


def validated_fields(doc, side, rank, expected):
    actual = doc["input_SHA256"]
    rows = doc["fields"]
    require(doc == document(side, rank, actual, expected, rows)
            and (side != "A100" or actual == expected)
            and type(rows) is list and 3 <= len(rows) <= 256, "Rank metadata identity differs")
    fields, children = {}, {}
    for row in rows:
        path, kind = row["path"], row["kind"]
        require(type(path) is list and len(path) <= 10 and all(
            type(key) is str and IDENTIFIER.fullmatch(key) or type(key) is int and 0 <= key < 256
            for key in path), "Unexpected field address")
        key = tuple(path)
        require(key not in fields, "Duplicate field address")
        keys = {"path", "kind"}
        if kind in CONTAINERS:
            keys.add("children")
            require(type(row["children"]) is int and 0 <= row["children"] <= 256, "Container size differs")
            children[key] = []
        else:
            require(kind == "tensor" or kind in SCALARS, "Unexpected field kind")
            keys.add("SHA256")
            require(valid_sha(row["SHA256"]), "Field digest differs")
            if kind == "tensor":
                keys.update(("dtype", "shape"))
                require(type(row["dtype"]) is str and DTYPE.fullmatch(row["dtype"])
                        and type(row["shape"]) is list and len(row["shape"]) <= 16
                        and all(type(axis) is int and axis >= 0 for axis in row["shape"]),
                        "Tensor metadata differs")
        require(set(row) == keys, "Unexpected field metadata column")
        fields[key] = row
        if key:
            parent = key[:-1]
            require(parent in children, "Field parent is absent or terminal")
            index = key[-1]
            require((fields[parent]["kind"] == "dict") == (type(index) is str), "Child address differs")
            children[parent].append(index)
    require(fields[()]["kind"] == "dict" and set(children[()]) == {"args", "kwargs"}
            and fields[("args",)]["kind"] in {"tuple", "list"}
            and fields[("kwargs",)]["kind"] == "dict", "Projection roots differ")
    for path, indices in children.items():
        require(len(indices) == fields[path]["children"] and
                (fields[path]["kind"] == "dict" or indices == list(range(len(indices)))),
                "Incomplete or reordered field children")
    return fields


def compare_ranks(reference, target, rank, expected):
    left = validated_fields(reference, "A100", rank, expected)
    right = validated_fields(target, "HIP", rank, expected)
    differences = []
    for path in sorted(left.keys() | right.keys(), key=lambda key: json.dumps(key)):
        if path not in left or path not in right:
            differences.append({"path": path, "reason": "missing_on_A100" if path not in left else "missing_on_HIP"})
            continue
        fields = [name for name in ("kind", "dtype", "shape", "children", "SHA256")
                  if left[path].get(name) != right[path].get(name)]
        if fields:
            differences.append({"path": path, "reason": fields})
    return {"rank": rank, "input_matches_original": target["input_matches_original"],
            "A100_fields": len(left), "HIP_fields": len(right), "differences": differences}
