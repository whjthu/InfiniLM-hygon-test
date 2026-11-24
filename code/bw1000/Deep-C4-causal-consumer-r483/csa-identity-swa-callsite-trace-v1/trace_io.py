"""Consume the owner-bound native call-site trace in the CSA differential runner."""

import ctypes as C
from hashlib import sha256
import json
from pathlib import Path


WIDTH = 8
SYMBOL = "modern_deep_csa_commit_trace_{}_v1"
FIELDS = ("tag", "tokens", "requests", "commit_rows", "direct_aliases", "index_select_submissions",
          "store_calls_returned", "original_drain_returned")
SCHEMA = "bw1000.deep.csa.identity-swa-callsite.v1"


def need(ok, message):
    if not ok:
        raise ValueError(message)


class Trace:
    """Use the same worker thread and actual Runtime pointer as CsaAttention::forward."""

    def __init__(self, library):
        self.arm = getattr(library, SYMBOL.format("arm"))
        self.arm.argtypes, self.arm.restype = [C.c_void_p, C.c_uint64], C.c_int
        self.clear = getattr(library, SYMBOL.format("clear"))
        self.clear.argtypes, self.clear.restype = [C.c_void_p], C.c_int
        self.read = getattr(library, SYMBOL.format("read"))
        self.read.argtypes, self.read.restype = [C.c_void_p, C.POINTER(C.c_uint64), C.c_size_t], C.c_int

    def forward(self, runtime, tag, operation):
        need(type(tag) is int and 0 < tag < 2**64 and self.arm(runtime, tag) == 0,
             "original runtime trace is already armed or unbound")
        fields = (C.c_uint64 * WIDTH)()
        try:
            result = operation()
            need(self.read(runtime, fields, WIDTH) == 0, "original matching forward did not return a trace")
        except BaseException:
            self.clear(runtime)
            raise
        return result, list(fields)


def validate(fields, index, requests, flag):
    need(type(fields) is list and len(fields) == WIDTH and all(type(x) is int and x >= 0 for x in fields),
         "invalid native trace readout")
    need(type(index) is int and index >= 0 and type(flag) is int and flag in (0, 1)
         and type(requests) is list and 1 <= len(requests) <= 2, "invalid frozen CSA request plan")
    need(all(type(r) in (list, tuple) and len(r) == 3
             and all(type(x) is int for x in r) and r[0] in (0, 1)
             and 0 <= r[1] < 32768 and 0 < r[2] <= 32768 - r[1] for r in requests)
         and len({r[0] for r in requests}) == len(requests), "invalid frozen CSA row or past")
    tokens = sum(r[2] for r in requests)
    commit_rows = sum(min(r[2], 128) for r in requests)
    need(0 < tokens <= 4096, "invalid packed hidden length")
    expected_direct = int(flag == 1 and tokens == commit_rows)
    need(fields == [index + 1, tokens, len(requests), commit_rows, expected_direct, 1 - expected_direct, 1, 1],
         "actual call site, store return or final drain differs from the frozen plan")
    return dict(zip(FIELDS, fields, strict=True))


def write_trace(directory, index, requests, flag, fields):
    data = dict(schema=SCHEMA, index=index, requests=requests, flag=flag,
                trace=validate(fields, index, requests, flag))
    raw = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with (directory / f"{index:03d}-commit-route.json").open("xb") as stream:
        stream.write(raw)
    return dict(commit_IndexSelect_calls=data["trace"]["index_select_submissions"],
                commit_callsite_trace_SHA256=sha256(raw).hexdigest())


def verify_trace(directory: Path, index, requests, flag, step):
    path = directory / f"{index:03d}-commit-route.json"
    need(path.resolve(strict=True) == path, "noncanonical raw call-site record")
    with path.open("rb") as stream:
        raw = stream.read(4097)
    need(0 < len(raw) <= 4096 and sha256(raw).hexdigest() == step["commit_callsite_trace_SHA256"],
         "native call-site trace bytes differ")
    data = json.loads(raw)
    need(data["schema"] == SCHEMA and data["index"] == index and data["requests"] == requests
         and type(data["flag"]) is int and data["flag"] == flag and set(data["trace"]) == set(FIELDS),
         "native trace provenance differs from the frozen capture")
    values = [data["trace"][key] for key in FIELDS]
    trace = validate(values, index, requests, flag)
    need(trace["index_select_submissions"] == step["commit_IndexSelect_calls"],
         "capture counter differs from its native trace")
    return trace
