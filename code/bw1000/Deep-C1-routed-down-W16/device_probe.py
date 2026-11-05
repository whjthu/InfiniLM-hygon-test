"""Explicit one-node component entry. Publication and Slurm controller are external."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import os
from pathlib import Path
import socket
import threading
from time import monotonic, perf_counter, time

from component_probe import FIXTURE_SHA, compare, digest, expert_plan, load_bank, read_fixtures, summarize

NATIVE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
OBJECT = "08ed7c8bfb7fe56cee66a6d6bdc5a5d569b05574ae68b2e7741d755a563ea756"
SOURCE = "e64eb976cd44760b3a01af5655e51e601dabf743fd78bf6a1c07cafcc1931c9a"
PREFIX = "deep_routed_w16_component_"
P = C.c_void_p
I = C.c_int
POSTLOAD = 6 << 30
FORWARD = 512 << 20


def read_pinned(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"artifact hash differs: {path.name}")
    return raw


def bind(build, deps, isolated_down=None):
    if (build["source_SHA256"], build["baseline_object_SHA256"],
            build["baseline_native_SHA256"]) != (SOURCE, OBJECT, NATIVE):
        raise ValueError("unbound C1 build")
    library = Path(build["library"]).resolve()
    read_pinned(library, build["library_SHA256"])
    for name, sha in deps.items():
        read_pinned(Path(name), sha)
    lib = C.CDLL(str(library))

    class DlInfo(C.Structure):
        _fields_ = [("fname", C.c_char_p), ("base", P), ("sname", C.c_char_p), ("address", P)]

    dladdr = C.CDLL(None).dladdr
    dladdr.argtypes = [P, C.POINTER(DlInfo)]
    dladdr.restype = I
    providers = {}
    for symbol in (PREFIX + "forward", "deep_routed_down_w16_c1",
                   "deep_routed_dot_r20", "deep_routed_validate_r20"):
        info = DlInfo()
        if not dladdr(C.cast(getattr(lib, symbol), P), C.byref(info)):
            raise ValueError(f"unresolved component symbol: {symbol}")
        actual = Path(os.fsdecode(info.fname)).resolve()
        expected = Path(build["baseline_native"] if symbol.endswith("_r20") else library).resolve()
        if actual != expected:
            raise ValueError(f"wrong component provider: {symbol}")
        sha = NATIVE if symbol.endswith("_r20") else build["library_SHA256"]
        read_pinned(actual, sha)
        providers[symbol] = sha

    funcs = {}
    for name, args, restype in (
        ("create", [I], P), ("forward", [P, I, C.POINTER(P)], I),
        ("destroy", [P, C.POINTER(I)], I), ("error", [], C.c_char_p),
        ("report", [], C.c_char_p),
    ):
        fn = funcs[name] = getattr(lib, PREFIX + name)
        fn.argtypes = args
        fn.restype = restype
    if isolated_down is not None:
        funcs["isolated_down"] = isolated_down.bind(lib)
    return funcs, providers


def check_claim(raw):
    claim = json.loads(raw)
    host = socket.gethostname().split(".")[0]
    parent = os.environ.get("SLURM_JOB_ID")
    step = os.environ.get("SLURM_STEP_ID")
    if (claim.get("GPU_GO") is not True or not parent or not step or not step.isdecimal()
            or (claim["parent"], claim["node"], claim["cards"], claim["fixture_SHA256"])
            != (parent, host, 8, FIXTURE_SHA)):
        raise ValueError("matching GPU claim and actual Slurm step required")
    limit = claim["remaining_deadline_unix"] - time()
    if not 0 < limit <= claim["max_wall_s"] <= 600:
        raise ValueError("invalid original deadline")
    return claim, monotonic() + limit, parent + "." + step


def run(claim_raw, fixture_dir, result_path):
    claim, deadline, step = check_claim(claim_raw)
    build_raw = read_pinned(Path(claim["build_receipt"]), claim["build_receipt_SHA256"])
    build = json.loads(build_raw)
    rows = read_fixtures(fixture_dir)
    isolated_down = None
    down_inputs = None
    if claim.get("isolated_down") is True:
        read_pinned(Path(__file__), claim["device_probe_SHA256"])
        read_pinned(Path(__file__).with_name("down_only_probe.py"), claim["down_only_probe_SHA256"])
        import down_only_probe as isolated_down

        down_inputs = isolated_down.read_inputs(fixture_dir)
    model = Path("/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8")
    plans = {rank: expert_plan(model, rows[rank]) for rank in range(8)}
    import torch
    from safetensors import safe_open

    funcs, providers = bind(build, build["deps"], isolated_down)

    torch.set_num_threads(1)
    if torch.cuda.device_count() != 8:
        raise RuntimeError("expected eight devices on one node")
    cpu_set = sorted(os.sched_getaffinity(0))
    if os.environ.get("INFINILM_OWNER_CPU_WIDTH_R2392") != "4" or len(cpu_set) < 32:
        raise RuntimeError("component requires the original four-core owner binding")
    barrier = threading.Barrier(8)
    owners = {}
    report = dict(actual_step=step, node=claim["node"], completed=False, providers=providers, owners=owners,
                  claim_SHA256=digest(claim_raw), build_receipt_SHA256=digest(build_raw),
                  fixture_SHA256=FIXTURE_SHA, scope="Eight rank slices, independent component, no TP collective")

    def left():
        value = deadline - monotonic()
        if value <= 0:
            raise TimeoutError("original component deadline exhausted")
        return value

    def owner(rank):
        record = owners[rank] = dict(rank=rank, normal_owner_closed=False)
        ctx = None
        try:
            torch.cuda.set_device(rank)
            props = torch.cuda.get_device_properties(rank)
            if ("gfx936" not in props.gcnArchName or props.multi_processor_count != 80
                    or torch.cuda.mem_get_info(rank)[0] < POSTLOAD):
                raise RuntimeError("unexpected device or pre-create floor")
            left()
            begin = perf_counter()
            banks, weight_sha = load_bank(plans[rank], rank, torch, safe_open)
            record.update(weight_slice_SHA256=weight_sha, CPU_load_s=perf_counter() - begin)
            if isolated_down is not None:
                start = perf_counter()
                rank_inputs = [down_inputs[rank, past] for past in isolated_down.PASTS]
                references = [isolated_down.cpu_reference(row, banks[2].numpy(), banks[3].numpy())
                              for row in rank_inputs]
                record["isolated_down_CPU_reference_s"] = perf_counter() - start
            begin = perf_counter()
            device = f"cuda:{rank}"
            banks = [t.to(device) for t in banks]
            fixtures = []
            output = torch.empty((1, 4096), device=device, dtype=torch.float32)
            for _, arrays in rows[rank]:
                hidden, ids, weights = (torch.from_numpy(a) for a in arrays)
                operands = [hidden.view(torch.bfloat16).to(device), ids.to(device),
                            weights.to(device), *banks, output]
                fixtures.append((operands, (P * 8)(*[a.data_ptr() for a in operands])))
            torch.cuda.synchronize(rank)
            record["H2D_s"] = perf_counter() - begin
            left()
            ctx = funcs["create"](rank)
            if not ctx:
                raise RuntimeError(funcs["error"]().decode())
            postload_free = torch.cuda.mem_get_info(rank)[0]
            if postload_free < POSTLOAD:
                raise RuntimeError("postload component floor")
            record["postload_free_bytes"] = postload_free
            record["minimum_forward_free_bytes"] = postload_free

            def align():
                barrier.wait(timeout=min(30, left()))

            def floor_check():
                free = torch.cuda.mem_get_info(rank)[0]
                if free < FORWARD:
                    raise RuntimeError("forward memory floor")
                record["minimum_forward_free_bytes"] = min(record["minimum_forward_free_bytes"], free)

            def forward(index, mode):
                align()
                begin = perf_counter()
                code = funcs["forward"](ctx, mode, fixtures[index][1])
                wall = perf_counter() - begin
                if code:
                    raise RuntimeError(funcs["error"]().decode())
                if os.sched_getaffinity(0) != set(cpu_set[4 * rank:4 * rank + 4]):
                    raise RuntimeError("unexpected owner affinity")
                native = json.loads(funcs["report"]())
                result = output.cpu().numpy()
                floor_check()
                return native, result, wall

            record["trials"] = []
            compare(forward, record["trials"])
            if isolated_down is not None:
                record["isolated_down_trials"] = isolated_down.run_owner(
                    torch, rank, rank_inputs, references, banks, funcs["isolated_down"], align, floor_check)
        except BaseException as error:
            record["error_type"] = type(error).__name__
            barrier.abort()
            raise
        finally:
            if ctx:
                expired = I()
                rc = funcs["destroy"](ctx, C.byref(expired))
                record["destroy_RC"] = rc
                record["normal_owner_closed"] = rc == 0 and expired.value == 1
                if not record["normal_owner_closed"]:
                    barrier.abort()
                    raise RuntimeError("Runtime did not close")

    with result_path.open("x", encoding="utf8") as out:
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(owner, range(8)))
            report["paired_max_owner"] = summarize({r: o["trials"] for r, o in owners.items()})
            if isolated_down is not None:
                report["isolated_down"] = dict(
                    probe_SHA256=claim["down_only_probe_SHA256"],
                    scope="Single down launch on PyTorch stream with original shadow quantization inputs; no TP collective",
                    paired_max_owner=isolated_down.summarize(
                        {r: o["isolated_down_trials"] for r, o in owners.items()}),
                )
            report["completed"] = True
        finally:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("claim", "fixture", "result"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.claim.read_bytes(), args.fixture, args.result)
