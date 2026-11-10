"""Measure the actual SDPA dispatch in two identical frozen SDXL replays."""

from collections import Counter
from contextlib import nullcontext
import ctypes as C
import gc
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys

WORK = Path("/t06-run")
MATH = "aten::_scaled_dot_product_attention_math"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def publish(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(torch):
    node = socket.gethostname().split(".")[0]
    require(node == "f11r4n11" and os.environ["SLURM_JOB_ID"] == "1087656", "foreign allocation")
    runtime = C.CDLL("libamdhip64.so")
    count = C.c_int()
    require(runtime.hipGetDeviceCount(C.byref(count)) == 0 and count.value == 1
            and torch.cuda.device_count() == 1 and bool(torch.version.hip),
            "SDXL comparison requires exactly one visible HIP device")
    pci = C.create_string_buffer(32)
    require(runtime.hipDeviceGetPCIBusId(pci, len(pci), 0) == 0, "missing HIP PCI identity")
    name = pci.value.decode().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", name):
        name = "0000:" + name
    require(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", name),
            "invalid HIP PCI identity")
    return dict(step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                node=node, pci=name, hip=torch.version.hip, visible_cards=count.value)


def replay(torch, arm):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    source = str(WORK / "source/tools/hygon_sdxl_first_step.py")
    output = WORK / f"replay-{arm}"
    policy = nullcontext() if arm == "default" else sdpa_kernel(SDPBackend.MATH)
    previous = sys.argv
    sys.argv = [source, "--reference", "/reference/reference.pt", "--policy", "/reference/policy.json",
                "--weights", "/model", "--output", str(output)]
    try:
        with policy, profile(activities=[ProfilerActivity.CPU]) as profiler:
            runpy.run_path(source, run_name="__main__")
    finally:
        sys.argv = previous
    events = Counter({row.key: row.count for row in profiler.key_averages()
                      if row.key.startswith(("aten::_scaled_dot_product", "aten::scaled_dot_product"))
                      or ("attention" in row.key.lower() and "backward" in row.key.lower())})
    selected = {name: count for name, count in events.items()
                if name.startswith("aten::_scaled_dot_product") and "backward" not in name.lower()}
    require(selected, "the profiler did not identify an executed SDPA implementation")
    if arm == "math":
        require(set(selected) == {MATH}, "the forced math SDPA route was not exclusive")
    gc.collect()
    torch.cuda.empty_cache()
    return dict(arm=arm, selected_forward_operators=selected, profiler_operator_counts=dict(events))


def main():
    import torch

    torch.set_num_threads(1)
    publish(WORK / "assigned.json", identity(torch))
    first = replay(torch, "default")
    publish(WORK / "route-default.json", first)
    if set(first["selected_forward_operators"]) == {MATH}:
        publish(WORK / "route-decision.json", dict(state="DEFAULT_ALREADY_MATH", compared=["default"]))
        return
    second = replay(torch, "math")
    publish(WORK / "route-math.json", second)
    publish(WORK / "route-decision.json", dict(state="DISTINCT_SDPA_ROUTES_EXECUTED",
                                               compared=["default", "math"]))


if __name__ == "__main__":
    main()
