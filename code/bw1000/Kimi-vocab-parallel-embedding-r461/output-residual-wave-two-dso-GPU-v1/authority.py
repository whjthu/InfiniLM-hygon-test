"""Bind the two-DSO trial to the actual source objects and passive CPU closures."""

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time


LABEL = "Kimi461-output-residual-wave-two-dso-GPU-v1"
REQUEST = "604638da75b54f94a77a2f402a856f68"
PARENT, NODE = "1103606", "f13r1n01"
CPU_NODE = "f12r2n13"
ORDER = ("full-tree", "wave32", "wave32", "full-tree", "full-tree", "wave32")
INPUT = "/refactor/results/Kimi461-output-residual-component-GPU-v1/inputs.json"
INPUT_SHA256 = "a2d7499f07e4e58eea68bf33b59e7e2794fd8314c0ec94fe53ffc1173c07724c"
FULL_LABEL = "Kimi461-original678-wave-passive-rocm-CPU-v1"
WAVE_LABEL = "Kimi461-original678-wave32-link-only-rocm-CPU-v1"
FULL_PROOF = ("proof-748.json", 1758, "16c46c04871edfb6bf8d5aab4340c6d833dc13d074703bcdceea3f7ba1ba3412")
WAVE_PROOF = ("proof-751.json", 1970, "b6d9333f3d49d030b9a6c48e4086df587ab93383247186165d39097432d90cd6")
FULL_RAW = (f"/refactor/results/{FULL_LABEL}/repair-result.json", 9549,
            "a6230f278fa7ef63831c3fc9425307adee3ead9e41e1ae9aa3b00c5b5ea49cd2")
WAVE_RAW = (f"/refactor/results/{WAVE_LABEL}/link-result.json", 12377,
            "bff4cfc62257ccc31469b41fb55eb135f2d80b43aabff22a3f3edd2bdfcb0ae2")
REF_330 = ("/refactor/candidates/Kimi461-output-residual-tile64-CPU-v1/CPU-result.json",
           "6d9421acefdb6091694ffe8429c3c87b90239e00eea9c95afb28a23844cb65e1")
DSO = {
    "full-tree": dict(path="/refactor/candidates/Kimi461-output-residual-wave-component-link-CPU-v2/full-tree.so",
                      SHA256="8fde97a0954117687a5e00d4026855f5aa3a3e3c6202290cbfe96ce4241224be", bytes=556104),
    "wave32": dict(path=f"/refactor/results/{WAVE_LABEL}/wave32.so",
                   SHA256="3159a438d23e1cb3b3b263ef422652dd1960e4eecc3824454166763e96d84802", bytes=556104),
}
CABI = {"kimi_output_residual_component_r461", "kimi_output_residual_error_r461"}
NUMA = "/.singularity.d/libs/libnuma.so.1"
NUMA_SHA256 = "4a0ebd05ec13854e7a875d8187d0153b57188f2a247ae20025e09a31075261d1"
CUDA_SHA256 = "ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e"
PRELOAD = "/refactor/candidates/blas-policy-r271/libblas_policy_r271.so"
PRELOAD_SHA256 = "7d3a235f5576adeafbf1ead1aa3443bfa6f5d2b108944a954b7a487301256a45"
ORIGINAL_GRAPH_BYTES = 6644398160
LDD_MAPPED = re.compile(r"\s*(?:\S+\s+=>\s+)?(?P<path>/\S+)\s+\(0x[0-9a-f]+\)\s*")
LDD_VDSO = re.compile(r"\s*linux-vdso\.so\.1\s+\(0x[0-9a-f]+\)\s*")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def pinned(path, pin, size=None, limit=4 * 1024**2):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and (size is None or len(raw) == size) and sha256(raw).hexdigest() == pin,
         f"exact evidence bytes differ: {Path(path).name}")
    return raw


def host_path(root, path):
    return root.joinpath(*PurePosixPath(path).relative_to("/refactor").parts)


def allocation(raw, *, gpu=False):
    fields = {}
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        if not sep or not value or key in fields:
            return False
        fields[key] = value
    gres = {key: value for key, value in fields.items() if key.startswith("gres/")}
    return fields.get("node") == "1" and gres == ({"gres/dcu": "1"} if gpu else {})


def terminal(row, step, label, node=NODE, *, gpu=False):
    need(len(row) == 7 and row[:4] == [step, label, "COMPLETED", "0:0"]
         and re.fullmatch(re.escape(PARENT) + r"\.\d+", step) and row[5] == node
         and row[4].isdecimal() and allocation(row[6], gpu=gpu), "actual completed numbered Slurm step differs")


def require_cpu_steps(rows, directory):
    full_name, full_bytes, full_pin = FULL_PROOF
    wave_name, wave_bytes, wave_pin = WAVE_PROOF
    full = json.loads(pinned(directory / full_name, full_pin, full_bytes))
    wave = json.loads(pinned(directory / wave_name, wave_pin, wave_bytes))
    for step, label, expected in (("1103606.748", FULL_LABEL, full["accounting"][0]),
                                  ("1103606.751", WAVE_LABEL, wave["actual_accounting"][0])):
        owned, = [row for row in rows if len(row) == 7 and row[0] == step]
        terminal(owned, step, label, CPU_NODE)
        need(owned == expected and dict(item.split("=") for item in owned[6].split(",")).get("cpu") == "4",
             "fresh accounting differs from independently consumed link or passive audit")
    need(full["independent_cleanup"]["post"] == "1103606.749"
         and full["independent_cleanup"]["same_PCI"] is True and full["independent_cleanup"]["kfd_processes"] == 0
         and full["independent_cleanup"]["own_device_fd_processes"] == 0
         and wave["independent_pre_post"] == dict(pre="1103606.750", step="1103606.752", same_pci=True, kfd=0, fd=0),
         "original passive/link post evidence differs")
    return {"full_tree": full_pin, "wave32": wave_pin}


def check_raw(full, wave, qualified):
    need(full["source_step"] == "1103606.678" and full["source_state"] == "FAILED"
         and full["observed_step"] == "1103606.748" and full["passive_loader_calls"] == 1
         and full["loader_exit_code"] == 0 and full["original_dso_rechecked"] is True
         and full["original_dso_SHA256"] == DSO["full-tree"]["SHA256"], "actual .748 original artifact differs")
    wave_obj = wave["wave32_DSO"]
    need(wave["step"] == "1103606.751" and wave["link_calls"] == 1
         and wave["passive_full_tree_receipt_SHA256"] == FULL_RAW[2] and wave["original_678_state"] == "FAILED"
         and wave["original_full_tree_SHA256"] == DSO["full-tree"]["SHA256"] and wave["completed"] is True
         and wave["compiled_HOST_objects"] == wave["compiled_HIP_objects"] == 0
         and {key: wave_obj[key] for key in ("path", "bytes", "SHA256")} == DSO["wave32"]
         and CABI <= set(wave_obj["exports"]), "actual .751 wave artifact differs")
    for receipt in (full, wave):
        need(receipt["node"] == CPU_NODE and receipt["passed"] is receipt["original_path_rule_pass"]
             is receipt["original_digests_complete"] is True
             and receipt["GPU_calls"] == receipt["native_application_loads"] == 0
             and receipt["provider_bytes_read"] == ORIGINAL_GRAPH_BYTES
             and receipt["loaded_count"] == receipt["provider_count_hashed"] == 68 and not receipt["byte_mismatches"],
             "reference did not verify the exact 68-provider graph")
    providers = set(full["aliases"].values())
    need(len(providers) == 68 and providers == set(wave["aliases"].values()), "exact reference provider paths differ")
    cuda, = [name for name in providers if PurePosixPath(name).name.startswith("libcudart.so.12")]
    eligible = {**qualified, cuda: CUDA_SHA256}
    need(providers <= eligible.keys() and NUMA in providers and eligible[NUMA] == NUMA_SHA256,
         "qualified ROCm provider digests differ")
    return {path: eligible[path] for path in sorted(providers)}


def qualify(root, directory, original_rows):
    proofs = require_cpu_steps(original_rows, directory)
    full = json.loads(pinned(host_path(root, FULL_RAW[0]), FULL_RAW[2], FULL_RAW[1]))
    wave = json.loads(pinned(host_path(root, WAVE_RAW[0]), WAVE_RAW[2], WAVE_RAW[1]))
    prior = json.loads(pinned(host_path(root, REF_330[0]), REF_330[1]))
    need(prior["actual_CPU"] == "1103606.330" and prior["passed"] is prior["completed"] is True,
         "original .330 reference differs")
    providers = check_raw(full, wave, prior["actual_passive_providers"])
    for library in DSO.values():
        pinned(host_path(root, library["path"]), library["SHA256"], library["bytes"])
    pinned(host_path(root, INPUT), INPUT_SHA256, limit=32768)
    return dict(request=REQUEST, libraries=DSO, providers=providers, input_receipt_SHA256=INPUT_SHA256,
                build_proofs=proofs)


def runtime_graph(text, resolve, providers, preload):
    paths = set()
    for line in text.splitlines():
        if not line.strip() or LDD_VDSO.fullmatch(line):
            continue
        match = LDD_MAPPED.fullmatch(line)
        need(match, "unexpected GPU-environment passive diagnostic")
        paths.add(resolve(match["path"]))
    need(paths == set(providers) | {preload}, "GPU environment selected different canonical providers")
    return paths


def runtime_authority(host_receipt, preload_env, time_remaining):
    need(host_receipt["libraries"] == DSO and host_receipt["request"] == REQUEST
         and host_receipt["input_receipt_SHA256"] == INPUT_SHA256 and preload_env == PRELOAD,
         "GPU provider and input authority differs")
    providers = host_receipt["providers"]
    need(len(providers) == 68 and providers.get(NUMA) == NUMA_SHA256,
         "detected drift in GPU expected providers")
    resolve = lambda value: str(Path(value).resolve(strict=True))
    preload_path = resolve(PRELOAD)
    for library in DSO.values():
        path = library["path"]
        need(resolve(path) == path, "component library cannot be aliased")
        pinned(path, library["SHA256"], library["bytes"])
        budget = min(25, time_remaining() - 20)
        need(budget > 0, "passive provider window expired")
        probe = subprocess.run(["/usr/bin/ldd", "-r", path], capture_output=True, check=True, timeout=budget,
                               env={**os.environ, "LC_ALL": "C"})
        need(not probe.stderr and 0 < len(probe.stdout) <= 32768, "GPU environment passive loader failed")
        runtime_graph(probe.stdout.decode("utf8"), resolve, providers, preload_path)
    total = 0
    extra_preload_bytes = 0
    for path, pin in {**providers, preload_path: PRELOAD_SHA256}.items():
        need(path not in providers or providers[path] == pin, "BLAS preload and original provider pins differ")
        digest = sha256()
        with Path(path).open("rb") as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024**2), b""):
                total += len(chunk)
                if path not in providers:
                    extra_preload_bytes += len(chunk)
                need(time_remaining() > 20 and total <= 8 * 1024**3, "GPU provider digest budget expired")
                digest.update(chunk)
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            need(all((row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns) == identity
                     for row in (os.fstat(stream.fileno()), Path(path).stat())), "GPU provider changed while hashing")
        need(digest.hexdigest() == pin, "GPU environment provider bytes differ")
    need(total - extra_preload_bytes == ORIGINAL_GRAPH_BYTES, "GPU provider byte denominator differs")
    return dict(passive_objects=2, qualified_providers=68, provider_bytes=ORIGINAL_GRAPH_BYTES, preload=preload_path)
