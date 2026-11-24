"""Derive Gemma's private train-only collator index on a separately admitted CPU step."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time


SOURCE = Path("/audit-source")
OVERLAY = Path("/candidate-source/gemma_vizwiz_two_updates_inputs.py")
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
INDEX_HELPER = (6487, "f0f181c029d8f63f0e56b2aa84958daeb241a38d4342356951311aa7a55b0469")
PINS = {
    "FRAMEWORK_FILES.json": (112315, "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"),
    "MODEL_PROCESSOR_FILES.json": (2107, "eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b"),
    "audit_gemma_training_native.py": (9368, "890b4ebcea30972f604ffaa1d4b7f5cca2b76e919722944123fa3368fd29b1ba"),
    "build_gemma_train_exposure.py": (4747, "c0c380c51ed4af09779fa270b4419ce511571137d95f1676b3dd9a4eaad715a7"),
    "inside.py": (8759, "c72024c88c2e3665a0a9b91ac20177e4af61c456740fdd9d3292b8a0763addb3"),
    "stage_receiver.py": (6479, "de9408fcde0835d030ed07b43a4562ab1fe37ca1ca5a0c0a0ca8d088fcdfc7d6"),
}
STREAMS = (
    ("training-gemma-vizwiz-five-native-input-audit-v1", 2164,
     "be03f5443f6158f1b8358410632100b88cc15e0bba6649ecebd54648cbb88717"),
    ("training-gemma-vizwiz-reviewed-train-native-v1", 3022,
     "3f9051bb78a71f49d1e8ef51b06e706c164ff01a7650bc80a15154182d9e2c66"),
)
ORIGINAL_WORKERS = (
    ("training-gemma-vizwiz-five-native-CPU-v1", 514,
     "910392d04b64a5286914f804c8fc644d4e43f9da2559de24c6ce1e76e48ba921", "1103606.2882",
     1959, "638d829d03fe71b6a45c5348207784ff29ac13bff3947ec28ce156f030545829",
     "9d780defe37f48ffbdb56fba0e528fd1", 5),
    ("training-gemma-vizwiz-reviewed-train-native-CPU-v1", 380,
     "90aa47f730c41c26877bb67169db61be6c7c3f10e975f769e084b71b738e11c2", "1103606.2883",
     1944, "97c517386eb307ee393faec9b76dd01d4168a922dad44d625628af3bcd96afba",
     "8bf59e75c0ef44198b61944b5f011afd", 7),
)
VIEW = "cebeecad13db010a06f2ae7ef48260c6b26cc5ad42ea4993f6b8a60849ce1ad4"
LIMIT = 120


def require(ok, gate):
    if not ok:
        raise ValueError(gate)


def read(path, size, digest):
    require(path.is_file() and path.resolve(strict=True) == path, "EXACT_READONLY_SOURCE_PATH")
    maximum = 65536 if size is None else size
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    require(0 < len(raw) <= maximum and (size is None or len(raw) == size)
            and sha256(raw).hexdigest() == digest, "EXACT_READONLY_SOURCE_BYTES")
    return raw


def bind_cpu(parent, node, epoch, now, env, hostname, cpus):
    step = env.get("SLURM_STEP_ID", "")
    tres = env.get("SLURM_TRES_PER_TASK", "").lower()
    require(parent == env.get("SLURM_JOB_ID") and re.fullmatch("[0-9]+", parent)
            and hostname == node and re.fullmatch("f[0-9]{2}r[0-9]+n[0-9]{2}", node)
            and re.fullmatch("[0-9]+", step) and cpus == 4 and env.get("SLURM_CPUS_PER_TASK") == "4",
            "EXACT_NUMERIC_CPU_STEP")
    require(env.get("SLURM_STEP_GPUS", "") in ("", "(null)")
            and env.get("SLURM_GPUS_ON_NODE", "") in ("", "0")
            and env.get("SLURM_GPUS_PER_TASK", "") in ("", "0")
            and not any(device in tres for device in ("gpu", "dcu"))
            and all(env.get(key) == "-1" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
            and all(env.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")), "NO_GPU_OFFLINE_STEP")
    require(0 < epoch <= now < epoch + LIMIT, "ORIGINAL_CPU_EPOCH")
    return parent + "." + step


def load(name, path, raw):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), vars(module))
    return module


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(raw) <= 16384, "PRIVATE_INDEX_EXTENT")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(raw)


class CountedTemplate:
    def __init__(self, template, progress, check_deadline):
        self.template = template
        self.progress = progress
        self.check_deadline = check_deadline

    def encode(self, row):
        self.check_deadline()
        encoded = self.template.encode(row)
        self.progress["native_encodes_completed"] += 1
        return encoded

    def data_collator(self, rows):
        self.check_deadline()
        collated = self.template.data_collator(rows)
        self.progress["native_collations_completed"] += 1
        return collated


def expire(*_args):
    raise TimeoutError("CPU index deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--epoch", required=True, type=float)
    parser.add_argument("--view", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()
    failure, summary_path = Path("/index-out/FAILED.json"), Path("/index-out/RESULT.json")
    require(args.index == Path("/index-out/native-index.json")
            and not any(path.exists() or path.is_symlink() for path in (args.index, failure, summary_path)),
            "INDEX_OUTPUT_EXCLUSIVITY")
    def check_deadline():
        require(time.time() < args.epoch + LIMIT, "ORIGINAL_CPU_EPOCH")

    phase, step = "worker_placement", None
    progress = {"native_encodes_completed": 0, "native_collations_completed": 0}
    try:
        step = bind_cpu(args.parent, args.node, args.epoch, time.time(), os.environ,
                        socket.gethostname().split(".")[0], len(os.sched_getaffinity(0)))
        signal.signal(signal.SIGALRM, expire)
        alarm = args.epoch + LIMIT - time.time()
        require(alarm > 0, "ORIGINAL_CPU_EPOCH")
        signal.setitimer(signal.ITIMER_REAL, alarm)
        phase = "original_provenance"
        files = {name: read(SOURCE / name, size, digest) for name, (size, digest) in PINS.items()}
        for stage, size, digest, observed_step, out_size, out_sha, request, count in ORIGINAL_WORKERS:
            original_root = RESULTS / stage
            output = json.loads(read(original_root / "outcome.json", out_size, out_sha))
            row = output["terminal"]["row"]
            witness = json.loads(read(original_root / "evidence/worker/result.json", size, digest))
            complete = witness.get("checked", witness.get("complete_native_rows"))
            require(row[0] == observed_step and row[2:4] == ["COMPLETED", "0:0"]
                    and row[6] == "cpu=4,mem=8G,node=1"
                    and output["passed"] is True and output["request"] == request and output["GPU_cards"] == 0
                    and witness["passed"] is True and witness["model_loaded"] is False
                    and witness["optimizer_updates"] == 0 and complete == count,
                    "ORIGINAL_NATIVE_CPU_STEP_IDENTITY")
        helper = load("original_gemma_index_helper", OVERLAY, read(OVERLAY, *INDEX_HELPER))
        sys.path.insert(0, str(SOURCE))
        original = load("original_2883_training_template", SOURCE / "inside.py", files["inside.py"])
        sys.path.pop(0)
        require(Path(original.original.__file__).resolve() == SOURCE / "audit_gemma_training_native.py"
                and Path(original.stage.__file__).resolve() == SOURCE / "stage_receiver.py", "ORIGINAL_PROVIDER_IDENTITY")
        require({name: importlib.metadata.version(name) for name in original.original.VERSIONS} == original.original.VERSIONS
                and "torch" not in sys.modules, "ORIGINAL_RUNTIME_IDENTITY")
        import torch
        require(not torch.cuda.is_initialized(), "CPU_GPU_CONTEXT")

        def forbid(*_args, **_kwargs):
            raise RuntimeError("device initialization forbidden during index derivation")

        torch.cuda._lazy_init = forbid
        torch.set_num_threads(2)
        claim = {"processor_manifest": pin(files["MODEL_PROCESSOR_FILES.json"]),
                 "framework_manifest": pin(files["FRAMEWORK_FILES.json"])}
        phase = "original_cpu_template"
        _, capped, _, _, _ = original.load_template(claim, torch, check_deadline)
        phase = "frozen_training_view_and_media"
        read(args.view, 345722, VIEW)
        roles = [read(RESULTS / stage / "data/train-swift.jsonl", size, digest) for stage, size, digest in STREAMS]
        phase = "native_train_only_collator_index"
        observed = CountedTemplate(capped, progress, check_deadline)
        index = helper.derive_index(observed, *roles, VIEW)
        require(set(progress.values()) == {12}, "NATIVE_INDEX_DENOMINATOR")
        require(not torch.cuda.is_initialized(), "CPU_GPU_CONTEXT")
        check_deadline()
        output = save(args.index, index)
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        summary = save(failure, {"state": "FAILED", "phase": phase, "CPU_step": step,
                                 "error_type": type(error).__name__, "planned_train_rows": 12,
                                 "calibration_roles": 0, "validation_roles": 0, "test_roles": 0,
                                 "qualified_index_rows": 0, "model_loads": 0, "optimizer_updates": 0,
                                 "GPU": 0, **progress})
        print(json.dumps({"state": "FAILED", "private_failure_receipt": summary}))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result = {"state": "EXACT_NATIVE_TRAIN_INDEX_WRITTEN", "CPU_step": step, "private_index": output,
              "planned_train_rows": 12, "native_source_rows": 12, "train_roles": 12, "calibration_roles": 0,
              "validation_roles": 0, "test_roles": 0, "model_loads": 0, "optimizer_updates": 0, "GPU": 0, **progress}
    summary = save(summary_path, result)
    print(json.dumps({"state": result["state"], "CPU_step": step, "private_index": output, "receipt": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
