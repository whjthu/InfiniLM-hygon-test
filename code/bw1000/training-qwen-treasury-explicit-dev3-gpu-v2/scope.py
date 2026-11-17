"""Frozen independent single-device evaluation placement and original prerequisites."""
import hashlib
import json
import math
from pathlib import Path
import stat

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
LABEL = "training-qwen-treasury-explicit-dev3-gpu-v2"
PACKAGE = RESULTS / LABEL
SRC = PACKAGE / "source"
CPU = RESULTS / "training-qwen-treasury-explicit-dev-CPU-v1"
MEDIA = RESULTS / "training-qwen-treasury-dev-native-CPU-v1/dev/media"
MODEL = RESULTS / "training-qwen-original-public-model-v1/model"
ARMS = {
    "base": dict(parent="1087656", node="a01r4n14", adapter=None),
    "old_domain": dict(parent="1087656", node="a01r4n16", adapter=(RESULTS / "training-qwen-treasury-sft-v3/continuous/train/checkpoint-5").as_posix()),
    "domain": dict(parent="1103606", node="f13r1n10", adapter=(RESULTS / "training-qwen-treasury-explicit-sft-v1/continuous/train/checkpoint-5").as_posix()),
}
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
INVENTORY_SHA = "07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"
CPU_PINS = {
    "intent.json": "6c0ea879c9d884f3938482b8ee613cd51233182420f742de5e37fd35f36d1e0c",
    "outcome.json": "e10122997c1418eed20431f56419b9242e90428f16f52d1372fe231ee3600ee0",
    "step.log": "5fc38b966f4c121a1f1ad961621109d35b605a5e0b4df2adb70eef9bd26a31b8",
    "source/SOURCE_FILES.json": "c0efd831d4882f1b4dc7fbc1efed49a96fb3093abdefd56cf5d371cb9f292e9a",
    "inventory.json": INVENTORY_SHA,
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 1024**2, "unsafe original metadata")
    return json.loads(path.read_bytes())


def pin(path, limit=32 * 1024**3):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "unsafe original file")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            h.update(data)
    after = path.lstat()
    require(all(getattr(before, x) == getattr(after, x) for x in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "original file changed while hashing")
    return dict(bytes=before.st_size, SHA256=h.hexdigest())


def seconds_left(deadline, now, requested, reserve=0):
    require(all(math.isfinite(x) for x in (deadline, now, requested, reserve)) and requested > 0 and reserve >= 0,
            "invalid original deadline")
    available = deadline - now - reserve
    require(available > 0, "original epoch exhausted")
    return min(requested, available)


def stage_limits(deadline, now, phase):
    gpu = phase == "gpu"
    reserve = 180 if gpu or phase == "pre" else 10
    cap = 960 if gpu else 120
    slurm = 60 * (int(seconds_left(deadline, now, cap, reserve)) // 60)
    require(slurm >= 60, "no whole-minute Slurm budget remains")
    client = min(slurm + 10, seconds_left(deadline, now, 1000, 120 if gpu else 5))
    return dict(slurm_seconds=slurm, client_seconds=client,
                slurm_time=f"00:{slurm // 60:02d}:00")


def verify_prerequisites(accounting):
    for name, digest in CPU_PINS.items():
        require(pin(CPU / name, 4 * 1024**2)["SHA256"] == digest, "actual CPU330 original differs: " + name)
    cpu = read(CPU / "outcome.json")
    require(cpu == read(SRC / "CPU330.json") and cpu["passed"] is True
            and cpu["accounting"]["row"] == accounting("1087656", "training-qwen-treasury-explicit-dev-CPU-v1")
            and cpu["accounting"]["row"][:4] == ["1087656.330", "training-qwen-treasury-explicit-dev-CPU-v1", "COMPLETED", "0:0"]
            and cpu["metrics"]["dev_pins_SHA256"] == INVENTORY_SHA
            and all(cpu["metrics"]["arms"][x]["rows"] == 108 for x in ("visual", "text_only"))
            and cpu["metrics"]["actual_GPU_cards"] == cpu["metrics"]["actual_generations"] == 0,
            "actual CPU330 native input gate differs")
    for name, info in read(CPU / "source/SOURCE_FILES.json")["files"].items():
        require(Path(name).name == name and pin(CPU / "source" / name, 1024**2) == info, "CPU330 source member differs")
    verify_entry_prerequisites(accounting)
    authority = read(SRC / "ROOT_AUTHORITY.json")
    require(authority["model_run_verified"] is True and authority["independent_post_verified"] is True,
            "new328 root authority is not qualified")
    for path, digest in authority["original_artifact_pins"].items():
        actual = Path(path)
        require(actual.is_relative_to(RESULTS) and pin(actual, 4 * 1024**2)["SHA256"] == digest,
                "new328 actual original artifact differs")
    for arm in ("old_domain", "domain"):
        adapter = Path(ARMS[arm]["adapter"])
        info = (read(SRC / "OLD_ADAPTER.json") if arm == "old_domain" else authority["checkpoint5_files"])
        for name, identity in info.items():
            require(Path(name).name == name and pin(adapter / name, 256 * 1024**2) == identity, "actual step5 adapter differs")
        require(read(adapter / "trainer_state.json")["global_step"] == 5, "candidate is not step5")


def verify_model():
    for name, info in read(SRC / "MODEL_ASSETS.json")["model"].items():
        require(pin(MODEL / name) == dict(bytes=info["bytes"], SHA256=info["SHA256"]), "actual node-visible model differs")


ENTRY_GATE_LABEL = "training-qwen-treasury-dev3-v2-input-gate-CPU-v1"
ENTRY_GATE = RESULTS / ENTRY_GATE_LABEL


def verify_entry_records(binding, intent, outcome, probes, rows, inventory, sources):
    """Pure admission check; records must also pass the separate original-file pins."""
    require(binding["state"] == "THREE_ORIGINAL_ZERO_GPU_ENTRY_GATES_INDEPENDENTLY_VERIFIED"
            and binding["originals_passed"] is True and binding["actual_GPU_steps"] == 0
            and binding["native_data_qualified_for_declared_container_view"] is True,
            "three-node CPU entry authority is not qualified")
    require(binding["request"] == "136804b0d42744a0ba38f2d84b94a4bf"
            and binding["source_publication"] == "7cd5cd855abca76bc2fd2e632c7456c06143b308"
            and binding["epoch"] == 1794885086.3294418 and binding["deadline"] == binding["epoch"] + 600,
            "original CPU gate identity/epoch differs")
    require(intent["request"] == outcome["request"] == binding["request"]
            and intent["publication"] == binding["source_publication"]
            and intent["epoch"] == outcome["epoch"] == binding["epoch"]
            and intent["deadline"] == outcome["deadline"] == binding["deadline"]
            and outcome["passed"] is True and outcome["all_three_actual_entry_gates_passed"] is True
            and outcome["GPU_GO"] is False, "actual original CPU gate did not pass")
    require(set(binding["arms"]) == set(outcome["arms"]) == set(probes) == set(rows) == set(ARMS)
            and len(inventory) == 39, "CPU entry complete denominator differs")
    for arm, placement in ARMS.items():
        row, actual, expected = rows[arm], probes[arm], binding["arms"][arm]
        require(row == expected["actual_accounting"] == outcome["arms"][arm]["accounting"]["row"]
                and row[1] == ENTRY_GATE_LABEL + "-" + arm and row[2:4] == ["COMPLETED", "0:0"]
                and row[5] == placement["node"] and row[0].startswith(placement["parent"] + ".")
                and "gres/dcu=" not in row[6] and "gres/gpu=" not in row[6]
                and outcome["arms"][arm]["passed"] is True,
                "actual CPU entry accounting differs: " + arm)
        require(actual["arm"] == arm and actual["passed"] is True
                and actual["actual_node"] == placement["node"] and actual["actual_step"] == row[0]
                and actual["actual_GPU_cards"] == 0 and actual["torch_imported"] is False
                and actual["explicit_SDK_calls"] is False and actual["generate_called"] is False
                and actual["tokenize_called"] is False and actual["actual_native_entry"] == "native_generation.bound_inputs"
                and actual["visual_rows"] == actual["text_only_rows"] == 108,
                "actual CPU native entry differs: " + arm)
        resolved = actual["resolved"]
        require(resolved["file_count"] == 39 and resolved["missing"] == resolved["extra"] == resolved["linked"] == []
                and resolved["traversal_truncated"] is False and actual["actual_member_SHA256"] == inventory
                and actual["native_source_SHA256"] == expected["native_source_SHA256"] == sources,
                "actual CPU member/source closure differs: " + arm)
    return True


def verify_entry_prerequisites(accounting):
    binding = read(SRC / "CPU_ENTRY_GATE_BINDING.json")
    for path, digest in binding["original_artifact_pins"].items():
        actual = Path(path)
        require(actual.is_relative_to(ENTRY_GATE) or actual == CPU / "inventory.json", "foreign entry artifact")
        require(pin(actual, 4 * 1024**2)["SHA256"] == digest, "actual CPU entry original differs")
    manifest = read(ENTRY_GATE / "source/SOURCE_FILES.json")["files"]
    require(len(manifest) + 1 == 13, "CPU entry source denominator differs")
    for name, info in manifest.items():
        require(Path(name).name == name and pin(ENTRY_GATE / "source" / name, 1024**2) == info,
                "actual CPU entry source member differs")
    probes, rows = {}, {}
    for arm, placement in ARMS.items():
        folder = ENTRY_GATE / "arms" / arm
        require(pin(folder / "probe.json", 1024**2)["SHA256"] == binding["arms"][arm]["probe_SHA256"],
                "original entry probe differs")
        probes[arm] = read(folder / "probe.json")
        rows[arm] = accounting(placement["parent"], ENTRY_GATE_LABEL + "-" + arm)
    sources = {name: pin(SRC / name, 1024**2)["SHA256"]
               for name in ("native_generation.py", "check_explicit_dev_inputs.py", "check_qwen_native_inputs.py")}
    verify_entry_records(binding, read(ENTRY_GATE / "intent.json"), read(ENTRY_GATE / "outcome.json"),
                         probes, rows, read(CPU / "inventory.json"), sources)
