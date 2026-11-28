"""Exact target paths and prior CPU evidence for the LLaVA continuation owner."""

import json
from pathlib import Path
import re


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESUME_STAGE = ROOT / "incoming/training-llava-clevr-cp2-resume25-native-v1-0b16fb53ca92"
SCIENCE = "0b16fb53ca923e027462c13ac8a5bb6e548e60d4c640fe1444fd6a793c8cc561"
CP2 = ROOT / "results/training-llava-clevr-first2-c59c90a2356d/train/checkpoint-2"
ARCHIVE_SHA = "825ef990fdcb3033e1031ab89c619bef6a56b497b7db010a2d3026e3def9b196"
PLAN = ROOT / "results/llava-clevr-cp2-order-dd988d55bc4748dba0394bb941d42540/output/CP2_NATIVE_RESUME_PLAN.private.json"
PLAN_SHA = "13577b6c5d372f3cf1ec6b0f8b80d2bb6b1a173abef60ae30ddb4c7411769fb5"
PLAN_BYTES = 226023
CPU_ROW = ["1103606.4993", "llava-clevr-cp2-order-dd988d55bc47", "COMPLETED", "0:0", "36", "f13r1n14",
           "cpu=16,mem=32G,node=1", "2026-11-28T04:21:23", "2026-11-28T04:21:59"]
TENSORS = "0b6ba7ff7701c165a114bc5460262dff4c053390312d086034e5019d8ecace93"
INPUTS = "d00032b2f0e5eced239d87b2d3f3164c43f369bc3796b0a1180913f3a8950b8b"


def root_fields(value, need):
    need(value["plan_SHA256"] == PLAN_SHA and value["CPU_step"] == CPU_ROW[0]
         and value["science_manifest_SHA256"] == SCIENCE
         and re.fullmatch(r"[a-f0-9]{64}", value["assembly_SHA256"])
         and re.fullmatch(r"1103606\.[0-9]+", value["assembly_step"])
         and value["assembly_step"] not in {"1103606.4983", CPU_ROW[0]},
         "EXACT_PRODUCTION_PLAN_AND_NEW_ZERO_GRES_ASSEMBLY")
    path = Path(value["assembly_path"])
    need(path.is_absolute() and path.is_relative_to(ROOT / "results") and path.name == "assembly.json",
         "EXACT_IN_PLACE_ASSEMBLY_RECEIPT_PATH")


def archive_requirements(pin, need):
    raw, _ = pin(RESUME_STAGE / "4983.json", ARCHIVE_SHA)
    old = json.loads(raw)
    plan, _ = pin(PLAN, PLAN_SHA)
    need(len(plan) == PLAN_BYTES, "EXACT_ORIGINAL_4993_PRIVATE_PLAN_SIZE")
    files = old["training"]["checkpoint_manifest"]
    need(len(files) == 17 and {p.name for p in CP2.iterdir()} == set(files), "CURRENT_ORIGINAL_CP2_CLOSURE")
    for member, item in files.items():
        p = CP2 / member
        need(p.resolve(strict=True) == p and p.is_file() and p.stat().st_size == item["bytes"],
             "CURRENT_ORIGINAL_CP2_CANONICAL_EXTENT")
    state_pin = files["trainer_state.json"]
    state = json.loads(pin(CP2 / "trainer_state.json", state_pin["SHA256"], limit=state_pin["bytes"])[0])
    need((state["global_step"], state["epoch"], state["max_steps"]) == (2, 1.0, 25),
         "EXACT_LIVE_CP2_CURSOR_NO_INTRA_EPOCH_SKIP")


def independent_cpu(auth, account, matching, tres, pin, need):
    need(matching(account(CPU_ROW[0]), step=CPU_ROW[0]) == CPU_ROW, "EXACT_ORIGINAL_4993_C0")
    raw, _ = pin(Path(auth["assembly_path"]), auth["assembly_SHA256"], limit=16 << 10)
    evidence = json.loads(raw)
    row = matching(account(auth["assembly_step"]), step=auth["assembly_step"])
    need(row is not None and row[1] == auth["label"] + "-assembly" and row[2:4] == ["COMPLETED", "0:0"]
         and row[5] == "f13r1n14" and tres(row[6], 0, 2, 2048)
         and (evidence["request"], evidence["step"], evidence["plan_SHA256"], evidence["science_SHA256"]) ==
             (auth["request"], auth["assembly_step"], PLAN_SHA, SCIENCE)
         and evidence["passed"] is True and evidence["model_loaded"] is False and evidence["actual_GPU"] == 0,
         "EXACT_FRESH_INDEPENDENT_ASSEMBLY_C0")


def future_auth(request):
    return dict(request=request, GPU_GO=True, zero_GRES_CPU_terminal_verified=True,
                original_CP2_members_verified=17, original_INPUTS_SHA256=INPUTS,
                original_TENSORS_SHA256=TENSORS, original_checkpoint=str(CP2), CPU_step=CPU_ROW[0], plan_SHA256=PLAN_SHA)


def mount_arguments(run, resume_authority=True):
    output = [
        (RESUME_STAGE / "source", "/llava-source", "ro"),
        (CP2, CP2.as_posix(), "ro"),
        (RESUME_STAGE / "4983.json", "/llava-resume-authority/4983.json", "ro"),
        (PLAN, "/llava-resume-authority/CP2_NATIVE_RESUME_PLAN.private.json", "ro"),
    ]
    if resume_authority:
        output.append((run / "RESUME-AUTH.json", "/llava-resume-authority/AUTH.json", "ro"))
    return output
