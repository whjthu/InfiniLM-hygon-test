"""Verify the provider substitution for an already admitted weighted C4 device arm."""

from hashlib import sha256
from importlib.util import find_spec
import json
from pathlib import Path, PurePosixPath
import runpy
import stat


FAMILY = "Deep-C4-causal-consumer-r483"
CPU_SCOPE = "DeepC4-T1-history-guarded-probe-link-v1"
GPU_NODE = "f13r2n01"
STEPS = ("1103606.156", "1103606.811", "1103606.812")
ORIGINAL_REQUEST = "ae85c97293ae410483e55d94383a6c48"
BUILD_PUBLICATION = "ce7657bfba6fbe7512c67bd286ea02ce1d26aac7"
FROZEN_LINK_MANIFEST_SHA = "a567990ca88b1ce7467a2fdb392110ef259b749be0f3deab7d15ba5443f95b21"
TERMINAL_156_SHA = "08eb7a9d738cd0618683c98705ba54a60eb3bca749a70a4d8164e042bcec8a1b"
TERMINAL_811_SHA = "1b56036ca1f5e782c570d8ea25918a4ec2d46e0c82317021c6570128045fb91f"
RAW_RESULT_SHA = "ed686a21da4474d6e07ebae3e504dae1f6c8bc2fefb236b47c1e92388cb008b1"
RAW_LINK_PLAN_SHA = "f9b0bd8cc2182d24b1ed5d3da0753f1f2b417971b488eeb7dd164c3576ade22f"
BUILD_PLAN_SHA = "587a51c72b7bfb3342676a90b73ffa7cab84bdbfd9c8271ff46d23048fea5082"
BASE_PROBE_SHA = "8e6b753787c92b1b5eb1f480fe5b24ed2c19b182ee649e988655ed1d759acccd"
PROBE_OBJECT_SHA = "9d3c7a2d910caa77e8c4c9645a5f4fef1890eac2995eb3f6a9f0f2594ac1c653"
CANDIDATE_PROBE_SHA = "34311b306d35dd4bf6a95be01f38c26ad7d70a021674c7f6e04225469f7d2745"
BASE_NATIVE_SHA = "90ba080a7666c35e4160003412be2f0c127b176eba1fc6c5fa0f1eb21ea244b1"
CANDIDATE_NATIVE_SHA = "75f23875b74a90f5405cc40ecf10034b2759780f91ed93a2dadea6fc27174e88"
PREVIOUS_CANDIDATE_RECEIPT_SHA = "f3e2687bb10d01e5340def99ca15b7cfabca17b936c4ed50eaf1191339d75d5f"
OWNER_HELPER_SHA = "ffaef4c941effc989d1c739c76bcfdec174b386c13ba6755aa5da8fca6c1e798"
BINDING_SHA = "5a1f3fdac61b9b258329633d2277b7df9cac07dde6a525d1f8c8a85262abe75f"
ENTRY_SHA = "ee4f5a50dfddec5d112a6d1ef5193ffd10f1ed5bf93b65047186f55e93f991b4"
RANGE_CHECK_SHA = "fe7e96d235e8d3b891692b9dea2a985715bbc883f3c248daf0623332f43bbff1"
METADATA_SHA = "f56b6d38bdf6097488e707b200ea2cc8b4466ce27750de1c038a6be987a4cf00"
BASE_PROBE = f"/refactor/candidates/{FAMILY}-CPU-v2/libDeep_c4_causal_probe_r483.so"
CANDIDATE_PROBE = f"/refactor/candidates/{CPU_SCOPE}/libDeep_c4_causal_probe_r483.so"
BASE_NATIVE = "/refactor/candidates/Deep-C4-constant-slots-r482-Native-CPU-v1/libDeep_constant_slot_metadata_r482.so"
CANDIDATE_NATIVE = "/refactor/candidates/DeepC4-T1-history-host-compile-v1/libDeep_c4_direct_history_t1.so"
BINDING = f"/refactor/incoming/{FAMILY}/CPU-v2/native-src/binding.py"
PROBE_OBJECT = f"/refactor/candidates/{FAMILY}-CPU-v2/probe.o"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return sha256(raw).hexdigest()


def pinned(path, digest, bound=128 * 1024):
    require(path.resolve(strict=True) == path and stat.S_ISREG(path.stat().st_mode),
            "original artifact needs its canonical regular path")
    with path.open("rb") as stream:
        raw = stream.read(bound + 1)
    require(0 < len(raw) <= bound and sha(raw) == digest, f"exact bytes differ: {path.name}")
    return raw


def in_tree(root, container_path):
    return root.joinpath(*PurePosixPath(container_path).relative_to("/refactor").parts)


def original_reports(baseline, terminal):
    require(sha(baseline) == TERMINAL_156_SHA and sha(terminal) == TERMINAL_811_SHA,
            "frozen independent terminal bytes differ")
    old = json.loads(baseline)["independent_original_build"]["result"]
    new = json.loads(terminal)
    result = new["independent_terminal"]
    accounting = result["accounting"]
    post = new["independent_post"]["selected"]
    owner = old["accounting"]
    require(owner["job"] == STEPS[0] and owner["state"] == "COMPLETED" and owner["exit_code"] == "0:0"
            and owner["node"] == GPU_NODE and old["passive_ELF_verified"] is True
            and old["probe_SHA256"] == BASE_PROBE_SHA and old["object_SHA256"] == PROBE_OBJECT_SHA
            and len(old["exports"]) == len(set(old["exports"])) == 13
            and new["label"] == CPU_SCOPE and new["request"] == ORIGINAL_REQUEST
            and new["source_publication"] == BUILD_PUBLICATION and new["checkpoint_weight_ranges"] == "UNKNOWN"
            and result["state"] == "ORIGINAL_CPU_LINK_COMPLETED_ARTIFACT_MATCHED"
            and accounting["step"] == STEPS[1] and accounting["state"] == "COMPLETED"
            and accounting["exit_code"] == "0:0" and accounting["node"] == GPU_NODE
            and accounting["AllocTRES"] == "cpu=4,mem=8G,node=1"
            and result["cpu_link_artifact_qualified"] is True and result["runtime_provider_verified"] is False
            and result["CPU_result_SHA256"] == RAW_RESULT_SHA and result["link_plan_SHA256"] == RAW_LINK_PLAN_SHA
            and result["probe_SHA256"] == CANDIDATE_PROBE_SHA and result["candidate_native_SHA256"] == CANDIDATE_NATIVE_SHA,
            "original passive candidate or baseline build scope differs")
    require(post["device_probe_step"] == STEPS[2] and post["node"] == GPU_NODE and post["parent"] == "1103606"
            and post["membership_valid"] is True and post["device_probe_observed"] is True
            and post["KFD_clean"] is post["PID_clean"] is True and post["active_steps"] == []
            and post["kfd_process_count"] == post["own_device_fd_process_count"] == 0
            and len(set(post["PCIs"])) == 8, "original CPU link independent cleanup differs")
    return old


def derive(old, build_plan, link_result, link_plan, arm):
    """Map the original project providers into the one permitted substitution."""
    require(arm in ("baseline", "candidate"), "unknown weighted arm")
    original = old["project_provider_SHA256"]
    archive = build_plan["dynamic_SHA256"]
    require(len(original) == 6 and len(archive) == 69 and original.get(BASE_NATIVE) == BASE_NATIVE_SHA
            and all(archive.get(path) == pin for path, pin in original.items())
            and CANDIDATE_NATIVE not in archive
            and old["needed"].count(BASE_NATIVE) == 1 and CANDIDATE_NATIVE not in old["needed"],
            "exact original project provider closure differs")
    direct = [CANDIDATE_NATIVE if path == BASE_NATIVE else path for path in old["needed"]]
    expected_inputs = {PROBE_OBJECT: PROBE_OBJECT_SHA, BASE_PROBE: BASE_PROBE_SHA,
                       BASE_NATIVE: BASE_NATIVE_SHA, CANDIDATE_NATIVE: CANDIDATE_NATIVE_SHA}
    dependencies = dict(archive)
    del dependencies[BASE_NATIVE]
    dependencies[CANDIDATE_NATIVE] = CANDIDATE_NATIVE_SHA
    require(link_result["step"] == STEPS[1] and link_result["request"] == ORIGINAL_REQUEST
            and link_result["publication"] == BUILD_PUBLICATION and link_result["node"] == GPU_NODE
            and link_result["source_manifest_SHA256"] == FROZEN_LINK_MANIFEST_SHA
            and link_result["completed"] is link_result["passed"] is link_result["guarded_probe_object_reused"] is True
            and link_result["linker_attempts"] == 1 and link_result["linker_returncode"] == 0
            and link_result["actual_GPU_calls"] == link_result["checkpoint_weight_reads"] == 0
            and link_result["runtime_provider_verified"] is False
            and link_result["relinked_probe_SHA256"] == CANDIDATE_PROBE_SHA
            and link_result["expected_DT_NEEDED"] == link_plan["expected_DT_NEEDED"] == direct
            and link_plan["candidate_receipt_SHA256"] == PREVIOUS_CANDIDATE_RECEIPT_SHA
            and link_plan["input_SHA256"] == expected_inputs
            and link_plan["expected_transitive_SHA256"] == dependencies,
            "original .811 object, linker result or permitted dependency substitution differs")
    if arm == "baseline":
        return dict(probe=BASE_PROBE, probe_SHA256=BASE_PROBE_SHA, Native_SHA256=BASE_NATIVE_SHA,
                    exports=old["exports"], project_provider_SHA256=dict(original))
    candidate = dict(original)
    del candidate[BASE_NATIVE]
    candidate[CANDIDATE_NATIVE] = CANDIDATE_NATIVE_SHA
    return dict(probe=CANDIDATE_PROBE, probe_SHA256=CANDIDATE_PROBE_SHA, Native_SHA256=CANDIDATE_NATIVE_SHA,
                exports=old["exports"], project_provider_SHA256=candidate)


def read_authority(package, root, arm):
    old = original_reports(pinned(package / "original-156-terminal.json", TERMINAL_156_SHA),
                           pinned(package / "original-811-terminal.json", TERMINAL_811_SHA))
    build_plan = json.loads(pinned(package / "BUILD_PLAN.json", BUILD_PLAN_SHA))
    scope = root / "candidates" / CPU_SCOPE
    link_result = json.loads(pinned(scope / "CPU-result.json", RAW_RESULT_SHA))
    link_plan = json.loads(pinned(scope / "link-plan.json", RAW_LINK_PLAN_SHA))
    return derive(old, build_plan, link_result, link_plan, arm)


def bind_after_admission(package, root, arm, range_receipt_sha, deadline):
    """Open one exact probe after the original receipt gate. Return no device claim."""
    require(root == Path("/refactor"), "bind only inside the original container")
    for name, digest in (("entry", ENTRY_SHA), ("range_receipt", RANGE_CHECK_SHA)):
        expected = package / f"{name}.py"
        pinned(expected, digest)
        spec = find_spec(name)
        require(spec is not None and spec.origin == str(expected), "weighted receipt module escaped the sealed package")
    from entry import admit

    metadata = pinned(package / "CHECKPOINT_METADATA.json", METADATA_SHA)
    deadline("direct_history.before_original_range")
    pins = admit(root / "results", range_receipt_sha, metadata)
    require(len(pins["weights"]) == 7 and pins["total_selected_weight_bytes"] == 13_144_320,
            "original weight receipt scope differs")
    audit = read_authority(package, root, arm)
    owner = package / "check_runtime_owner.py"
    binding_path = in_tree(root, BINDING)
    pinned(owner, OWNER_HELPER_SHA)
    pinned(binding_path, BINDING_SHA)
    binding = runpy.run_path(str(binding_path))
    load = runpy.run_path(str(owner))["load_and_verify"]
    deadline("direct_history.before_verified_probe_dlopen")
    probe, api, loaded = load(in_tree(root, audit["probe"]), audit, binding, root)
    require(len(api) == 13 and loaded["entry_calls"] == 0
            and loaded["loaded_project_DSO_SHA256"] == {audit["probe"]: audit["probe_SHA256"],
                                                        **audit["project_provider_SHA256"]},
            "live C ABI entry and provider ownership differs")
    deadline("direct_history.after_verified_probe_dlopen")
    return probe, binding, dict(loaded, arm=arm, Native_SHA256=audit["Native_SHA256"],
                                CPU_link_result_SHA256=RAW_RESULT_SHA,
                                range_receipt_SHA256=range_receipt_sha, operator_branch_execution_verified=False)
