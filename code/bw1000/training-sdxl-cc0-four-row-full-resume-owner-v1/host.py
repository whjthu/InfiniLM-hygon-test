"""Connect the frozen SDXL v3 owner to the four .2766 rows and exact new CP20 recovery."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
FOUR_ARCHIVE = "c1a5ed37c352ccf03f5a4ee9b4a3073e6b70dc8a63cdde1b960322b28efab472"
FOUR_INPUT = RESULTS / "sdxl-cc0-four-imagefolder-native-ae669d303b204e93bf5e419225be27bb" / "imagefolder"
IMAGE = ROOT / "containers/dtk26-sandbox-r1"
PARENT, NODE = "1194900", "f13r4n03"
BASE = "686c7cc61b2b8b3862ad47a58376700cf5d2f946ba7c8fca392023335eae16ab"
DEVICE = "2e6bc5e319b281df833c327b7a80a7e5c8eb15126a7531d2933050163c0d12fa"
FULL_MANIFEST = "19843aa3c7c74267088759f43edd19980e77abdbf5b9c935a471e2a7905a34c7"
FULL = {
    "derive.py": "39c07525a6fcd696aff5f016ed34343315ea1fb446ef75da6c7a6cff67157445",
    "run.py": "5c06c20b4fa649fe4acc4e8ac1d07ad3b207710fbc4b7e27bb17d984b464fecd",
    "train4.py": "35a96d77e39b04c87704c2692a7a9407cbfbd2acc11a895323ebfd0cf598eb32"
}
DERIVED = {
    "sdxl_hip_checkpoint.py": "7105c88a0f9a292396c37acec217e66e8cc60f7142903251a8be219544e223f0",
    "sdxl_hip_common.py": "d6b577e79ab5ba8cb72a1d87e517aaae6c3188d221cad8101fd618cc4e438707",
    "train_text_to_image_lora_sdxl.py": "cc8bd4a3cdeb890c7ee2f13d8c7d2f1135fcc048c6b4a52dd4bc6beadd72b68f",
}
HELPER = "3392f0ea6755e8d734ee02cc9d518436f9f579a555b599d6d46720d092048f38"
SCHEMA = "sdxl-native-lora-full-step-hip-v1"
ORIGINAL_2670 = "24624803a2f89d04f89d832c0cef459e2fda57eb3cf85092351b56c87781d2b7"
HASH = re.compile(r"[a-f0-9]{64}")
IMAGE_ID = "sha256:9906c8cc26fefb917b5067a792f976b8e672118e4ee1f52f5826ec7fe72a567e"
UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def label(request):
    need(re.fullmatch(r"[a-f0-9]{32}", request) and request[:12] not in
         {"abd5c98bc5dc", "ae669d303b20", "c5eda9cfa6ad", "fd93b793bbbd", "e2f599910efc"}, "FRESH_CONTINUITY_REQUEST")
    return "sdxl-cc0-four-full25-cp20-" + request[:12]


def paths(request):
    return ROOT / "incoming" / (label(request) + "-stage"), RESULTS / label(request)


def read_module(path, name, pin):
    need(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
         "CANONICAL_PINNED_SOURCE")
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    need(len(raw) <= 128 * 1024 and sha256(raw).hexdigest() == pin, "EXACT_FROZEN_MODULE")
    doc = module_from_spec(spec_from_file_location(name, path))
    exec(compile(raw, str(path), "exec"), doc.__dict__)
    return doc


def bind(base_path):
    """Load the v3 controller with only the four-row ingress and guest substitutions."""
    base = read_module(base_path, "frozen_sdxl_v3_owner", BASE)
    native_staged, native_guest = base.staged, base.guest
    base.ROOT, base.RESULTS, base.HERE = ROOT, RESULTS, HERE
    base.INPUT = FOUR_INPUT
    assets = inspect.getsource(base.assets)
    marker = '    need(INPUT.resolve(strict=True) == INPUT'
    need(assets.count(marker) == 1, "EXACT_V3_OLD_INPUT_GATE_ANCHOR")
    exec(compile(assets.split(marker)[0], "<frozen-v3-assets-before-old-input>", "exec"), base.__dict__)
    staging = inspect.getsource(base.stage)
    marker = "native_input5039_members=4"
    need(staging.count(marker) == 1, "EXACT_V3_STAGE_INPUT_METADATA_ANCHOR")
    exec(compile(staging.replace(marker, "native_input2766_members=5"),
                 "<v3-stage-four-row-evidence>", "exec"), base.__dict__)
    base.paths, base.label = paths, label
    base.staged = lambda common, target, request, digest=None: staged(base, common, target, request, native_staged, digest)
    base.guest = lambda target, work, scratch, request, pin: guest(base, native_guest, target, work, scratch, request, pin)
    base.observe = lambda request, work, common, old: observe(base, request, work, common, old)
    control = inspect.getsource(base.start)
    failure_count = "planned_updates=1, actual_optimizer_updates=None,"
    need(control.count(failure_count) == 1, "EXACT_V3_FAILURE_DENOMINATOR_ANCHOR")
    exec(compile(control.replace(failure_count, "planned_updates=30, actual_updates=None,"),
                 "<frozen-v3-owner-continuity-denominator>", "exec"), base.__dict__)
    return base


def stamp():
    need(IMAGE.resolve(strict=True) == IMAGE and IMAGE.is_dir(), "EXACT_CANONICAL_DTK_ROOTFS")
    st = IMAGE.lstat()
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def proof(value, image_stamp):
    need(type(value) is dict and set(value) == {"rootfs", "provider"}, "EXACT_INDEPENDENT_RUNTIME_PROOF_FIELDS")
    rootfs, provider = value["rootfs"], value["provider"]
    need(set(rootfs) == {"host_path", "image_id", "method", "source_receipt_SHA256", "stat"}
         and rootfs["host_path"] == IMAGE.as_posix() and rootfs["image_id"] == IMAGE_ID
         and type(rootfs["method"]) is str and re.fullmatch(r"[a-z][a-z0-9_-]{2,79}", rootfs["method"])
         and HASH.fullmatch(rootfs["source_receipt_SHA256"]) and rootfs["stat"] == image_stamp,
         "EXACT_INDEPENDENT_DTK_ROOTFS_IDENTITY")
    need(set(provider) == {"image_id", "receipt_SHA256", "step", "parent", "node", "accounting", "zero_gres",
                          "hip_full_sources_SHA256", "native_cli_worker_count", "native_full_checkpoint"}
         and provider["image_id"] == rootfs["image_id"] and HASH.fullmatch(provider["receipt_SHA256"])
         and provider["parent"] == PARENT and provider["node"] == NODE
         and re.fullmatch(PARENT + r"\.\d+", provider["step"]) and provider["zero_gres"] is True
         and type(provider["native_cli_worker_count"]) is int and provider["native_cli_worker_count"] == 0
         and provider["native_full_checkpoint"] is True and provider["hip_full_sources_SHA256"] == DERIVED,
         "EXACT_INDEPENDENT_ZERO_GRES_PROVIDER")
    row = provider["accounting"]
    need(type(row) is list and len(row) == 9 and row[0] == provider["step"] and row[1]
         and row[2:4] == ["COMPLETED", "0:0"] and row[4].isdigit() and row[5] == NODE
         and all(UTC.fullmatch(s) for s in row[7:9]) and row[7] <= row[8], "EXACT_INDEPENDENT_NUMBERED_RUNTIME_ACCOUNTING")
    values = [part.split("=", 1) for part in row[6].split(",")]
    tres = dict(values)
    need(len(values) == len(tres) == 3 and set(tres) == {"cpu", "mem", "node"} and tres["node"] == "1"
         and re.fullmatch(r"[1-9]\d*", tres["cpu"]) and re.fullmatch(r"[1-9]\d*(?:\.\d+)?[MG]", tres["mem"]),
         "EXACT_RUNTIME_PROBE_ZERO_GRES_TRES")
    return rootfs["image_id"]


def full_sources(common, folder):
    manifest = common.read(folder / "SOURCE_FILES.json", FULL_MANIFEST)
    declared = json.loads(manifest)
    sources = {name: common.read(folder / name, digest) for name, digest in FULL.items()}
    need(all(declared[name] == {"bytes": len(raw), "SHA256": FULL[name]} for name, raw in sources.items()),
         "EXACT_CONTINUITY_SOURCE_MANIFEST")
    return dict(sources, **{"SOURCE_FILES.json": manifest})


def four_helper(folder):
    return read_module(folder / "train4.py", "frozen_sdxl_four_row_contract", FULL["train4.py"])


def stage(base, args):
    target, _ = paths(args.request)
    common = read_module(args.common, "frozen_stage_common", base.COMMON)
    sources = full_sources(common, args.full_source)
    train4 = four_helper(args.full_source)
    four_raw, _ = train4.archive(args.four_proof)
    need(train4.ARCHIVE == FOUR_ARCHIVE and len(train4.verify(FOUR_INPUT, args.four_proof)) == 4,
         "FOUR_ROW_HOST_INGRESS")
    need(HASH.fullmatch(args.runtime_proof_sha), "ROOT_RUNTIME_PROOF_PIN")
    runtime = common.read(args.runtime_proof, args.runtime_proof_sha, 16 << 10)
    image_id = proof(json.loads(runtime), stamp())
    common.read(HERE / "device.py", DEVICE)
    common.read(base.FRAME / "genmlu/sdxl_training_resume_checkpoint.py", HELPER)
    base_receipt = base.stage(args)
    common.put(target / "base-host.py", common.read(args.base_host, BASE))
    (target / "full").mkdir(mode=0o700)
    for name, raw in sources.items():
        common.put(target / "full" / name, raw)
    common.put(target / "continuity-runtime.json", runtime)
    common.put(target / "four-row-2766.json", four_raw)
    receipt = dict(request=args.request, old_stage_SHA256=base_receipt["stage_SHA256"], full_source_SHA256=FULL_MANIFEST,
                   runtime_proof_SHA256=args.runtime_proof_sha, image_id=image_id, resume_step=20,
                   planned_updates=30, original_first_update_SHA256=ORIGINAL_2670,
                   input_2766_SHA256=FOUR_ARCHIVE, input_members=5, input_rows=4, GPU_GO=False)
    return dict(receipt, stage_SHA256=common.put(target / "continuity-stage.json", common.wire(receipt)))


def staged(base, common, target, request, native_staged, digest=None):
    raw = common.read(target / "continuity-stage.json", digest)
    spec = json.loads(raw)
    need(spec["request"] == request and spec["resume_step"] == 20 and spec["planned_updates"] == 30
         and spec["GPU_GO"] is False and spec["full_source_SHA256"] == FULL_MANIFEST
         and spec["original_first_update_SHA256"] == ORIGINAL_2670
         and spec["input_2766_SHA256"] == FOUR_ARCHIVE and spec["input_members"] == 5 and spec["input_rows"] == 4,
         "EXACT_CONTINUITY_STAGE")
    native_staged(common, target, request, spec["old_stage_SHA256"])
    full_sources(common, target / "full")
    four_helper(target / "full").verify(FOUR_INPUT, target / "four-row-2766.json")
    raw_runtime = common.read(target / "continuity-runtime.json", spec["runtime_proof_SHA256"], 16 << 10)
    need(proof(json.loads(raw_runtime), stamp()) == spec["image_id"], "CURRENT_EXACT_DTK_IMAGE_BINDING")
    return spec


def guest(base, native_guest, target, work, scratch, request, pin):
    argv = native_guest(target, work, scratch, request, pin)
    previous = f"{(base.PIPE / 'source').as_posix()}:/scope:ro"
    need(argv.count(previous) == argv.count("/scope/pipeline.json") == argv.count("--pipeline-manifest-sha") == 1,
         "FROZEN_V3_ARGV_ANCHORS")
    offset = argv.index(previous)
    argv[offset:offset+1] = [f"{(target / 'full').as_posix()}:/scope:ro", "--bind",
                            f"{(base.PIPE / 'source').as_posix()}:/first-update-source:ro", "--bind",
                            f"{(target / 'four-row-2766.json').as_posix()}:/four-row-2766.json:ro"]
    argv[argv.index("/scope/pipeline.json")] = "/first-update-source/pipeline.json"
    four_input = f"{FOUR_INPUT.as_posix()}:/two-cc0:ro"
    need(argv.count(four_input) == argv.count("/two-cc0") == 1, "EXACT_V3_INPUT_BIND_ANCHOR")
    argv[argv.index(four_input)] = f"{FOUR_INPUT.as_posix()}:/four-cc0:ro"
    argv[argv.index("/two-cc0")] = "/four-cc0"
    index = argv.index("--pipeline-manifest-sha")
    del argv[index:index+2]
    common, _, _, _, _ = base.common(target)
    spec = base.staged(common, target, request)
    argv += ["--firstupdate-source", "/first-update-source", "--image-id", spec["image_id"], "--resume-step", "20",
             "--four-proof", "/four-row-2766.json"]
    return argv


def observe(base, request, work, common, old):
    roles = {role: common.role_row(old, request, role) for role in ("pre", "gpu", "post")}
    result = dict(request=request, planned_updates=30, accounting=roles, actual_updates=None,
                  cleanup_verified=False, scientific_pass=False, formal_domain_credit=0)
    if not (work / "auth.json").is_file():
        return result
    auth = base.scope(json.loads(common.read(work / "auth.json")), request)
    gpu = roles["gpu"]
    if all(roles.values()) and roles["post"][2:4] == ["COMPLETED", "0:0"]:
        pre, gpu, post = [common.role_row(old, request, role, terminal=True) for role in ("pre", "gpu", "post")]
        before, _ = common.proof(work, request, "pre", pre)
        after, _ = common.proof(work, request, "post", post)
        intended = json.loads(common.read(work / "post-intent.json"))
        need(before["boot_id"] == after["boot_id"] == auth["boot"] and before["pci"] == after["pci"] == auth["pci"]
             and after["owned_processes_remaining"] == 0 and pre[2:4] == ["COMPLETED", "0:0"]
             and intended["gpu_step"] == gpu[0] and pre[8] <= gpu[7] <= gpu[8] <= post[7],
             "INDEPENDENT_PHYSICAL_CLEAN_POST")
        result["cleanup_verified"] = True
    out = work / "durable/once"
    if not (out / "result.json").is_file():
        return result
    raw = common.read(out / "result.json")
    science = json.loads(raw)
    need(science["request"] == request and science["planned_updates"] == 30, "EXACT_CONTINUITY_RESULT_SCOPE")
    result.update(actual_updates=science.get("actual_updates"), completed_continuous=science.get("continuous_updates"),
                  completed_resume=science.get("resumed_updates"), scientific_result_SHA256=sha256(raw).hexdigest(),
                  native_pass=science.get("passed") is True, first_different_update=science.get("first_different_update"))
    if science.get("passed") is not True:
        return result
    target, _ = paths(request)
    spec = base.staged(common, target, request, auth["stage_SHA256"])
    device = json.loads(common.read(work / "durable/device.json"))
    need(gpu is not None and device["request"] == request and device["step"] == gpu[0]
         and device["boot"] == auth["boot"] and device["pci"] in auth["pci"]
         and device["visible_devices"] == 1 and device["ordinal"] == 0, "EXACT_ONE_CARD_RESULT_BINDING")
    need(science["source_SHA256"] == DERIVED and science["restored_full_state_exact"] is True
         and science["resumed_from"] == 20 and science["actual_updates"] == 30
         and (science["continuous_updates"], science["resumed_updates"]) == (25, 5)
         and science["final_full_state_exact"] is True and science["first_different_update"] is None,
         "EXACT_CONTINUOUS_AND_RESUMED_COUNTS")
    train4 = four_helper(target / "full")
    data = train4.bind_data(FOUR_INPUT, target / "four-row-2766.json")
    records = data["records"]
    need(science["input_2766_SHA256"] == FOUR_ARCHIVE and science["input_2766_rows"] == len(records),
         "INDEPENDENT_EXACT_FOUR_ROW_RESULT_BINDING")
    source = out / "source/examples/text_to_image"
    for filename, pin in DERIVED.items():
        common.read(source / filename, pin)
    helper = read_module(source / "sdxl_hip_checkpoint.py", "continuity_independent_checkpoint", DERIVED["sdxl_hip_checkpoint.py"])
    originals = {step: helper.preflight(out / f"continuous/checkpoint-{step}") for step in (10, 20, 25)}
    final = helper.preflight(out / "resume/checkpoint-25")
    for step, document in (*originals.items(), (25, final)):
        need(document["schema"] == SCHEMA and document["step"] == step and
             document["contract"]["plan"]["genmlu_runtime_image_id"] == spec["image_id"]
             and document["contract"]["data"] == data,
             "EXACT_NATIVE_CHECKPOINT_AND_ROOTFS")
    need(originals[20]["state_sha256"] == science["source_checkpoint_state_SHA256"]
         and originals[25]["state_sha256"] == final["state_sha256"] == science["continuous_final_state_SHA256"]
         == science["resume_final_state_SHA256"] and all(value["contract"] == final["contract"] for value in originals.values()),
         "INDEPENDENT_FULL_STATE_BOUNDARY_CHAIN")
    cursor = json.loads(common.read(out / "cp20-cursor.json", science["cp20_cursor_SHA256"], 8 << 10))
    need(set(cursor) == {"step", "epoch", "consumed", "order", "state_sha256", "checkpoint_manifest_SHA256"}
         and (cursor["step"], cursor["epoch"], cursor["consumed"]) == (20, 4, 4)
         and sorted(cursor["order"]) == list(range(4))
         and cursor["state_sha256"] == originals[20]["state_sha256"]
         and cursor["checkpoint_manifest_SHA256"] ==
         helper.file_sha(out / "continuous/checkpoint-20" / helper.MANIFEST), "INDEPENDENT_CP20_CURSOR_BINDING")
    ledgers = {}
    for arm, start in (("continuous", 1), ("resume", 21)):
        raw = common.read(out / arm / "steps.ndjson", limit=2 << 20)
        ledgers[arm] = train4.check_updates(raw, start, 25, records)
    need([row["sample_witness"]["sample_ids"][0] for row in ledgers["continuous"][16:20]] == cursor["order"],
         "INDEPENDENT_CP20_ACTUAL_SAMPLER_ORDER")
    need(all(left[field] == right[field] for left, right in zip(ledgers["continuous"][20:], ledgers["resume"], strict=True)
             for field in ("sample_witness", "loss", "grad_norm")), "INDEPENDENT_PAIRED_UPDATE_PARITY")
    before = json.loads(common.read(out / "continuous/preupdate-witness.json"))
    restored = json.loads(common.read(out / "resume/preupdate-witness.json"))
    need(before["completed_step"] == 0 and restored["completed_step"] == 20 and restored["next_step"] == 21
         and (before["process_pid"], before["process_startticks"]) !=
         (restored["process_pid"], restored["process_startticks"]) and restored["restored_full_state_exact"] is True
         and restored["state_sha256"] == originals[20]["state_sha256"]
         and restored["restored"]["path"] == "/durable/once/continuous/checkpoint-20"
         and restored["restored"]["manifest_sha256"] ==
         helper.file_sha(out / "continuous/checkpoint-20" / helper.MANIFEST), "INDEPENDENT_NEW_PROCESS_CP20_RESTORE")
    result.update(actual_PCI=device["pci"], checkpoints_verified=4, independent_state_exact=True,
                  verified_input_2766_SHA256=FOUR_ARCHIVE, verified_input_rows=len(records), cp20_epoch=4,
                  cp20_consumed=4, cp20_cursor_verified_by_native_guest=True)
    result["scientific_pass"] = result["cleanup_verified"] and gpu[2:4] == ["COMPLETED", "0:0"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("stage", "start", "terminal", "compute-pre", "compute-gpu", "compute-post"):
        command = sub.add_parser(name)
        command.add_argument("--request", required=True)
        if name == "stage":
            for key in ("base-host", "common", "support", "pilot", "legacy", "runtime-import-proof", "tracker-proof",
                        "runtime-terminal-proof", "full-source", "runtime-proof", "four-proof"):
                command.add_argument("--" + key, required=True, type=Path)
            for key in ("manifest-sha", "runtime-proof-sha"):
                command.add_argument("--" + key, required=True)
        elif name != "terminal":
            command.add_argument("--auth-sha", required=True)
            if name == "start":
                command.add_argument("--auth", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(TZ="UTC", SLURM_TIME_FORMAT="standard", LC_ALL="C")
    target, work = paths(args.request)
    base = bind(args.base_host if args.mode == "stage" else target / "base-host.py")
    if args.mode == "stage":
        print(json.dumps(stage(base, args), sort_keys=True))
        return 0
    common, old, snap, account, owned = base.common(target)
    if args.mode == "start":
        return base.start(args, target, work, common, old, account)
    if args.mode == "terminal":
        base.staged(common, target, args.request)
        print(json.dumps(observe(base, args.request, work, common, old), sort_keys=True))
        return 0
    return base.compute(args, target, work, common, old, snap, owned)


if __name__ == "__main__":
    raise SystemExit(main())
