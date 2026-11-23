"""Stage one paired VGGT validation using the original target inputs and head."""

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "training-vggt-tum-two-scene-bidir-dev24-pair-v2"
REQUEST = "6bd97bc61f0249c0a255ad26fe55a162"
INDEX = "coordination/training/vggt-tum-cross-camera-controller-files-v1.json"
ORIGINAL_STAGE = ROOT / "results/training-vggt-tum-cross-camera-dev24-v1-staged"
SOURCE_RUN_ROOT = ROOT / "results/training-vggt-tum-bidirectional30-v1"
STAGE = ROOT / "results" / LABEL
STAGE_SHA = "d221478fe51e14ccdf8ca858632c60aee67993e693c58f29b2c9357d373dc903"
SOURCE_RUN_SHA = "af61e5d953b51a87fd1975238fa700b3037061e81690309ee49491b4a20a8d20"
SOURCE_RUN_REQUEST = "ab8a312813524d949608defa7124c3bf"
SOURCE_RUN_GPU, SOURCE_RUN_POST = "1103606.498", "1103606.507"
HEAD = "arms/continuous/train/camera_head.pt"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def hash_value(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value)


def file_extent(path):
    info=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(info.st_mode), "noncanonical original file")
    return info.st_size


def metadata(path, pin):
    need(hash_value(pin.get("SHA256")), "expected 64-character original metadata SHA")
    size=file_extent(path)
    need(0<size==pin["bytes"]<=65536, "original metadata extent differs")
    with path.open("rb") as stream:
        raw=stream.read(size+1)
    need(len(raw)==size and digest(raw)==pin["SHA256"], "original metadata SHA differs")
    return json.loads(raw)


def write_json(path, value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(raw)


def validate_public(raw):
    need(digest(raw) == SOURCE_RUN_SHA, "published source training result differs")
    proof = json.loads(raw)["VGGT"]
    expected = {"pre": "1103606.497", "gpu": SOURCE_RUN_GPU, "post": SOURCE_RUN_POST}
    accounting = proof["steps"]
    need(proof["request"] == SOURCE_RUN_REQUEST
         and proof["post_clean"] is proof["recovery_passed"] is True
         and proof["formal_CO3D_effect_or_A100_performance"] is False
         and proof["planned_updates"] == proof["independently_verified_updates"] == 30
         and set(accounting) == set(expected)
         and all(accounting[phase][0] == handle and accounting[phase][2:4] == ["COMPLETED", "0:0"]
                 for phase, handle in expected.items()), "published training terminal differs")
    return accounting


def bind_head(raw, binding):
    accounting = validate_public(raw)
    files = binding["target_metadata"]
    outcome = metadata(SOURCE_RUN_ROOT / "outcome.json", files["outcome.json"])
    intent = metadata(SOURCE_RUN_ROOT / "intent.json", files["intent.json"])
    combined = metadata(SOURCE_RUN_ROOT / "native-result.json", files["native-result.json"])
    need(outcome["request"] == intent["request"] == SOURCE_RUN_REQUEST
         and outcome["passed"] is True and outcome["formal_acceptance"] is False
         and outcome["physical_scenes"] == outcome["physical_training_pairs"] == 2
         and outcome["ordered_training_views"] == 4 and outcome["planned_actual_updates"] == 30
         and outcome["result_SHA256"] == files["native-result.json"]["SHA256"]
         and combined["actual_DCU"] == 1 and combined["formal_CO3D_eligible"] is False
         and all(outcome["stages"][phase]["row"] == row and outcome["stages"][phase]["passed"] is True
                 for phase, row in accounting.items()), "original target intent or controller differs")
    receipt_key = "arms/continuous/receipt.json"
    native_key = "arms/continuous/native-result.json"
    receipt = metadata(SOURCE_RUN_ROOT / receipt_key, files[receipt_key])
    native = metadata(SOURCE_RUN_ROOT / native_key, files[native_key])
    need(receipt["arm"] == "continuous" and receipt["executed_optimizer_steps"] == 25
         and receipt["result_SHA256"] == files[native_key]["SHA256"]
         and receipt["physical_scene_count"] == receipt["physical_training_pairs"] == 2
         and receipt["ordered_training_views"] == 4, "continuous receipt differs")
    metrics = native["metrics"]
    artifacts = [item for item in native["artifacts"] if item["path"] == "/pilot/" + HEAD]
    head_size = file_extent(SOURCE_RUN_ROOT / HEAD)
    need(metrics["global_optimizer_step"] == metrics["executed_optimizer_steps"] == 25
         and metrics["cached_valid_batches"] == 4 and metrics["training_scope"] == "camera_head"
         and metrics["trainable_backbone_parameters"] == 0
         and native["checks"]["frozen_aggregator_unchanged"] is True
         and len(artifacts) == 1 and hash_value(artifacts[0]["sha256"])
         and hash_value(metrics["camera_head_sha_after"])
         and binding["source_training_request"] == SOURCE_RUN_REQUEST
         and binding["source_gpu_step"] == SOURCE_RUN_GPU
         and 0 < head_size == binding["head_bytes_stat"] <= 4 * 1024**3
         and artifacts[0]["sha256"] == binding["file_SHA256"]
         and metrics["camera_head_sha_after"] == binding["module_SHA256"], "original trained head identity differs")
    return {"two_scene_bidir25": dict(
        gpu_step=SOURCE_RUN_GPU, request=SOURCE_RUN_REQUEST, clean_post=SOURCE_RUN_POST, global_step=25,
        training_receipt_SHA256=files[receipt_key]["SHA256"],
        original_outcome_SHA256=files["outcome.json"]["SHA256"],
        original_combined_native_SHA256=files["native-result.json"]["SHA256"],
        original_continuous_result_SHA256=files[native_key]["SHA256"],
        file_SHA256=binding["file_SHA256"], module_SHA256=binding["module_SHA256"],
        head_bytes_stat=head_size, actual_checkpoint_bytes_read=0)}


def run(payload):
    os.umask(0o077)
    sources,context=payload["sources"],payload["context"]
    need((context["request"],context["label"],context["stage_SHA256"])==(REQUEST,LABEL,STAGE_SHA)
         and context["epoch"]<=time.time()<context["deadline"]-1550 and context["deadline"]==context["epoch"]+1800,
         "request or original budget differs")
    install=ROOT/"incoming"/LABEL
    need(install.parent.resolve(strict=True)==install.parent and ORIGINAL_STAGE.resolve(strict=True)==ORIGINAL_STAGE
         and not STAGE.exists() and not STAGE.is_symlink() and not install.exists(),
         "aliased or already attempted scoring namespace")
    marker=ORIGINAL_STAGE/"stage-complete.json"
    marker_size=file_extent(marker)
    staged=metadata(marker,dict(bytes=marker_size,SHA256=STAGE_SHA))
    binding=payload["head_binding"]
    need(digest(binding)==context["head_binding_SHA256"],"published checkpoint binding SHA differs")
    authority=bind_head(payload["source_terminal"],json.loads(binding))
    raw_index=sources[INDEX]
    need(digest(raw_index)==context["source_index_SHA256"],"derived scorer source index differs")
    index=json.loads(raw_index)
    need(len(index)==15 and set(sources)=={*index,INDEX},"unexpected source manifest")
    for name,data in sources.items():
        path=PurePosixPath(name)
        need(not path.is_absolute() and ".." not in path.parts and 0<len(data)<=2*1024**2
             and (name==INDEX or digest(data)==index[name]),"unexpected scorer artifact")
    auxiliary=payload["auxiliary"]
    wanted=context["auxiliary_SHA256"]
    names={"audit_terminal.py","read_vggt_dev24_roundoff_audit.py","dev24_decision.py"}
    need(set(auxiliary)==set(wanted)==names
         and all(len(data)<=65536 and digest(data)==wanted[name] for name,data in auxiliary.items()),
         "independent audit helper SHA differs")
    need(digest(sources["tools/audit_vggt_tum_cross_camera_dev.py"])==context["pair_reader_SHA256"],
         "derived independent reader SHA differs")
    install.mkdir()
    for name,data in auxiliary.items():
        with (install/name).open("xb") as stream:
            stream.write(data)
    for name,data in sources.items():
        path=install/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    STAGE.mkdir()
    (STAGE/"inputs").mkdir()
    input_files=sorted((ORIGINAL_STAGE/"inputs").iterdir())
    need(len(input_files)==23,"original staged input inventory differs")
    for file in input_files:
        file_extent(file)
        os.link(file,STAGE/file.relative_to(ORIGINAL_STAGE))
    staged["head_authority_SHA256"]=write_json(STAGE/"head-authority.json",authority)
    staged["original_stage_SHA256"]=STAGE_SHA
    staged["new_head_public_result_SHA256"]=SOURCE_RUN_SHA
    staged["pair_reader_SHA256"]=context["pair_reader_SHA256"]
    staged["head_binding_SHA256"]=context["head_binding_SHA256"]
    staged.pop("scorer_arguments",None)
    stage_sha=write_json(STAGE/"stage-complete.json",staged)
    print(json.dumps(dict(request=REQUEST,stage_SHA256=stage_sha,head_authority_SHA256=staged["head_authority_SHA256"],
                          windows=24,admitted=22,planned_predictions=44,optimizer_updates=0)),flush=True)
    timeout=min(1510,context["deadline"]-time.time())
    need(timeout>=1505,"insufficient remaining original controller budget")
    command=[sys.executable,"-B",str(install/"tools/control_vggt_tum_cross_camera_dev.py"),
             str(STAGE),"--stage-sha",stage_sha,"--request",REQUEST,"--publication",context["publication"],
             "--manifest-sha",context["source_index_SHA256"]]
    return subprocess.run(command,timeout=timeout).returncode


if __name__=="__main__":
    raise SystemExit(run(INPUT))
