"""Integrate the observed epoch-one matrix with original checkpoint-2 custody."""
import json
from hashlib import sha256
import math
import os
from pathlib import Path
import stat


REQUEST = "4c23112f1a064af6bfd27d23e55ef89e"
LABEL = "training-internvl-gqa56-cp2-to-cp3-native-v1"
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
ORIGINAL_CP2 = ROOT / "results/training-internvl-gqa56-first2-native-observed-v3/train/checkpoint-2"
CHECKPOINT = Path("/gqa-recovery-cp2")
ARROW_ROOT = ROOT / "incoming/internvl-gqa56-native-arrow-epoch1-c6d2ed29e9454b72a3ba57be5a6c3f1a/output"
ARROW_REQUEST = "c6d2ed29e9454b72a3ba57be5a6c3f1a"
ARROW_RESULT = "0edb15c8762aac40a9fab7954cac0a62bf898a5ab263a04b58342d8b9dc5be3a"
ARROW_ORDER = "5d7ec6e740dfca0cf3004fb0d979fd37ce0d8878ef9a6af1e46b24a89ee33f4e"
MATRIX = "30f9d0c16edd2926d51a0df98847414a42a4376e77343605f1eda630881943cf"
FIRST4 = "1cf3008dd1992482d4706a2090ff59acd4cf0c5812fc09bebeec4e25903b8934"
EPOCH_ARCHIVE = "49f2fb9a42de79269ce723d9c9df2e0c44c8e1e8d3a81e8b7f856465c2702730"
CPU_RESTORE_ARCHIVE = "37a59fec9f72c31bae49159c7df5ac59fbf77d3f6ad76d75aeead06f06fb5691"
CP2_ARCHIVE = "6b52687a21e1991ef410b3bb8e4bc411cdd03ddad0cea0c16b1f3e68aac26258"
CP2_FP = "cdf23c3c1b82b2132b9242d285993f6ab8a44006d11093c0c992364c1beaf797"
LR = 9.957224306869053e-05
SCHEDULE_LR_AFTER = 1e-4 * (0.5 * (1 + math.cos(math.pi * (3-1) / (25-1))))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return sha256(value).hexdigest()


def validate_arrow(result, matrix, first):
    need((result["request"], result["actual_step"], result["epoch_index"], result["expected_sampler_seed"],
          result["data_seed"]) == (ARROW_REQUEST, "1103606.4726", 1, 43, 42) and
         result["actual_arrow_mapping_verified"] is True and result["actual_optimizer_updates"] == 0 and
         result["observed_loader_workers"] == 0 and result["production_workers"] == 4 and
         result["actual_template_collator_invoked"] is False and result["actual_forward_order_verified"] is False and
         (result["physical_occurrences"], result["unique_source_rows"], result["repeat_occurrences"],
          result["internal_source_rows"], result["cp3_first4_verified_slots"]) == (64, 56, 8, 0, 32) and
         result["actual_rank_occurrence_SHA256"] == MATRIX and result["cp3_first4_SHA256"] == FIRST4 and
         result["actual_arrow_logical_to_physical_SHA256"] == ARROW_ORDER, "EXACT_4726_CPU_RESULT")
    need(len(matrix) == 8 and all(type(row) is list and len(row) == 8 for row in matrix) and
         first == dict(epoch_index=1, next_global_step=3,
                       ranks=[dict(rank=rank, rows=row[:4]) for rank, row in enumerate(matrix)]),
         "EXACT_CP3_FIRST4_OF_EIGHT_REAL_RANKS")


def load_arrow(root, pin):
    result = json.loads(pin(root/"result.json", expected=ARROW_RESULT, extent=6740))
    matrix = json.loads(pin(root/"physical-rank-matrix.private.json", expected=MATRIX, extent=10610))
    first = json.loads(pin(root/"cp3-first4.private.json", expected=FIRST4, extent=5510))
    validate_arrow(result, matrix, first)
    return matrix


def optimizer_mapping(optimizer, named, saved):
    """Follow actual load_state_dict group/order, then require distinct live named parameters."""
    by_identity = {id(parameter): name for name, parameter in named}
    need(len(by_identity) == len(named) == 504 and len(set(by_identity.values())) == 504,
         "EXACT_504_LIVE_NAMED_PARAMETERS")
    groups = optimizer.param_groups
    need(len(groups) == len(saved["param_groups"]) and len(optimizer.state) == 504,
         "EXACT_LIVE_ADAMW_GROUPS")
    plan = []
    for current, original in zip(groups, saved["param_groups"], strict=True):
        need(len(current["params"]) == len(original["params"]) and current["lr"] == original["lr"] == LR,
             "EXACT_ADAMW_GROUP_CARDINALITY_AND_LR")
        for parameter, key in zip(current["params"], original["params"], strict=True):
            need(type(key) is int and id(parameter) in by_identity and parameter in optimizer.state,
                 "EXACT_ADAMW_INDEX_TO_NAMED_PARAMETER")
            plan.append((key, by_identity[id(parameter)], parameter))
    need(len(plan) == len({key for key, _, _ in plan}) == len({name for _, name, _ in plan}) == 504 and
         {key for key, _, _ in plan} == set(saved["state"]), "ADAMW_INDEX_TO_NAME_BIJECTION")
    return plan


def resume_argv(command):
    result = list(command)
    start = result.index("--callbacks") + 1
    stop = next((index for index in range(start, len(result)) if result[index].startswith("--")), len(result))
    need(result[start:stop] == ["internvl_gqa56_cp3_observed_engineering"], "EXACT_CP3_CALLBACK_LIST")
    need(not any(arg in result for arg in ("--resume_from_checkpoint", "--ignore_data_skip", "--resume_only_model"))
         and result[result.index("--max_steps")+1] == "25"
         and result[result.index("--dataloader_num_workers")+1] == "4", "ORIGINAL_25_STEP_FOUR_WORKER_SFT")
    return result + ["--resume_from_checkpoint", CHECKPOINT.as_posix(), "--resume_only_model", "false",
                     "--ignore_data_skip", "false"]


def verify_cp2(archive, cp, pin, clock):
    need(cp.resolve(strict=True) == cp and os.statvfs(cp).f_flag & os.ST_RDONLY, "EXACT_CP2_READONLY_MOUNT")
    record = json.loads(pin(archive, expected=CP2_ARCHIVE, cap=20000))
    checkpoint = record["checkpoint_independent_readback"]
    members = checkpoint["actual_members"]
    need(checkpoint["actual_step"] == "1103606.4703" and checkpoint["matched_live_final_params"] is True and
         checkpoint["native_final_param_SHA256"] == CP2_FP and len(members) == 17 and
         {p.name for p in cp.iterdir()} == set(members), "EXACT_17_ORIGINAL_CP2_MEMBERS")
    for name, row in sorted(members.items()):
        path = cp/name
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_size == row["bytes"], "ORIGINAL_CP2_EXTENT")
            checksum = sha256()
            while chunk := stream.read(4*1024**2):
                clock()
                checksum.update(chunk)
            after = os.fstat(fd)
            current = path.lstat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            need(all(getattr(before, key) == getattr(after, key) == getattr(current, key) for key in fields)
                 and checksum.hexdigest() == row["sha256"], "EXACT_ORIGINAL_CP2_STABLE_SHA256")
    return len(members)
