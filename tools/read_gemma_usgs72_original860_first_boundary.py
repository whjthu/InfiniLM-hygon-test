"""Read only verified metadata for the first restored Gemma accumulation."""

import argparse
import hashlib
import json
import math
from pathlib import Path


CONTINUOUS_GATE = "ba9e7917eb04b298de2104aa5fce80f38b19aa39486d18a2f1f681f446c07a09"
RESUMED_GATE = "c3bc5b4fc69c5bc30c18287f4c51853a4a3e9238099f293905e13353c4585c68"
STATE = ("adapter", "optimizer", "scheduler")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, expected):
    require(path.is_file() and not path.is_symlink(), "missing original metadata file")
    with path.open("rb") as stream:
        data = stream.read(8 * 1024**2 + 1)
    digest = hashlib.sha256(data).hexdigest()
    require(0 < len(data) <= 8 * 1024**2 and digest == expected["SHA256"],
            "original metadata digest or bound differs")
    if "bytes" in expected:
        require(len(data) == expected["bytes"], "original metadata bytes differ")
    return json.loads(data)


def first_boundary(asset, operation):
    continuous = pinned_json(asset / "continuous-independent-terminal.json",
                             {"SHA256": CONTINUOUS_GATE})
    resumed = pinned_json(operation / "independent-terminal.json", {"SHA256": RESUMED_GATE})
    require(continuous["actual_steps"] == ["1103606.850", "1103606.851", "1103606.852"]
            and resumed["actual_steps"] == ["1103606.859", "1103606.860", "1103606.861"]
            and resumed["independent_passed"] is True and resumed["exact_paired_recovery"] is False,
            "original step or independent admission differs")

    records = []
    for rank in (0, 1):
        paths = [asset / arm / "train" / witness / f"rank-{rank:02d}"
                 for arm, witness in (("continuous", "continuous-witness"), ("resumed", "resume-witness"))]
        gates = [terminal["rank_receipts"][rank] for terminal in (continuous, resumed)]
        left, right = [pinned_json(path / "result.json", gate["result"])
                       for path, gate in zip(paths, gates, strict=True)]
        saved_restore = pinned_json(paths[1] / "restored-before-RNG.json",
                                    gates[1]["restored_before_RNG"])
        skip = pinned_json(paths[1] / "native-data-skip.json", gates[1]["native_data_skip"])
        require(left["rank"] == right["rank"] == rank and left["world_size"] == right["world_size"] == 8
                and len(left["inputs"]) == len(left["outputs"]) == 100
                and len(right["inputs"]) == len(right["outputs"]) == 20
                and skip == dict(rank=rank, before=100, skipped=80, after=20),
                "original rank denominators or cursor differs")
        require(len(right["RNG_loads"]) == 1 and right["RNG_loads"][0]["saved_equal"] is True
                and right["RNG_loads"][0]["global_step"] == 20
                and saved_restore["restored_parameter_optimizer_scheduler_equal"] is True,
                "checkpoint restore boundary missing")
        require(left["steps"][19]["global_step"] == 20 and left["steps"][20]["global_step"] == 21
                and right["steps"][0]["global_step"] == 21,
                "step 20/21 metadata differs")
        input_pairs = []
        for index in range(4):
            before, after = left["inputs"][80 + index], right["inputs"][index]
            original, restored = left["outputs"][80 + index], right["outputs"][index]
            gap = abs(original["loss_scalar"] - restored["loss_scalar"])
            require(before["ordinal"] == original["ordinal"] == 81 + index
                    and after["ordinal"] == restored["ordinal"] == 1 + index
                    and before["global_step"] == after["global_step"] == 20 and math.isfinite(gap),
                    "original microbatch ordinal or loss bound differs")
            accumulation = after["runtime_cursor"]
            require(accumulation["micro_in_update"] == index + 1, "wrong restored accumulation position")
            input_pairs.append(dict(
                tail_ordinal=index + 1, exact_input=before["full_input"] == after["full_input"],
                exact_loss=gap == 0 and original["loss"] == restored["loss"],
                absolute_loss_gap=gap, continuous_loss=original["loss_scalar"],
                resumed_loss=restored["loss_scalar"], actual_accumulation_size=accumulation["actual_accumulation_group_size"]))
        before_equal = [key for key in STATE if
                        left["steps"][19]["state"][key] == saved_restore["actual"][key]]
        changed = [key for key in STATE if
                   left["steps"][20]["state"][key] != right["steps"][0]["state"][key]]
        records.append(dict(rank=rank, first_four=input_pairs, restored_state_equal=before_equal,
                            state_differences_after_update_21=changed, checkpoint_rng_load_saved_equal=True,
                            rank_result_SHA256=[gate["result"]["SHA256"] for gate in gates]))
    require(all(pair["exact_input"] for record in records for pair in record["first_four"]),
            "first-window input identity diverged")
    return dict(continuous_gate_SHA256=CONTINUOUS_GATE, resumed_gate_SHA256=RESUMED_GATE,
                checked_ranks=records, raw_input_transferred=False, forward_entry_rng_observed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("operation", type=Path)
    args = parser.parse_args()
    print(json.dumps(first_boundary(args.asset, args.operation), allow_nan=False))
