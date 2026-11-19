"""Layer and caller geometry of the exact deployed GLM config."""
import json
from pathlib import Path

BINDING = json.loads(Path(__file__).with_name("LAYER_BINDING.json").read_bytes())
INDEXER_TYPES = tuple(BINDING["indexer_types"])
MLP_TYPES = tuple(BINDING["mlp_layer_types"])
PAIRS = tuple((row["full"], row["shared"]) for row in BINDING["layers"])
EVENTS_UPPER_BOUND = 454

if (len(INDEXER_TYPES) != 78 or set(INDEXER_TYPES) != {"full", "shared"}
        or len(MLP_TYPES) != 78 or set(MLP_TYPES) != {"dense", "sparse"} or BINDING["TP"] != 16
        or [row["ordinal"] for row in BINDING["layers"]] != [1, 2, 3]
        or len({layer for pair in PAIRS for layer in pair}) != 6
        or any(INDEXER_TYPES[f] != "full" or INDEXER_TYPES[s] != "shared"
               or MLP_TYPES[f] != "sparse" or MLP_TYPES[s] != "sparse" for f, s in PAIRS)):
    raise ValueError("depth observer layer binding differs")


def frame_layers(frame):
    ordinal = frame.get("detail_frame_ordinal")
    if type(ordinal) is not int or not 1 <= ordinal <= len(PAIRS):
        raise ValueError("depth observer ordinal differs")
    actual = (frame["full_detail_layer"], frame["shared_detail_layer"])
    if any(type(layer) is not int for layer in actual) or actual != PAIRS[ordinal - 1]:
        raise ValueError("depth observer selected layers differ")
    return actual


def gate_up_width(layer):
    global_width = (BINDING["full_mlp_width"] if MLP_TYPES[layer] == "dense"
                    else BINDING["sparse_shared_expert_width"])
    local, extra = divmod(global_width, BINDING["TP"])
    if extra or local % 128:
        raise ValueError("depth observer projection shape differs")
    return local
