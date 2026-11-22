"""Use both orientations of the seven frozen near pairs in the native backend."""
from pathlib import Path

from .tum_vggt_pair import prepare_pair
from .vggt_eight_pair_settings import SETTINGS as ORIGINAL_SETTINGS, check_config
from .vggt_three_recording_inputs import checked_input
from bidirectional import reverse_pair
from split_contract import BUNDLE_SHA, TRAIN, descriptors, encoded, need, sha
from view_schedule import (PAIR_COUNT, VIEW_BATCHES, bindings, validate_observed_views,
                           view_order_sha256)


SETTINGS = dict(ORIGINAL_SETTINGS, cached_batches=VIEW_BATCHES)
SHAPES = dict(images=(1, 2, 3, 224, 224), depths=(1, 2, 224, 224), extrinsics=(1, 2, 3, 4),
              intrinsics=(1, 2, 3, 3), point_masks=(1, 2, 224, 224))


def batches(torch, config, tensor_batch, valid_depth):
    need(type(config.get("cached_batches")) is int and config["cached_batches"] == VIEW_BATCHES,
         "Expected fourteen derived training batches")
    root = check_config(dict(config, cached_batches=ORIGINAL_SETTINGS["cached_batches"]))
    need(config["dataset_sha"] == BUNDLE_SHA, "Original complete source bundle differs")
    train, _, declaration = descriptors(root)
    roster = bindings(item["pin"] for item in train)
    originals = []
    for item in train:
        raw, evidence = prepare_pair(Path(item["root"]), item["pair"])
        layouts = checked_input(raw, TRAIN)
        originals.append((raw, evidence, layouts))

    values, rows = [], []
    for binding in roster:
        raw, evidence, layouts = originals[binding["physical_index"]]
        if binding["direction"] == "reverse":
            raw = reverse_pair(raw)
        value = tensor_batch(torch, raw)
        need(all(tuple(value[key].shape) == shape for key, shape in SHAPES.items())
             and valid_depth(value["point_masks"]), "Native training layout or mask differs")
        values.append(value)
        rows.append(dict(binding, sequence=TRAIN, input_frame_ids=list(raw["ids"]),
                         source_layouts=layouts, source_valid_pixels=evidence["valid_depth_pixels"]))
    validate_observed_views(rows)
    return values, dict(verified_pairs=PAIR_COUNT, derived_views=VIEW_BATCHES,
                        physical_recordings={TRAIN: PAIR_COUNT}, rows=rows,
                        input_order_SHA256=sha(encoded(rows)), view_order_SHA256=view_order_sha256(),
                        split=declaration, heldout_geometry_loaded_by_training=False, formal_CO3D=False)
