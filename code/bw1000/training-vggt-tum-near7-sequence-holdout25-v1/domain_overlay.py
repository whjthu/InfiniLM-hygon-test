"""Select seven train pairs before native collection; mathematical backend unchanged."""
from .vggt_eight_pair_settings import SETTINGS as ORIGINAL_SETTINGS, check_config
from .vggt_three_recording_inputs import checked_input
from .tum_vggt_pair import prepare_pair
from split_contract import BUNDLE_SHA, TRAIN, descriptors, sha, encoded, need

SETTINGS = dict(ORIGINAL_SETTINGS, cached_batches=7)


def batches(torch, config, tensor_batch, valid_depth):
    need(type(config.get('cached_batches')) is int and config['cached_batches'] == 7,
         'only seven training batches permitted')
    root = check_config(dict(config, cached_batches=ORIGINAL_SETTINGS['cached_batches']))
    need(config['dataset_sha'] == BUNDLE_SHA, 'original complete source bundle differs')
    train, _, declaration = descriptors(root)
    values, rows = [], []
    shapes = dict(images=(1,2,3,224,224), depths=(1,2,224,224), extrinsics=(1,2,3,4),
                  intrinsics=(1,2,3,3), point_masks=(1,2,224,224))
    for item in train:
        from pathlib import Path
        raw, evidence = prepare_pair(Path(item['root']), item['pair'])
        layouts = checked_input(raw, TRAIN)
        value = tensor_batch(torch, raw)
        need(all(tuple(value[k].shape) == shape for k, shape in shapes.items()) and
             valid_depth(value['point_masks']), 'native train layout/mask differs')
        values.append(value)
        rows.append(dict(sequence=TRAIN, source_manifest_SHA256=item['pin'], layouts=layouts,
                         valid_pixels=evidence['valid_depth_pixels']))
    return values, dict(verified_pairs=7, physical_recordings={TRAIN:7}, rows=rows,
        input_order_SHA256=sha(encoded(rows)), split=declaration,
        heldout_geometry_loaded_by_training=False, formal_CO3D=False)
