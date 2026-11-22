"""Native base/new-head pose metric. Seal four predictions before loading holdout truth."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from split_contract import descriptors, need, read, sha, encoded, child

PROVIDERS = {
    'models/vggt.py':'dbc92d6214882161d9042b1bc60cc932a9fd6bb609fc913d14906328f7833607',
    'models/aggregator.py':'c994d7231eda2380f208355a64bce3cd7c4b97ef06393355e632415bfc1513e5',
    'heads/camera_head.py':'d177f3d64d53210209b7bbb8496faaaec91f3fc51d04a4ee8d9d6b70f053e132',
    'utils/pose_enc.py':'845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b',
    'utils/rotation.py':'ad3a904e243c6654f0aa5f1d0206ace5f2d61011bc1bf8f492b40e0b76cb2070',
}
SCORER_SHA = '6454c9a5f10459e5f2eb8b268918d1303d3f04c2038a12cca22c20ac87b58e22'


def save(path, value):
    raw = encoded(value)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(bytes=len(raw), SHA256=sha(raw))


def load_state(torch, path, expected):
    import hashlib
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1<<20), b''):
            digest.update(chunk)
        need(digest.hexdigest() == expected, 'native state file SHA differs')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
        after = os.fstat(stream.fileno())
    identity = lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    need(identity(before) == identity(after) == identity(path.stat()), 'state changed during load')
    return state


def run(data, weights, upstream, trained_head, head_pin, head_module_sha, output):
    import numpy as np
    import torch
    from gencuda.backends import vggt_training as backend
    from gencuda.runtime import autocast_bf16, synchronize
    from gencuda.tum_vggt_pair import parse_index, read_frame, prepare_pair

    output.mkdir()
    _, held, declaration = descriptors(data)
    for name, pin in PROVIDERS.items():
        read(upstream/'vggt'/name, pin, 65536)
    scorer_path = Path(__file__).with_name('score_vggt_tum_pair.py')
    raw = read(scorer_path, SCORER_SHA, 65536)
    scope = {'__name__':'frozen_pose_score', '__file__':str(scorer_path)}
    exec(compile(raw, str(scorer_path), 'exec'), scope)
    need(torch.cuda.device_count() == 1 and torch.version.hip == '6.3.26113', 'one original HIP device required')
    sys.path.insert(0, str(upstream))
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    state = load_state(torch, weights/'model.pt', scope['WEIGHT_SHA'])
    model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    missing, extra = model.load_state_dict(state.get('model', state), strict=False)
    del state
    need(not missing and all(k.startswith(('point_head.','depth_head.','track_head.')) for k in extra),
         'original camera-only base structure differs')
    need(backend._module_sha(model.camera_head) == scope['BASE_HEAD'] and
         backend._module_sha(model.aggregator) == scope['AGGREGATOR'], 'original base module identity differs')
    model.requires_grad_(False).eval().cuda()
    inputs = []
    for item in held:
        directory, pair = Path(item['root']), item['pair']
        times, names = parse_index(read(directory/'rgb.txt', pair['files_SHA256']['rgb.txt'], 1<<20), 'rgb')
        images = [read_frame(read(child(directory, names[i]), pair['files_SHA256'][names[i]], 4<<20), 'rgb')
                  for i in pair['rgb_indices']]
        array = np.stack(images)
        need(array.shape == (2,224,224,3) and array.dtype == np.uint8, 'holdout RGB native geometry differs')
        inputs.append((item, array))
    predictions = []
    for arm in ('baseline','updated'):
        if arm == 'updated':
            state = load_state(torch, trained_head, head_pin)
            model.camera_head.load_state_dict(state, strict=True)
            del state
        expected = scope['BASE_HEAD'] if arm == 'baseline' else head_module_sha
        need(backend._module_sha(model.camera_head) == expected, 'active camera-head module differs')
        for item, array in inputs:
            row = dict(arm=arm, sequence=item['pair']['sequence'], source_manifest_SHA256=item['pin'],
                       frame_ids=item['pair']['rgb_indices'], state='NOT_COMPLETED', extrinsics=None)
            started = perf_counter()
            try:
                tensor = torch.from_numpy(array.astype(np.float32)).permute(0,3,1,2).div(255).unsqueeze(0).cuda()
                with torch.inference_mode(), autocast_bf16():
                    prediction = model(tensor)['pose_enc']
                synchronize()
                need(tuple(prediction.shape) == (1,2,9) and prediction.is_cuda, 'native camera output differs')
                value, _ = pose_encoding_to_extri_intri(prediction.float(), (224,224), build_intrinsics=False)
                result = value[0].cpu().numpy()
                scope['relative_pose'](result)
                row.update(state='PREDICTED', extrinsics=result.tolist())
            except (ValueError, RuntimeError) as error:
                row.update(state='FAILED_PREDICTION', error_type=type(error).__name__)
            row['forward_seconds'] = perf_counter()-started
            predictions.append(row)
        need(backend._module_sha(model.camera_head) == expected and
             backend._module_sha(model.aggregator) == scope['AGGREGATOR'], 'inference changed frozen state')
    need(len(predictions) == 4, 'all four prediction slots must be retained')
    sealed = save(output/'predictions-sealed-private.json', dict(declaration=declaration, rows=predictions,
        gold_read=False, planned_predictions=4, training_updates_in_evaluator=0))
    # First access to pose/depth truth in this evaluator occurs only below the durable seal.
    scored = []
    for item, _ in inputs:
        truth, _ = prepare_pair(Path(item['root']), item['pair'])
        row = dict(sequence=item['pair']['sequence'], frame_ids=item['pair']['rgb_indices'])
        for arm in ('baseline','updated'):
            prediction, = [r for r in predictions if r['arm'] == arm and r['sequence'] == row['sequence']]
            errors = dict(pair_error_deg=None)
            if prediction['extrinsics'] is not None:
                errors = scope['angular_errors'](prediction['extrinsics'], truth['extrinsics'])
            row[arm] = errors
        scored.append(row)
    summary = scope['summarize_development'](scored)
    save(output/'metric-private.json', dict(rows=scored, summary=summary, prediction_seal=sealed,
        planned_pairs=2, planned_predictions=4, whole_recording_split=True,
        candidate_relative_holdout_only=True, fresh_global_blind=False, formal_CO3D=False,
        scorer_SHA256=SCORER_SHA, selection_SHA256=declaration['selection_SHA256']))
    return dict(summary=summary, prediction_seal=sealed,
                metric_SHA256=sha((output/'metric-private.json').read_bytes()),
                planned_pairs=2, planned_predictions=4, formal_CO3D=False, global_fresh_blind=False)
