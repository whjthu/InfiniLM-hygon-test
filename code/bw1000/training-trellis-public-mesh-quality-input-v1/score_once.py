"""Score an explicit future prediction set; never generate/fill missing predictions."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

FRAME = 'reference-normalized-glTF-Yup-v1'
D = Path(__file__).resolve().parent


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pin(raw):
    return {'bytes': len(raw), 'SHA256': hashlib.sha256(raw).hexdigest()}


def bounded(path, maximum, expected=None):
    with Path(path).open('rb') as f:
        before = os.fstat(f.fileno())
        raw = f.read(maximum + 1)
        after = os.fstat(f.fileno())
    current = Path(path).stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    need(identity(before) == identity(after) == identity(current), 'Input changed during read')
    need(len(raw) <= maximum, 'File byte budget exceeded')
    if expected is not None:
        need(pin(raw) == expected, 'Frozen bytes/SHA differs')
    return raw


def arrays(raw, keys):
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        need(len(members) == len(keys) and {x.filename for x in members} == {k + '.npy' for k in keys}, 'Exact NPZ keys required')
        need(all(x.file_size <= 2 * 1024 * 1024 and not x.flag_bits & 1 for x in members)
             and sum(x.file_size for x in members) <= 3 * 1024 * 1024, 'NPZ decoded byte budget exceeded')
        for member in members:
            with z.open(member) as f:
                version = np.lib.format.read_magic(f)
                need(version in ((1, 0), (2, 0)), 'NPY header version not supported')
                read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = read_header(f, max_header_size=4096)
                need(not dtype.hasobject and dtype.kind in 'fui' and len(shape) <= 2
                     and all(type(x) is int and 0 <= x <= 300000 for x in shape)
                     and math.prod(shape) * dtype.itemsize <= 2 * 1024 * 1024,
                     'NPY allocation/object budget exceeded')
                need(f.tell() + math.prod(shape) * dtype.itemsize == member.file_size, 'NPY payload length differs')
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


def reference(root, spec):
    import numpy as np
    a = arrays(bounded(root / spec['name'], 1024 * 1024, spec['file']),
               ('vertices', 'faces', 'reference_points', 'center_m', 'longest_extent_m'))
    v, f, p = a['vertices'], a['faces'], a['reference_points']
    need(v.shape == (spec['vertices'], 3) and v.dtype == np.float32 and np.isfinite(v).all(), 'Reference vertices differ')
    need(f.shape == (spec['faces'], 3) and f.dtype.kind in 'ui' and int(f.max()) < len(v), 'Reference triangles differ')
    need(p.shape == (2048, 3) and p.dtype == np.float32 and np.isfinite(p).all(), 'Full reference points required')
    need(a['center_m'].shape == (3,) and np.isfinite(a['center_m']).all()
         and a['longest_extent_m'].shape == () and np.isfinite(a['longest_extent_m']).all()
         and float(a['longest_extent_m']) > 0, 'Reference frame invalid')
    return a


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--predictions-SHA256', required=True)
    p.add_argument('--split', choices=('train', 'dev'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # This new recipe never fetches, decodes a model, uses formal T10 operands,
    # aligns each prediction to the target, or consumes another split's targets.
    registry_raw = bounded(D / 'REFERENCE_INPUTS.json', 16384)
    registry = json.loads(registry_raw)
    raw = bounded(a.predictions, 16384)
    need(hashlib.sha256(raw).hexdigest() == a.predictions_SHA256, 'Prediction manifest SHA differs')
    predictions = json.loads(raw)
    need(predictions.get('schema') == 'TRELLIS_PUBLIC_REFERENCE_PREDICTIONS_V1'
         and predictions.get('split') == a.split, 'Explicit prediction schema/split required')
    selected = {k: v for k, v in registry['objects'].items() if v['split'] == a.split}
    need(isinstance(predictions.get('objects'), dict)
         and not set(predictions['objects']) - set(selected), 'Unknown or cross-split prediction group')
    from mesh_quality import point_metrics
    out = {'schema': 'TRELLIS_PUBLIC_MESH_QUALITY_NEW_RECIPE_SCORE_V1', 'split': a.split,
           'planned_objects': len(selected), 'completed_objects': 0, 'failed_objects': 0,
           'planned_reference_points': 2048 * len(selected), 'objects': {},
           'prediction_manifest': pin(raw), 'reference_registry': pin(registry_raw),
           'model_prediction_provenance_independently_validated': False,
           'model_quality_qualified': False, 'formal17_GO': False, 'READY': False}
    for name, spec in selected.items():
        item = {'planned_reference_points': 2048, 'passed': False}
        out['objects'][name] = item
        try:
            need(time.monotonic() - started < 60, 'Local scoring budget elapsed')
            r = reference(a.reference_root, spec)
            record = predictions['objects'].get(name)
            need(isinstance(record, dict), 'Missing prediction; denominator retained')
            need(record.get('frame') == FRAME, 'Prediction must already use the frozen reference frame; no fit')
            receipt = record.get('predictor_receipt_SHA256')
            need(isinstance(receipt, str) and len(receipt) == 64
                 and all(c in '0123456789abcdef' for c in receipt), 'Predictor receipt identity required')
            f = record['points']
            q = Path(f['path'])
            need(q.is_absolute() and q.resolve() != (a.reference_root / spec['name']).resolve(), 'Separate absolute prediction artifact required')
            b = bounded(q, 1024 * 1024, {'bytes': f['bytes'], 'SHA256': f['SHA256']})
            actual = arrays(b, ('points',))['points']
            item.update(point_metrics(actual, r['reference_points']), prediction=pin(b),
                        predictor_receipt_SHA256=receipt)
            need(item['passed'], 'Prediction shape/nonfinite failure; do not drop points')
            out['completed_objects'] += 1
        except Exception as e:
            item.update(passed=False, error_type=type(e).__name__, reason=str(e)[:256],
                        Chamfer_L1=None, Fscore_at_1percent=0.)
            out['failed_objects'] += 1
    out.update(passed=out['completed_objects'] == out['planned_objects'],
               wall_seconds=time.monotonic() - started,
               aggregate_policy='All planned objects retained; any failure prohibits complete quality conclusion. Per-object values only; no success-only average.')
    with a.output.open('x', encoding='utf8') as f:
        json.dump(out, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: out[k] for k in ('passed', 'planned_objects', 'completed_objects', 'failed_objects', 'model_quality_qualified')}))
    return 0 if out['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
