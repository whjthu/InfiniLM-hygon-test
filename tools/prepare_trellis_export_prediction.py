"""Bind an exact native TRELLIS GLB export to a frozen geometry scoring frame."""

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import tarfile
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / 'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'
EXPORTER = 'framework/generation/src/gencuda/trellis_cpu_texture_export.py'
ARCHIVE_SHA = '85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'
EXPORTER_SHA = 'f8fbbe622e6a792824b532906ae9d2c19405005519e1f2169d7fb0d97af35e85'
QUALITY = REPO / 'code/bw1000/training-trellis-public-mesh-quality-input-v1'
REGISTRY_SHA = '3055ccccd1d23f99d5ccf005dbd8229fdfb230f9e017c464010fef14879a68eb'
MAX_GLB = 64 * 1024**2
MAX_FACES = 300_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exact(path, maximum, wanted):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and digest(raw) == wanted, f'input pin differs: {path.name}')
    return raw


def original_export_functions(*, serializer=False):
    body = exact(ARCHIVE, 1024**2, ARCHIVE_SHA)
    with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as archive:
        source = archive.extractfile(EXPORTER).read()
    require(digest(source) == EXPORTER_SHA, 'original exporter source differs')
    parsed = ast.parse(source)
    schema = next(node.value for node in parsed.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'SCHEMA' for target in node.targets))
    names = {'require', 'glb_document', 'read_glb'}
    if serializer:
        names.update(('canonical', 'rotate', 'png_bytes', 'encode_glb', 'corner_geometry'))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in functions} == names, 'original geometry/export function set differs')
    scope = dict(np=np, io=io, json=json, struct=struct, Path=Path, Any=Any, SCHEMA=ast.literal_eval(schema))
    exec(compile(ast.Module(body=functions, type_ignores=[]), EXPORTER, 'exec'), scope)
    return scope


def quality_modules():
    modules = []
    for name, identity in (
        ('score_once', 'ebc1cf3eb4ec877975134d91849d7b5e28290adc8e4e452c5e86ae4f0e30f69a'),
        ('mesh_quality', '8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a'),
    ):
        path = QUALITY / (name + '.py')
        raw = exact(path, 32768, identity)
        module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        modules.append(module)
    return modules


class PinnedGLB:
    """Expose the already hashed bytes to the original reader without reopening a path."""

    def __init__(self, raw):
        self.raw = raw

    def stat(self):
        return SimpleNamespace(st_size=len(self.raw))

    def read_bytes(self):
        return self.raw


def normalize(vertices, reference, registration):
    rotation = np.asarray(registration['rotation'], dtype=np.float64)
    shift = np.asarray(registration['translation_m'], dtype=np.float64)
    scale = registration['meters_per_export_unit']
    require(rotation.shape == (3, 3) and np.isfinite(rotation).all()
            and np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-8)
            and abs(float(np.linalg.det(rotation)) - 1) < 1e-8, 'registration must be a proper rotation')
    require(shift.shape == (3,) and np.isfinite(shift).all()
            and type(scale) in (float, int) and math.isfinite(scale) and scale > 0, 'registration scale/shift invalid')
    # The native serializer has already applied (x,z,-y). Register those emitted axes once.
    world = (vertices.astype(np.float64) * scale) @ rotation.T + shift
    extent = float(reference['longest_extent_m'])
    require(math.isfinite(extent) and extent > 0, 'reference extent is invalid')
    prediction = ((world - reference['center_m']) / extent).astype(np.float32)
    require(np.isfinite(prediction).all(), 'registered geometry is nonfinite')
    return prediction


def sample(raw, texture_size, reference, registration, seed, reader, surface):
    require(0 < len(raw) <= MAX_GLB and type(texture_size) is int
            and 32 <= texture_size <= 2048 and texture_size & (texture_size - 1) == 0,
            'export exceeds the geometry bridge limits')
    arrays, _ = reader(PinnedGLB(raw), dict(max_bytes=MAX_GLB, max_faces=MAX_FACES, texture_size=texture_size))
    vertices = normalize(arrays[0], reference, registration)
    faces = arrays[3].reshape(-1, 3)
    points = surface(vertices, faces, seed)
    return points, dict(corners=len(vertices), triangles=len(faces), sampled_points=len(points))


def convert(glb, receipt, registration_evidence, reference_root, binding, output):
    score, metrics = quality_modules()
    registry = json.loads(exact(QUALITY / 'REFERENCE_INPUTS.json', 16384, REGISTRY_SHA))
    name = binding['object']
    spec = registry['objects'][name]
    require(binding['schema'] == 'TRELLIS_EXPORT_SCORING_BINDING_V1' and spec['split'] == binding['split']
            and binding['input_frame'] == 'native-exported-glTF-Yup-v1'
            and binding['output_frame'] == score.FRAME, 'split or coordinate binding differs')
    calibration = binding['independent_registration_evidence_SHA256']
    registration = json.loads(exact(registration_evidence, 16384, calibration))
    require(registration['schema'] == 'TRELLIS_FIXED_EXPORT_REGISTRATION_V1'
            and registration['object'] == name and registration['input_frame'] == binding['input_frame']
            and registration['reference_SHA256'] == spec['file']['SHA256'],
            'registration evidence differs from the frozen reference')
    raw = exact(glb, MAX_GLB, binding['GLB_SHA256'])
    receipt_raw = exact(receipt, 1024**2, binding['export_receipt_SHA256'])
    report = json.loads(receipt_raw)
    declared_sha = report.pop('payload_sha256')
    original = original_export_functions()
    require(digest(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) == declared_sha
            and report['schema'] == original['SCHEMA']
            and report['exporter_sha256'] == EXPORTER_SHA and report['output']['sha256'] == digest(raw)
            and Path(report['output']['path']).resolve() == glb.resolve()
            and report['runtime']['device'] == 'cpu' and report['runtime']['native_preflight_executed'] is True,
            'native export receipt does not prove the original GLB identity')
    reference = score.reference(reference_root, spec)
    seed = int(digest((REGISTRY_SHA + '/' + name + '/prediction-v1').encode())[:16], 16)
    points, counts = sample(raw, report['config']['texture_size'], reference, registration['transform'], seed,
                            original['read_glb'], metrics.sample_surface)
    require(report['product']['faces'] == counts['triangles'], 'native receipt face count differs')
    output.mkdir(mode=0o700)
    point_path = output / 'points.npz'
    with point_path.open('xb') as stream:
        np.savez_compressed(stream, points=points)
    manifest = dict(schema='TRELLIS_PUBLIC_REFERENCE_PREDICTIONS_V1', split=spec['split'], objects={
        name: dict(frame=score.FRAME, predictor_receipt_SHA256=digest(receipt_raw),
                   points=dict(path=str(point_path.resolve()), **score.pin(point_path.read_bytes())))})
    private = output / 'predictions.json'
    with private.open('xb') as stream:
        stream.write((json.dumps(manifest, sort_keys=True) + '\n').encode())
    return dict(**counts, prediction_manifest_SHA256=digest(private.read_bytes()), exporter_receipt_SHA256=digest(receipt_raw),
                independent_registration_evidence_SHA256=calibration, model_checkpoint_independently_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('glb', 'receipt', 'registration-evidence', 'reference-root', 'binding', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--binding-sha256', required=True)
    args = parser.parse_args()
    binding = json.loads(exact(args.binding, 16384, args.binding_sha256))
    print(json.dumps(convert(args.glb, args.receipt, args.registration_evidence, args.reference_root,
                             binding, args.output), allow_nan=False))


if __name__ == '__main__':
    main()
