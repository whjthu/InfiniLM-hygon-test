"""Run resident VGGT camera-head inference on a byte-pinned CO3D test plan."""

import argparse
from hashlib import sha256
from importlib import import_module
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import sys

from vggt_co3d_effect import CASE, UPSTREAM, planned_pairs, read_exact, require, sha
from vggt_co3d_pose_receipts import COORDINATES, checked_pose


SOURCE_MANIFEST_SHA256 = '64380b341ce377fb7e4b24c3f4f5986b6557b5339f4f7db57950dfa7e66b04c0'
BASE_WEIGHTS_SHA256 = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
PLATFORM_ARMS = {'A100': ('A100',), 'Hygon': ('Hygon_before', 'Hygon')}


def resolved_member(root, name):
    require(isinstance(name, str) and '\\' not in name, 'expected a relative POSIX asset path')
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and '..' not in path.parts and ':' not in name,
            'asset path escapes the declared root')
    resolved = (root / path).resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), 'asset path escapes the declared root')
    return resolved


def read_media_plan(plan, media):
    require(plan.get('case') == CASE and sha(plan.get('dataset_SHA256')) and
            media.get('dataset_SHA256') == plan['dataset_SHA256'], 'CO3D media dataset identity differs')
    planned_pairs(plan)
    expected = [(row['category'], row['sequence'], row['frame_ids']) for row in plan['sequences']]
    actual = []
    for row in media['sequences']:
        frames = row['frames']
        require(all(type(frame['frame_id']) is int and sha(frame['SHA256']) and
                    isinstance(frame['image'], str) and frame['image'] for frame in frames),
                'invalid CO3D media frame inventory')
        require(len({frame['image'] for frame in frames}) == len(frames), 'duplicate image in the same sequence')
        actual.append((row['category'], row['sequence'], [frame['frame_id'] for frame in frames]))
    require(actual == expected, 'CO3D media sequence or frame order differs from plan')
    return media['sequences']


def verify_source(root, manifest_path):
    manifest = json.loads(read_exact(manifest_path, SOURCE_MANIFEST_SHA256))
    files = {row['name']: row['SHA256'] for row in manifest['files'] if row['name'].startswith('vggt/')}
    require({'vggt/models/vggt.py', 'vggt/utils/load_fn.py', 'vggt/utils/pose_enc.py'} <= files.keys(),
            'missing original model and preprocessing source identities')
    for name, digest in files.items():
        read_exact(resolved_member(root, name), digest)
    observed = {path.relative_to(root).as_posix() for path in (root / 'vggt').rglob('*.py')}
    require(observed == {name for name in files if name.endswith('.py')}, 'unbound Python source in VGGT package')


def load_weights(torch, path, wanted):
    require(sha(wanted), 'expected a checkpoint hash')
    with path.open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        require(digest.hexdigest() == wanted, 'checkpoint byte identity differs')
        stream.seek(0)
        return torch.load(stream, map_location='cpu', weights_only=True)


class NativeVGGT:
    def __init__(self, source_root, base, platform):
        import torch

        require(torch.cuda.device_count() == 1, 'CO3D evaluation requires one visible accelerator')
        properties = torch.cuda.get_device_properties(0)
        if platform == 'Hygon':
            require(bool(torch.version.hip) and getattr(properties, 'gcnArchName', '').split(':')[0] == 'gfx936',
                    'expected the Hygon gfx936 runtime')
        else:
            require(torch.version.cuda and not torch.version.hip and 'A100' in properties.name,
                    'expected the NVIDIA A100 runtime')
        require(not any(name == 'vggt' or name.startswith('vggt.') for name in sys.modules),
                'VGGT source was imported before verification')
        sys.path.insert(0, str(source_root))
        modules = [import_module(name) for name in ('vggt.models.vggt', 'vggt.utils.load_fn', 'vggt.utils.pose_enc')]
        require(all(Path(module.__file__).resolve().is_relative_to(source_root) for module in modules),
                'VGGT module loaded outside the verified source tree')
        model_type = modules[0].VGGT
        self.preprocess = modules[1].load_and_preprocess_images
        self.decode = modules[2].pose_encoding_to_extri_intri
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.runtime = dict(torch=str(torch.__version__), hip=torch.version.hip, cuda=torch.version.cuda,
                            device=properties.name, precision='bf16-forward/fp32-pose-decode')
        self.model = model_type(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
        state = load_weights(torch, base, BASE_WEIGHTS_SHA256)
        require(isinstance(state, dict), 'expected an original VGGT state dictionary')
        weights = state.get('model', state)
        missing, extra = self.model.load_state_dict(weights, strict=False)
        require(not missing and all(key.startswith(('depth_head.', 'point_head.', 'track_head.')) for key in extra),
                'original VGGT checkpoint differs from camera-only model')
        self.model.requires_grad_(False).eval().cuda()

    def install_head(self, filename, digest):
        state = load_weights(self.torch, filename, digest)
        self.model.camera_head.load_state_dict(state, strict=True)

    def predict(self, images):
        torch = self.torch
        inputs = self.preprocess(images, mode='crop').cuda()
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pose = self.model(inputs)['pose_enc']
            require(tuple(pose.shape) == (1, len(images), 9) and pose.is_cuda, 'VGGT camera output shape or device differs')
            pose, _ = self.decode(pose.float(), inputs.shape[-2:], build_intrinsics=False)
        return pose[0].cpu().tolist()


def export_sequences(sequences, media_root, runner):
    for sequence in sequences:
        frames = sequence['frames']
        images = [BytesIO(read_exact(resolved_member(media_root, frame['image']), frame['SHA256'])) for frame in frames]
        try:
            poses = runner.predict(images)
        finally:
            for stream in images:
                stream.close()
        converted = []
        for frame, pose in zip(frames, poses, strict=True):
            result = dict(frame_id=frame['frame_id'])
            try:
                checked_pose(pose)
            except ValueError:
                result.update(status='failed', reason='model returned an invalid camera extrinsic')
            else:
                result.update(status='scored', extrinsic=pose)
            converted.append(result)
        yield dict(category=sequence['category'], sequence=sequence['sequence'], frames=converted)


def write_receipt(output, filename, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    with (output / filename).open('xb') as stream:
        stream.write(raw)
    return dict(path=filename, SHA256=sha256(raw).hexdigest())


def generate(args):
    plan = json.loads(read_exact(args.plan, args.plan_sha256))
    media_sha = plan['media_manifest_SHA256']
    media = json.loads(read_exact(args.media, media_sha))
    sequences = read_media_plan(plan, media)
    source_root, media_root = args.source_root.resolve(strict=True), args.media_root.resolve(strict=True)
    verify_source(source_root, args.source_manifest)
    require(args.output.parent.resolve(strict=True).is_dir() and not args.output.exists(),
            'CO3D output requires a fresh directory')
    runner = NativeVGGT(source_root, args.base_weights, args.platform)
    args.output.mkdir(mode=0o700)
    arms = {}
    for arm in PLATFORM_ARMS[args.platform]:
        checkpoint = BASE_WEIGHTS_SHA256
        if arm != 'Hygon_before':
            checkpoint = args.head_sha256
            runner.install_head(args.head, checkpoint)
        receipt = dict(arm=arm, checkpoint_SHA256=checkpoint, coordinates=COORDINATES,
                       pose_evaluator_commit=UPSTREAM, evaluation_plan_SHA256=args.plan_sha256,
                       media_manifest_SHA256=media_sha, source_manifest_SHA256=SOURCE_MANIFEST_SHA256,
                       base_weights_SHA256=BASE_WEIGHTS_SHA256, runtime=runner.runtime,
                       sequences=list(export_sequences(sequences, media_root, runner)))
        arms[arm] = dict(write_receipt(args.output, arm + '-poses.json', receipt), checkpoint_SHA256=checkpoint)
    return write_receipt(args.output, 'arm-files.json', dict(evaluation_plan_SHA256=args.plan_sha256, arms=arms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'media', 'media-root', 'source-root', 'source-manifest', 'base-weights', 'head', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('plan-sha256', 'head-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--platform', choices=PLATFORM_ARMS, required=True)
    args = parser.parse_args()
    require(sha(args.head_sha256), 'expected a SHA-256 for the trained camera head')
    print(json.dumps(generate(args)))


if __name__ == '__main__':
    main()
