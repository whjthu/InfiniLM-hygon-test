"""Record original native CPU template identities from a frozen VLM recipe."""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hygon_vlm_recipe import A100_LAUNCH_SHA256, build_launch
from vlm.runtime.template_inputs import capture_native_inputs, settings_from_launch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=sorted(A100_LAUNCH_SHA256))
    for name in ('a100-launch', 'model-root', 'model-manifest', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dataset-sha256', required=True)
    parser.add_argument('--epoch', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args(argv)
    launch = build_launch(args.case, args.a100_launch.read_bytes(), PurePosixPath('/unused/swift'),
        PurePosixPath(args.model_root.parent.as_posix()), PurePosixPath(args.dataset.as_posix()),
        PurePosixPath('/unused/plugins'), PurePosixPath(args.output.as_posix()), 29501)
    if launch['required_assets']['model_directory'] != args.model_root.as_posix():
        raise ValueError('processor directory basename differs from original model')
    manifest = json.loads(args.model_manifest.read_bytes())
    result = capture_native_inputs(settings_from_launch(launch), args.model_root, args.dataset,
        args.output, manifest['files'], args.dataset_sha256, epoch=args.epoch, deadline=args.deadline)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
