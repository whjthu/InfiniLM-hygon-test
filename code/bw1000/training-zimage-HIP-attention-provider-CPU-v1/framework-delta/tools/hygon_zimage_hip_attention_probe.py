"""Run a pinned HIP SDK diagnostic on actual model-observed attention shapes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'generation/src'))
from gencuda.zimage_hip_attention_probe import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    value = run(args.output, json.loads(args.binding.read_bytes()), epoch=args.epoch, deadline=args.deadline)
    print(json.dumps({k: value[k] for k in ['passed', 'counters', 'actual_native_model_loads', 'Model_GO']}))
    raise SystemExit(0 if value['passed'] else 1)
