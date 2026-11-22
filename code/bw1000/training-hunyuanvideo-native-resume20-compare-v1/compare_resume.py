"""Compare one HunyuanVideo restart to the original continuous Hygon run."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import stat
import sys


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
FRAMEWORK = ROOT / 'results/training-qwen-a68952c-swift-initial-CPU-v1/framework'
CHECKPOINT_PROVIDER = FRAMEWORK / 'generation/src/gencuda/hunyuanvideo_resume_checkpoint.py'
PROVIDER_SHA = 'b9ed0e8ae1c815f8bb32896d51d2d69b14805fa4962a8899b8ed5f509d18eb4e'
ORIGINAL_LABEL = 'training-hunyuanvideo-cached-native25-target-v3'
ORIGINAL = ROOT / 'results' / ORIGINAL_LABEL / 'outputs/hunyuanvideo-1.5/train' / ORIGINAL_LABEL
CONTINUOUS = {
    'steps.ndjson': (6642, '81ad37d6b57c939e10a5b4891102b47f138ae0f13687aaf598a89a58d2fea73b'),
    'checkpoint-20/state.pt': (7479771, '53f89e26678ff2853d2c8d7482ba63d80c6ec2c1a28e8f0881856e2c29ce3331'),
    'checkpoint-25/state.pt': (7479771, '18f00020affba2b0feb4468ac249f3a7d0cdbc92ae3c275b427bcee40afe77b3'),
}
CHECKPOINT_FIELDS = ('metadata', 'state', 'rng', 'adapter_sha256', 'integrity_sha256')
TOLERANCE_FIELDS = ('loss', 'grad_norm')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bytes(path, limit, identity=None):
    before = path.lstat()
    require(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            f'Unexpected artifact path or size: {path.name}')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    check = path.lstat()
    stamp = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    actual = len(raw), sha256(raw).hexdigest()
    require(stamp(check) == stamp(before) and before.st_size == len(raw), f'Artifact changed: {path.name}')
    require(identity is None or identity == actual, f'Artifact hash differs: {path.name}')
    return raw, actual


def steps(raw, first, last):
    decoded = [json.loads(line) for line in raw.splitlines()]
    expected = list(range(first, last + 1))
    require([row.get('global_step') for row in decoded] == expected
            and all(type(row['global_step']) is int for row in decoded), 'Optimizer step sequence differs')
    for row in decoded:
        require(all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in TOLERANCE_FIELDS)
                and row['grad_norm'] > 0, 'Nonfinite record or zero gradient')
    return decoded


def compare(reference_raw, resumed_raw, summary, final_reference, final_resumed, tree_hash):
    reference = steps(reference_raw, 1, 25)[20:]
    resumed = steps(resumed_raw, 21, 25)
    require(summary.get('external_resume') is True and type(summary.get('completed_global_step_at_start')) is int
            and summary['completed_global_step_at_start'] == 20, 'Native resume did not start at checkpoint 20')
    checks = summary.get('checks', {})
    require(summary.get('optimizer') == 'Muon' and checks.get('muon_optimizer') is True,
            'Native optimizer proof differs')
    require(checks.get('checkpoint_resume') is not False,
            'Optional in-process recovery validation failed')
    require(summary.get('invocation') == {'executed_global_steps': 5, 'executed_warmup_steps': 0,
                                          'executed_measured_steps': 5, 'first_global_step': 21},
            'Native external restore and suffix accounting differ')
    same = {key: tree_hash(final_reference[key]) == tree_hash(final_resumed[key]) for key in CHECKPOINT_FIELDS}
    errors = {key: max(abs(base[key] - candidate[key]) for base, candidate in zip(reference, resumed, strict=True))
              for key in TOLERANCE_FIELDS}
    return dict(passed=all(same.values()) and all(value == 0 for value in errors.values()),
                resumed_updates=5, first_step=21, last_step=25, max_absolute_error=errors,
                final_checkpoint_exact=same, performance_eligible=False, A100_comparison_qualified=False)


def quantify(resumed_root):
    require(resumed_root != ORIGINAL and resumed_root.is_relative_to(ROOT / 'results'),
            'Restored output must be a separate target results directory')
    original_steps, _ = read_bytes(ORIGINAL / 'steps.ndjson', 65536, CONTINUOUS['steps.ndjson'])
    _, start_pin = read_bytes(ORIGINAL / 'checkpoint-20/state.pt', 64 << 20, CONTINUOUS['checkpoint-20/state.pt'])
    reference_path = ORIGINAL / 'checkpoint-25/state.pt'
    resumed_path = resumed_root / 'checkpoint-25/state.pt'
    _, reference_pin = read_bytes(reference_path, 64 << 20, CONTINUOUS['checkpoint-25/state.pt'])
    _, resumed_pin = read_bytes(resumed_path, 64 << 20)
    resumed_steps, steps_pin = read_bytes(resumed_root / 'steps.ndjson', 65536)
    summary, summary_pin = read_bytes(resumed_root / 'summary.json', 65536)
    _, (_, provider_sha) = read_bytes(CHECKPOINT_PROVIDER, 128 << 10)
    require(provider_sha == PROVIDER_SHA, 'Deployed checkpoint provider bytes differ')
    sys.path.insert(0, str(FRAMEWORK / 'generation/src'))
    from gencuda import hunyuanvideo_resume_checkpoint as checkpoint

    require(Path(checkpoint.__file__).resolve(strict=True) == CHECKPOINT_PROVIDER,
            'Active checkpoint module differs from the deployed framework')
    final_reference = checkpoint.load_checkpoint(reference_path)
    final_resumed = checkpoint.load_checkpoint(resumed_path)
    read_bytes(reference_path, 64 << 20, reference_pin)
    read_bytes(resumed_path, 64 << 20, resumed_pin)
    result = compare(original_steps, resumed_steps, json.loads(summary), final_reference,
                     final_resumed, checkpoint.tree_sha256)
    return dict(result, checkpoint20_SHA256=start_pin[1], original_final_SHA256=reference_pin[1],
                restarted_final_SHA256=resumed_pin[1], restarted_steps_SHA256=steps_pin[1],
                restarted_summary_SHA256=summary_pin[1], actual_checkpoint_provider_SHA256=provider_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resumed-output', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    require(args.result.parent.is_relative_to(ROOT / 'results') and args.result.parent.resolve(strict=True) == args.result.parent,
            'Result must remain within the original project results filesystem')
    result = quantify(args.resumed_output)
    with args.result.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, allow_nan=False))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
