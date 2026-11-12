"""Test one BF16 GEMM reduction policy with the frozen native SDXL update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hygon_sdxl_first_step import replay, require, sha, verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-only', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'precision-probe output already exists')
    args.output.mkdir()
    binding = verify_reference(args.reference, args.policy)
    import torch

    require(bool(torch.version.hip), 'expected the original HIP runtime')
    require(not torch.cuda.is_initialized(), 'probe must start before device initialization')
    backend = torch.backends.cuda.matmul
    before = backend.allow_bf16_reduced_precision_reduction
    require(type(before) is bool, 'missing native BF16 reduction policy')
    base = dict(**binding, original_policy=before, requested_policy=False,
                policy='torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction',
                torch=torch.__version__, HIP=torch.version.hip,
                replay_source_SHA256=sha(Path(__file__).with_name('hygon_sdxl_first_step.py')),
                qualification='Native policy value is recorded; actual GEMM accumulator implementation is not inferred.')
    if args.policy_only or not before:
        base.update(state='POLICY_PREFLIGHT' if before else 'ALREADY_DISABLED_NO_GPU_RUN',
                    GPU_initialized=torch.cuda.is_initialized(), executed_update=False)
        (args.output / 'policy.json').write_text(json.dumps(base, indent=2) + '\n')
        return
    require(args.weights is not None, 'native update requires original weights')
    from torch.profiler import ProfilerActivity, profile
    backend.allow_bf16_reduced_precision_reduction = False
    result = dict(base, executed_update=False, state='STARTED')
    try:
        require(backend.allow_bf16_reduced_precision_reduction is False,
                'requested native BF16 reduction policy was not applied')
        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            replay(args.reference, args.weights, args.output / 'replay', binding)
        operators = {row.key: row.count for row in profiler.key_averages()
                     if row.key.startswith(('aten::_scaled_dot_product', 'aten::scaled_dot_product'))
                     or row.key in {'aten::mm', 'aten::bmm', 'aten::addmm'}}
        require(backend.allow_bf16_reduced_precision_reduction is False,
                'native reduction policy changed during the update')
        result.update(state='ONE_NATIVE_UPDATE_RETURNED', executed_update=True,
                      observed_policy=False, operators=operators)
    except BaseException as error:
        result.update(state='FAILED', error_type=type(error).__name__)
        raise
    finally:
        backend.allow_bf16_reduced_precision_reduction = before
        result['restored_policy'] = backend.allow_bf16_reduced_precision_reduction
        (args.output / 'policy.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
