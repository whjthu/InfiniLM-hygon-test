"""Current T1 HC arithmetic and conditional envelope limits, never Model peaks.

Only reads published scalar metadata. No runtime, GPU, weights, or raw trace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def analyze(root):
    day = root / 'results/bw1000/2026-11-02'
    profile, profile_sha = read(day / 'Deep353r1-1261-qualified-current-C1-summary.json')
    source, source_sha = read(day / 'Deep353-current-HC-C4-exact-source-binding.json')
    hardware, hardware_sha = read(root / 'analysis/bw1000/hardware-platform-reference-r9038.json')
    achieved, achieved_sha = read(root / 'results/bw1000/2026-10-24/achieved-hardware-references.json')
    assert source['Native_SHA256'] == 'afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
    assert source['HC']['source_SHA256'] == '4b178cd98cd9e8dd2e24614bf022c1297d6332f22b2c790530fbb06866c5d313'
    assert profile['passed'] and profile['new_performance_gain'] is False
    windows = []
    for request in profile['profiles']:
        for sample in request['profiles']:
            assert sample['native_window_closed'] and sample['owner_event_validation_passed']
            assert sample['owner_rows'] == sample['inner_rows'] == 344
            assert sample['geometry']['tokens'] == 1
            owners = []
            for device in range(8):
                values = {}
                for namespace in ['old_perstage_perdevice_ms', 'old_host_wall_perstage_perdevice_ms',
                                  'old_thread_cpu_perstage_perdevice_ms']:
                    stages = sample[namespace]
                    assert all(len(stages[k]) == 8 for k in
                               ['decoder_envelope', 'attention_HC_pre', 'FFN_HC_pre'])
                    envelope = stages['decoder_envelope'][device]
                    hc = stages['attention_HC_pre'][device] + stages['FFN_HC_pre'][device]
                    assert 0 < hc < envelope
                    values[namespace] = dict(envelope_ms=envelope, HC_pre_ms=hc,
                                             HC_pre_fraction=hc/envelope)
                stream = values['old_perstage_perdevice_ms']
                # Conditional Amdahl calculation: other ranges and overlap stay fixed.
                # Not a proven cross-rank/model critical-path bound.
                values['perfect_HC_removal_stream_envelope_factor'] = 1/(1-stream['HC_pre_fraction'])
                owners.append(dict(device=device, **values))
            windows.append(dict(request_index=request['request_index'],
                                decode_call_1based=sample['geometry']['decode_call_1based'],
                                owners=owners))
    assert len(windows) == 4
    factors = [o['perfect_HC_removal_stream_envelope_factor'] for w in windows for o in w['owners']]
    fractions = [o['old_perstage_perdevice_ms']['HC_pre_fraction'] for w in windows for o in w['owners']]
    # Current source: each of 43 layers has attention and FFN HC pre, each with
    # F32 [1,16384] x [16384,24]. FMA is counted as two operations.
    rows, inputs, outputs, layers, per_layer = 1, 16384, 24, 43, 2
    flops = 2*rows*inputs*outputs
    fn_bytes = inputs*outputs*4
    operands = fn_bytes + rows*inputs*4 + rows*outputs*4
    declared = hardware['primary_source']['declared']
    caps = achieved['achieved']
    return dict(
        scope='Current afc6 T1 HC only; four instrumented windows; not a physical or Model peak',
        source=source['HC'], Native_SHA256=source['Native_SHA256'],
        input_refs_SHA256=dict(profile=profile_sha, source_binding=source_sha,
                               declared_reference=hardware_sha, achieved_reference=achieved_sha),
        current_F32_GEMV=dict(input_shape=[1, 16384], weight_shape=[24, 16384],
                              FLOPs_per_HC=flops, calls_per_owner_decode=layers*per_layer,
                              FLOPs_per_owner_decode=flops*layers*per_layer,
                              fn_bytes_per_HC=fn_bytes,
                              unreused_operand_bytes_per_HC=operands,
                              unreused_arithmetic_intensity_FLOP_per_byte=flops/operands),
        source_dispatches=dict(T1_BF16_high_level_operator_and_custom_calls=23,
                               category_counts=dict(cast_and_norm=6, F32_Gemm=1,
                                                    pre_gate=4, post_gate=4,
                                                    comb_affine_softmax_epsilon=4,
                                                    retained_Sinkhorn=1, collapse_and_cast=3),
                               limit='Source-level entries, not measured kernel launches. SDK can launch multiple kernels. t512 padding is ineligible.'),
        conditional_reference_seconds_per_HC=dict(
            GEMV_FLOPs_over_platform_FP32=flops/(declared['FP32_TFLOPS']*1e12),
            unreused_operands_over_platform_HBM=operands/(declared['HBM_bandwidth_TB_s']*1e12),
            GEMV_FLOPs_over_achieved_large_GEMM=flops/(caps['FP32_effective_TFLOPS']*1e12),
            unreused_operands_over_achieved_read=operands/(caps['read_TB_s']*1e12)),
        reference_limit='These are conditional arithmetic/traffic reference divisions, not lower bounds on real HC wall or attained T1 rates. Cache/real HBM traffic, launch, synchronization, allocator, and other HC operations are unmeasured. Platform 60TF/1.6TB is not exact-SKU certification; achieved large GEMM/read is capability evidence, not a peak. BF16 physical peak remains unverified.',
        windows=windows,
        perfect_removal_summary=dict(HC_stream_fraction_min=min(fractions),
                                     HC_stream_fraction_max=max(fractions),
                                     factor_min=min(factors), factor_max=max(factors),
                                     factor_median=statistics.median(factors)),
        conditional_limit='Removing both HC pre spans with all remaining durations/overlap fixed gives these owner stream-envelope factors. Not a complete end-to-end speedup, global physical upper bound, or prediction. Four observed instrumented windows are not all 97 decode calls. Never sum different owners or nested ranges.',
        decision='Proceed with 358 whole-HC component measurement, but HC alone does not explain a >5x decode target gap under this fixed-rest scenario. Keep distinct cache/index candidate359 and system parallelism. A fast epilogue kernel alone is insufficient: require full HC drained wall, path hits, numerics, drift and eventual Model coverage.',
        new_Model_gain=False, GPU_work_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.root)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(value['perfect_removal_summary']))
