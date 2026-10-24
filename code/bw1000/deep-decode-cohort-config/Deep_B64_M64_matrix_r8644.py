"""Immutable 0e051 configuration comparison; two frozen prompts duplicated to 64 rows."""
import ctypes
import hashlib
import json
import time
import numpy as np

REFERENCE_SHA = 'ba577de61c056f6e701005f9b1622cfac284d38e1671bb480087187544f35797'
ORDER = [64, 32, 32, 64]  # ON, OFF, OFF, ON; only cohort width changes.


def run(g):
    r, lib, ctx, cfg, check, report, path = [g[k] for k in ['r', 'lib', 'ctx', 'cfg', 'check', 'report', 'path']]
    refpath = r / 'incoming/Deep-B32-samegeom-reference-r7269.json'
    assert hashlib.sha256(refpath.read_bytes()).hexdigest() == REFERENCE_SHA
    ref = json.loads(refpath.read_text())
    assert report['library_sha256'] == ref['DSO_SHA256'] == '0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
    assert len(ref['input_ids']) == 32 and all(len(x) == 512 for x in ref['input_ids'])
    assert all(ref['input_ids'][i] == ref['input_ids'][i % 2] for i in range(32))
    assert all(ref['generated_ids'][i] == ref['generated_ids'][i % 2] for i in range(32))
    prompts = [ref['input_ids'][i % 32] for i in range(64)]
    gold = np.array([ref['generated_ids'][i % 32] for i in range(64)], dtype=np.int64)
    assert gold.shape == (64, 128) and not np.any(gold == cfg['eos_token_id'])
    report.update(config_trials_r8644=[], performance_adopted=False, performance_qualified=False,
                  outcome='RUNNING', input_scope='64 logical rows, exactly two frozen 512-token prompts repeated32 times; not64 independent prompts',
                  reference_SHA256=REFERENCE_SHA, reference_scope='Frozen679/current699 regression; duplicate-row extension is a gate, not assumed cross-batch parity',
                  order=ORDER, firstpair_repeat_threshold=1.01, profiling_enabled=False,
                  host_buffer_count=1, host_buffer_payload_bytes=64 * cfg['vocab_size'] * 2,
                  memory_scope='cudaMemGetInfo snapshots only, not peak memory or an admission proof; OOM fails without retry',
                  scope='Same0e051/TP8PP1/cache128x1024/precision5/projection16/C4=16/layout0HC0tail0; only decode cohort M32vsM64; prefillchunk4 fixed; cap128.')

    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(path)

    def memory(label):
        value = {'label': label, 'device_free_total_bytes': [list(g['torch'].cuda.mem_get_info(d)) for d in range(8)]}
        report.setdefault('device_memory_snapshots', []).append(value)
        save()

    def policy(tile):
        g['layout_mode'] = g['HC_mode'] = g['tail_SUM_mode'] = 0
        g['projection_tile_mode'] = g['c4_tile_mode'] = tile
        g['precision_mode'] = 5
        check(lib.modern_deep_precision_policy_r3443(5))
        check(lib.modern_deep_layout_policy_r7052(0))
        check(lib.deep_hc_canonical_policy_r7130(0))
        check(lib.deep_moe_tail_set_r7211(0))
        check(lib.modern_deep_c4_tile_policy_r5920(tile))
        g['os'].environ['INFINILM_DEEP_OUTPUT_TILE_R4808'] = str(tile)

    def counters():
        return {'projection': [int(lib.deep_output_tile_calls_r4808(d)) for d in range(8)],
                'C4': [[int(lib.modern_deep_c4_tile_calls_r5920(d, k)) for k in range(2)] for d in range(8)]}

    memory('after_load_before_QA')
    policy(0)
    g['run_chat']('before')
    assert report['finite_before_passed'], 'before_QA_failed'
    policy(16)
    buffer = np.full((64, cfg['vocab_size']), 0x7fc0, dtype=np.uint16)
    emitted = np.full((128, 64), -999, dtype=np.int64)
    completion = np.full((128, 64), np.nan, dtype=np.float64)
    csa = sum(int(x) == 4 for x in cfg['compress_ratios'][:43])
    for trial, micro in enumerate(ORDER):
        check(lib.modern_native_engine_reset(ctx))
        emitted.fill(-999)
        completion.fill(np.nan)
        entry = {'trial': trial, 'micro': micro, 'batch': 64, 'gate26_passed': False,
                 'full_IDs_EOS_exact': False, 'CAPI_windows': [], 'generated_ids': [], 'complete': False}
        report['config_trials_r8644'].append(entry)
        memory(f'trial{trial}-M{micro}-before')
        before = counters()
        step_times = []
        validation_seconds = 0.0
        started = time.perf_counter()
        for step in range(128):
            width = 4 if step == 0 else micro
            for a in range(0, 64, width):
                rows = list(range(a, min(64, a + width)))
                inputs = [prompts[i] if step == 0 else [int(emitted[step - 1, i])] for i in rows]
                past = [0 if step == 0 else 512 + step - 1] * len(rows)
                widths = list(map(len, inputs))
                positions = sum([list(range(p, p + w)) for p, w in zip(past, widths)], [])
                arrays = [np.ascontiguousarray(v, dtype=np.int64 if i >= 4 else np.int32)
                          for i, v in enumerate([rows, past, [p + w for p, w in zip(past, widths)],
                                                 [0] + np.cumsum(widths).tolist(), sum(inputs, []), positions])]
                out, idsout = buffer[:len(rows)], emitted[step, a:a + len(rows)]
                tick = time.perf_counter_ns()
                code = lib.modern_native_engine_forward(ctx, sum(widths), len(rows),
                          *[ctypes.c_void_p(v.ctypes.data) for v in arrays],
                          ctypes.c_void_p(out.ctypes.data), ctypes.c_void_p(idsout.ctypes.data))
                ended = time.perf_counter_ns()
                completion[step, rows] = ended / 1e9 - started
                check(code)
                entry['CAPI_windows'].append({'step': step, 'chunk': a, 'rows': len(rows), 'seconds': (ended - tick) / 1e9})
                vt = time.perf_counter()
                values = (out.astype(np.uint32) << 16).view(np.float32)
                finite = bool(np.isfinite(values).all())
                argmax_ok = values.argmax(-1).tolist() == idsout.tolist()
                del values
                validation_seconds += time.perf_counter() - vt
                exact = idsout.tolist() == gold[rows, step].tolist()
                if not (finite and argmax_ok and exact):
                    entry.update(failure={'step': step, 'rows': rows, 'finite': finite, 'CPUargmax_equal': argmax_ok,
                                          'actual_IDs': idsout.tolist(), 'reference_IDs': gold[rows, step].tolist()},
                                 generated_ids=emitted[:step + 1].T.tolist())
                    report.update(outcome='NEGATIVE_QUALITY_STOP', quality_passed=False,
                                  negative_stop_no_rebaseline=True, after_QA_skipped='first_quality_failure')
                    memory(f'trial{trial}-M{micro}-quality_stop')
                    save()
                    policy(0)
                    return
            step_times.append(time.perf_counter() - started)
            if step == 25:
                entry['gate26_passed'] = True
                entry['generated_ids'] = emitted[:26].T.tolist()
                entry['gate26_elapsed_seconds'] = step_times[-1]
                save()  # Continue this same cache/trajectory; no repeat prefill or rebaseline.
        entry['driver_loop_wall_seconds'] = time.perf_counter() - started
        after = counters()
        calls = 127 * (64 // micro)
        entry['projection_counter_delta'] = [b - a for a, b in zip(before['projection'], after['projection'])]
        entry['C4_counter_delta'] = [[after['C4'][d][k] - before['C4'][d][k] for k in range(2)] for d in range(8)]
        assert entry['projection_counter_delta'] == [43 * calls] * 8
        assert entry['C4_counter_delta'] == [[2 * csa * calls] * 2 for _ in range(8)]
        assert np.array_equal(emitted.T, gold)
        pre = sum(v['seconds'] for v in entry['CAPI_windows'] if v['step'] == 0)
        dec = sum(v['seconds'] for v in entry['CAPI_windows'] if v['step'] > 0)
        driver_dec = step_times[-1] - step_times[0]
        entry.update(complete=True, full_IDs_EOS_exact=True, generated_ids=emitted.T.tolist(),
                     CPUargmax_every_window=True, CPU_validation_seconds_outside_CAPI=validation_seconds,
                     native_prefill_seconds=pre, native_decode_seconds=dec,
                     native_prefill_input_tok_s=32768 / pre, native_decode_tok_s=8128 / dec,
                     native_decode_tok_s_per_card=8128 / dec / 8,
                     amortized_native_decode_tok_s_per_logical_row=127 / dec,
                     native_E2E_output_tok_s=8192 / (pre + dec), driver_decode_seconds=driver_dec,
                     driver_decode_tok_s=8128 / driver_dec, driver_decode_tok_s_per_card=8128 / driver_dec / 8,
                     amortized_driver_decode_tok_s_per_logical_row=127 / driver_dec,
                     driver_E2E_output_tok_s=8192 / entry['driver_loop_wall_seconds'],
                     driver_prefill_wall_seconds=step_times[0], per_step_completion_seconds=step_times)
        gaps = np.diff(completion, axis=0)
        assert np.isfinite(completion).all() and np.all(gaps > 0)
        entry['request_completion_offsets_seconds'] = completion.T.tolist()
        entry['request_latency_scope'] = ('All64 benchmark requests arrive at trajectory start after reset/memory observation; '
            'timestamps use existing chunk CAPI return on driver clock, before CPU finite/argmax. '
            'TTFT includes each row waiting for earlier prefill chunks; first-to-last includes later prefill waits. '
            'Amortized per-row rates above are benchmark divisions, not individual request rates. '
            'ITL percentiles use linear interpolation; no new GPU synchronization or per-token IO.')
        entry['request_metrics'] = [dict(row=i, non_EOS_output_tokens=128,
            first_to_last_seconds=float(completion[-1,i]-completion[0,i]),
            decode_tok_s=float(127/(completion[-1,i]-completion[0,i])),
            TTFT_seconds=float(completion[0,i]),
            ITL_p95_seconds=float(np.percentile(gaps[:,i],95)), max_gap_seconds=float(np.max(gaps[:,i])))
            for i in range(64)]
        entry['request_latency_summary'] = dict(TTFT_p95_seconds=float(np.percentile(completion[0],95)),
            TTFT_max_seconds=float(np.max(completion[0])), pooled_ITL_p95_seconds=float(np.percentile(gaps,95)),
            max_gap_seconds=float(np.max(gaps)), minimum_request_decode_tok_s=min(v['decode_tok_s'] for v in entry['request_metrics']))
        memory(f'trial{trial}-M{micro}-after')
        if trial == 1:
            first, second = report['config_trials_r8644']
            report['firstpair_CAPI_decode_speedup'] = second['native_decode_seconds'] / first['native_decode_seconds']
            report['firstpair_driver_decode_speedup'] = second['driver_decode_seconds'] / first['driver_decode_seconds']
            if min(report['firstpair_CAPI_decode_speedup'], report['firstpair_driver_decode_speedup']) <= 1.01:
                report['outcome'] = 'NO_MATERIAL_FIRSTPAIR_GAIN_STOP_NO_REPEAT'
                break
    policy(0)
    g['run_chat']('after_restore')
    assert report['finite_after_restore_passed'], 'after_QA_failed'
    report['quality_passed'] = True
    if len(report['config_trials_r8644']) == 4:
        report['outcome'] = 'ONNO_COMPLETE_PENDING_INDEPENDENT_AUDIT'
        report['performance_qualified'] = True
    save()
