"""Recompute engineering rates from the pinned audits of original A100 windows."""
import math
from statistics import median

from training_formal_metrics import need, nonnegative, sha


GENERATION_CASES = {
    'T06.sft': 1, 'T06.ddpo': 4, 'T07.sft': 1, 'T07.flow-grpo': 4,
    'T08.flow-matching-lora': 1, 'T08.motion-rft': 4, 'T10.shape-flow-lora': 1,
    'T11.material-head': 1,
}
VLM_CASES = {f'T{group:02d}.sft' for group in range(1, 6)}
VGGT_CASE = 'T12.camera-head'


def match(value, expected, context):
    need(nonnegative(value) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.0), context)


def register(content, seen, context):
    need(sha(content) and content not in seen, context)
    seen.add(content)


def repeats(rows):
    need([row['repeat'] for row in rows] == [1, 2, 3], 'expected exactly three ordered timing repeats')
    return rows


def check_generation(cases, audit):
    need(set(audit['cases']) == GENERATION_CASES.keys(), 'generation timing case inventory differs')
    need(audit['original_launches_read'] == audit['exact_original_result_receipts_reconstructed']
         == audit['original_step_logs_read'] == 24, 'generation timing audit is missing original records')
    result, seen = {}, set()
    for name, multiplier in GENERATION_CASES.items():
        case, record = cases[name], audit['cases'][name]
        units = 'updates_per_second' if multiplier == 1 else 'rollout_samples_per_second'
        need(case['original_cards'] == record['a100_recipe_cards'] == 1
             and record['original_rate_units'] == units, f'{name}: original allocation or units differ')
        rates, native, boundaries = [], [], set()
        for row in repeats(record['repeats']):
            need(row['launch_SHA256'] == case['launch_SHA256'], f'{name}: different original launch')
            need(row['global_updates_logged'] == 25 and row['measured_updates'] == 20
                 and row['measured_reported_work_units'] == 20 * multiplier and row['original_rate_units'] == units,
                 f'{name}: original work denominator differs')
            for key in ('original_result_SHA256', 'raw_step_log_SHA256'):
                register(row[key], seen, f'{name}: reused or invalid original timing record')
            duration = row['complete_window_seconds']
            need(nonnegative(duration) and duration > 0,
                 f'{name}: missing complete original timing window')
            boundary = row['reported_timing_boundary']
            need(boundary is None or isinstance(boundary, str) and boundary.strip(),
                 f'{name}: invalid timing boundary description')
            boundaries.add(boundary)
            rate = 20 / duration
            match(row['original_rate'], multiplier * rate, f'{name}: original unit rate differs')
            match(row['updates_per_second'], rate, f'{name}: normalized update rate differs')
            rates.append(rate)
            native.append(multiplier * rate)
        need(len(boundaries) == 1, f'{name}: original timing boundary changes across repeats')
        result[name] = median(rates)
        match(record['median_updates_per_second'], result[name], f'{name}: normalized median differs')
        match(record['original_rate_median'], median(native), f'{name}: original unit median differs')
    return result


def check_vlm(cases, audit):
    need(set(audit['cases']) == VLM_CASES, 'VLM timing case inventory differs')
    need(audit['original_launches_read'] == audit['original_final_trainer_states_read'] == 15
         and audit['exact_original_rank_receipts_reconstructed'] == 120, 'VLM timing audit is missing original ranks')
    result, seen = {}, set()
    for name, record in audit['cases'].items():
        case = cases[name]
        need(case['original_cards'] == record['a100_cards'] == 8 and record['effective_batch'] == 32
             and record['warmup_updates'] == 5 and record['measured_updates'] == 20,
             f'{name}: VLM workload differs')
        rows = repeats(record['repeats'])
        launches = {row['launch_SHA256'] for row in rows}
        need(len(launches) == 3 and case['launch_SHA256'] in launches, f'{name}: pinned VLM launch missing')
        rates = []
        for row in rows:
            durations = row['rank_window_seconds']
            pins = row['rank_timing_SHA256']
            need(len(durations) == len(pins) == 8 and all(nonnegative(t) and t > 0 for t in durations),
                 f'{name}: incomplete rank window')
            need(sha(row['final_trainer_state_SHA256']), f'{name}: invalid final trainer state identity')
            for content in (*pins, row['launch_SHA256']):
                register(content, seen, f'{name}: reused or invalid VLM timing record')
            slowest = max(durations)
            rank = row['slowest_rank']
            need(type(rank) is int and 0 <= rank < 8 and durations[rank] == slowest,
                 f'{name}: slowest rank differs')
            rate = 20 / slowest
            match(row['updates_per_second'], rate, f'{name}: slowest-rank update rate differs')
            match(row['samples_per_second'], 32 * rate, f'{name}: sample rate differs')
            rates.append(rate)
        result[name] = median(rates)
        match(record['median_updates_per_second'], result[name], f'{name}: VLM update median differs')
        match(record['median_samples_per_second'], 32 * result[name], f'{name}: VLM sample median differs')
    return result


def check_vggt(case, audit):
    need(audit['case'] == case['case'] == VGGT_CASE and case['original_cards'] == 1,
         'VGGT timing case differs')
    runs = audit['runs']
    failure = audit['archived_pretraining_failure']
    need(len(runs) == len({row['run'] for row in runs}) == 3
         and failure['run'] not in {row['run'] for row in runs} and failure['phase'] == 'before training',
         'VGGT successful repeats and pretraining failure differ')
    need(audit['original_result_jsons_read'] == 3
         and sha(audit['original_result_tar_SHA256']) and sha(audit['original_result_manifest_SHA256'])
         and sha(audit['original_backend']['SHA256']), 'VGGT original results or backend are not bound')
    rates, seen = [], set()
    for row in runs:
        need(row['config_SHA256'] == case['launch_SHA256']
             and row['measured_update_count'] == 20 and row['measured_frames'] == 40,
             'VGGT timing workload differs')
        need(row['original_result_SHA256'] == row['summary_result_SHA256'],
             'VGGT original result SHA differs from the archived summary')
        for key in ('step_log_SHA256', 'summary_result_SHA256'):
            register(row[key], seen, 'reused or invalid VGGT timing record')
        duration = row['full_measurement_seconds']
        steps = row['recorded_step_sum_seconds']
        need(nonnegative(duration) and nonnegative(steps) and 0 < steps < duration,
             'VGGT timing must use the complete wall window')
        rate = 20 / duration
        match(row['updates_per_second'], rate, 'VGGT update rate differs')
        match(row['frames_per_second'], 2 * rate, 'VGGT frame rate differs')
        rates.append(rate)
    result = median(rates)
    match(audit['a100_median_updates_per_second'], result, 'VGGT update median differs')
    match(audit['a100_median_frames_per_second'], 2 * result, 'VGGT frame median differs')
    return {VGGT_CASE: result}
