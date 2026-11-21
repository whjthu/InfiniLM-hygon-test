"""Derive training timing targets from verified A100 windows and bind a real Qwen comparator."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from training_formal_metrics import MEASURED_UPDATES, WARMUP_UPDATES
from verify_training_acceptance import verify


QWEN = 'T01.sft'
HYGON = 'results/bw1000/2026-11-20/Qwen-native-paired-Swift-original417-terminal-and-checkpoints.json'
HYGON_SHA = '7418d3f843f18ef93dfa9bf5d7c35366b486b1180173d5c1c1dad104c5ff5421'
SOURCE = 'code/bw1000/training-qwen-native-paired-layer-Swift-AB-v3/'
SOURCE_SHA = 'd684c15927c09d1913f048a0c793106e2c975079e318ac118a3d97bdebb051ed'
CAPTURED = 'before update 6 through completed checkpoint 25; includes checkpoint 10/20/25 I/O'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def attachment(root, path, expected=None):
    raw = (root / path).read_bytes()
    digest = sha256(raw).hexdigest()
    require(expected is None or digest == expected, f'original evidence changed: {path}')
    return json.loads(raw), dict(path=path, SHA256=digest, bytes=len(raw))


def timing_budget(median, ratio):
    require(math.isfinite(median) and median > 0 and math.isfinite(ratio) and 0 < ratio <= 1,
            'invalid reference throughput or stage threshold')
    return MEASURED_UPDATES / (median * ratio)


def qwen_hygon_scope(root, case, a100_audit, median, limit):
    target, target_pin = attachment(root, HYGON, HYGON_SHA)
    index, source_index_pin = attachment(root, SOURCE + 'SOURCE_FILES.json', SOURCE_SHA)
    sources = index['files']
    assets, assets_pin = attachment(root, SOURCE + 'ASSETS.json', sources['ASSETS.json']['SHA256'])
    claim, claim_pin = attachment(root, SOURCE + 'CLAIM.json', sources['CLAIM.json']['SHA256'])
    require(case['case'] == QWEN and claim['case'] == QWEN and
            assets['staged_source']['original-launch.json'] == case['launch_SHA256'] and
            case['original_cards'] == claim['resources']['DCU'] == a100_audit['a100_cards'] == 8 and
            a100_audit['effective_batch'] == 32 and
            (a100_audit['warmup_updates'], a100_audit['measured_updates']) == (WARMUP_UPDATES, MEASURED_UPDATES)
            and target['timed_scope'] == CAPTURED, 'Qwen engineering timing scope differs')
    pre, gpu, post = (target['accounting'][job] for job in ('1103606.416', '1103606.417', '1103606.419'))
    require(all(row[2:4] == ['COMPLETED', '0:0'] and row[5] == claim['resources']['node']
                for row in (pre, gpu, post)) and 'gres/dcu=8' in gpu[6].split(',')
            and (pre[1], gpu[1], post[1]) == (claim['label'] + '-pre', claim['label'], claim['label'] + '-post')
            and target['independent_post_clean'] is True and target['measured_pairs'] == 1, 'original Qwen steps differ')
    a100_windows = [max(run['rank_window_seconds']) for run in a100_audit['repeats']]
    durations = {}
    for name in ('stock', 'paired'):
        arm = target['by_arm'][name]
        ranks = arm['rank_seconds']
        require(len(ranks) == 8 and all(math.isfinite(seconds) and 0 < seconds <= int(gpu[4]) + 1 for seconds in ranks),
                'Qwen engineering window has missing or invalid ranks')
        wall = max(ranks)
        rate = MEASURED_UPDATES / wall
        require(wall == arm['slowest_rank_seconds'] and rate == arm['updates_per_second']
                and [ckpt['global_step'] for ckpt in arm['checkpoints']] == [10, 20, 25],
                'Qwen engineering clock or checkpoint coverage differs')
        durations[name] = dict(complete_window_seconds=wall, updates_per_second=rate,
                               relative_to_A100_engineering_median=rate / median,
                               seconds_above_stage3_budget=wall - limit,
                               required_wall_reduction_fraction=max(0., 1. - limit / wall))
    improvement = durations['stock']['complete_window_seconds'] - durations['paired']['complete_window_seconds']
    return dict(qualification='ENGINEERING_CROSS_PLATFORM_CONTEXT_ONLY', target_steps=[pre[0], gpu[0], post[0]],
                measured_Hygon_windows_per_arm=1, A100_slowest_rank_windows_seconds=a100_windows,
                target=durations, paired_saved_seconds=improvement,
                paired_fraction_of_stock_stage3_gap=improvement / durations['stock']['seconds_above_stage3_budget']
                if durations['stock']['seconds_above_stage3_budget'] > 0 else None,
                evidence=[target_pin, source_index_pin, assets_pin, claim_pin],
                numerical_or_effect_accepted=False, formal_comparison=False)


def report(root):
    audited = verify(root)
    rates = audited['verified_A100_engineering_medians_updates_per_second']
    status, _ = attachment(root, 'status/bw1000/latest.json')
    index = status['unified_training_acceptance_matrix']
    matrix, matrix_pin = attachment(root, index['matrix'], index['matrix_SHA256'])
    cases = {c['case']: c for c in matrix['cases']}
    require(set(rates) == set(cases) and len(cases) == audited['cases'] == 17, 'missing A100 timing targets')
    ratio = matrix['stage3']['minimum_A100_throughput_ratio']
    budgets = {name: dict(A100_engineering_median_updates_per_second=rates[name], original_cards=case['original_cards'],
                         stage3_maximum_complete_20_update_seconds=timing_budget(rates[name], ratio))
               for name, case in cases.items()}
    vlm_pin = matrix['original_A100_engineering_timing_audits']['vlm']
    vlm, vlm_evidence = attachment(root, vlm_pin['path'], vlm_pin['SHA256'])
    require(vlm['timing_boundary'] == CAPTURED, 'A100 timing boundary differs')
    qwen = qwen_hygon_scope(root, cases[QWEN], vlm['cases'][QWEN], rates[QWEN],
                            budgets[QWEN]['stage3_maximum_complete_20_update_seconds'])
    return dict(reference_scope='Verified original A100 engineering timing, no formal promotion',
                matrix=matrix_pin, vlm_reference=vlm_evidence, threshold=ratio, budgets=budgets, qwen=qwen,
                audited_original_evidence_files=audited['evidence_files'], formal_stage3_passed=audited['stage3_passed'],
                new_Slurm_submissions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.repo.resolve()), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
