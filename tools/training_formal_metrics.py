"""Recompute formal paired training acceptance from pinned scalar evidence."""
import math
import statistics

METRICS = {
    'T01': {'TextVQA': 'higher', 'COCO CIDEr': 'higher'},
    'T06': {'CLIP Score': 'higher', 'FID': 'lower'},
    'T07': {'CLIP Score': 'higher', 'FID': 'lower'},
    'T08': {'R@1': 'higher', 'MotionFD': 'lower'},
    'T10': {'Chamfer-L1': 'lower', 'F-score@1%': 'higher'},
    'T11': {'LPIPS': 'lower', 'albedo MAE': 'lower', 'roughness MAE': 'lower', 'metallic MAE': 'lower'},
    'T12': {'AUC@5': 'higher', 'AUC@15': 'higher', 'AUC@30': 'higher'},
}
for model in ('T02', 'T03', 'T04', 'T05'):
    METRICS[model] = METRICS['T01']


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


WARMUP_UPDATES = 5
MEASURED_UPDATES = 20


def complete_window(window):
    # Ordinals are relative to this run, including its warmup updates.
    ordinals = window["update_ordinals"]
    expected = list(range(WARMUP_UPDATES + 1, WARMUP_UPDATES + MEASURED_UPDATES + 1))
    need(isinstance(ordinals, list) and all(type(step) is int for step in ordinals)
         and ordinals == expected and type(window["measured_updates"]) is int
         and window["measured_updates"] == MEASURED_UPDATES,
         "timed window does not contain the complete ordered update sequence")
    seconds = window["seconds"]
    need(nonnegative(seconds) and seconds > 0, "invalid timed window duration")
    return seconds


def sage_wall_seconds(run, cards):
    # The original controller fences both devices around one host clock.
    need(cards == 2 and 'ranks' not in run, 'SAGE shared clock requires two devices and one window')
    clock = run['shared_device_clock']
    need(set(clock) == {'seconds', 'measured_updates', 'actor_device', 'vae_device',
                        'synchronized_before', 'synchronized_after', 'update_ordinals'},
         'invalid SAGE shared clock fields')
    devices = [clock['actor_device'], clock['vae_device']]
    need(all(type(device) is int for device in devices) and set(devices) == {0, 1},
         'SAGE clock must identify distinct actor and VAE devices')
    need(clock['synchronized_before'] == devices and clock['synchronized_after'] == devices
         and all(type(device) is int for key in ('synchronized_before', 'synchronized_after')
                 for device in clock[key]), 'SAGE clock must fence both devices at each boundary')
    return complete_window(clock)


def timing(runs, cards, recipe, boundary, verify_raw, *, provenance, shared_device_clock=False):
    need(len(runs) == 3 and len({r['run_id'] for r in runs}) == 3
         and all(isinstance(r['run_id'], str) and r['run_id'] for r in runs),
         'formal timing requires three distinct runs per platform')
    rates = []
    for run in runs:
        need(all(run.get(key) == value for key, value in provenance.items())
             and run['recipe_SHA256'] == recipe and run['boundary_SHA256'] == boundary
             and type(run['warmup_updates']) is int and run['warmup_updates'] == WARMUP_UPDATES
             and type(run['measured_updates']) is int and run['measured_updates'] == MEASURED_UPDATES
             and run['profiler'] is False and run['completed'] is True
             and sha(run['raw_receipt']['SHA256']), 'unqualified timing window')
        need(verify_raw(run['raw_receipt']) == {k: v for k, v in run.items() if k != 'raw_receipt'},
             'timing fields differ from the pinned receipt')
        if shared_device_clock:
            seconds = sage_wall_seconds(run, cards)
        else:
            need('shared_device_clock' not in run, 'expected a complete window for each rank')
            ranks = run['ranks']
            need(all(type(row['rank']) is int for row in ranks)
                 and sorted(row['rank'] for row in ranks) == list(range(cards)),
                 'incomplete or invalid rank inventory')
            seconds = max(complete_window(row) for row in ranks)
        for field in ('admission_seconds', 'load_seconds', 'cache_preparation_seconds'):
            need(nonnegative(run[field]), 'invalid separately reported startup timing')
        rates.append(MEASURED_UPDATES / seconds)
    return statistics.median(rates)


def verify_sample_inventory(row, metric, provenance, checkpoints, verify_raw):
    """Check a frozen ID plan and a manifest of private per-sample artifacts.

    These attachments are SHA-pinned JSON. Artifact locations and digests
    identify private outputs or failure records for independent scoring audits.
    """
    # Each metric may use a different holdout, such as TextVQA versus COCO.
    identity = dict(provenance, metric=metric)
    planned = verify_raw(row['sample_plan'])
    need(all(planned.get(key) == value for key, value in identity.items()),
         'sample plan belongs to different held-out evidence')
    sample_ids = planned['sample_ids']
    need(isinstance(sample_ids, list) and len(sample_ids) == row['planned_samples']
         and all(isinstance(sample_id, str) and sample_id.strip() for sample_id in sample_ids),
         'invalid planned held-out sample IDs')
    expected = set(sample_ids)
    need(len(expected) == len(sample_ids), 'duplicate planned held-out sample ID')

    inventory = verify_raw(row['sample_inventory'])
    need(all(inventory.get(key) == value for key, value in identity.items())
         and inventory['sample_plan_SHA256'] == row['sample_plan']['SHA256']
         and inventory['checkpoint_SHA256'] == checkpoints,
         'sample inventory belongs to a different plan, scorer or checkpoint')
    need(inventory['arms'].keys() == checkpoints.keys(), 'incorrect held-out arms')
    for platform, samples in inventory['arms'].items():
        observed = set()
        failed = 0
        for sample in samples:
            sample_id = sample['sample_id']
            need(isinstance(sample_id, str) and sample_id in expected and sample_id not in observed,
                 'unexpected or duplicate held-out sample ID')
            artifact = sample['artifact']
            need(isinstance(artifact['location'], str) and artifact['location'].strip()
                 and sha(artifact['SHA256']) and sample['status'] in ('scored', 'failed'),
                 'missing held-out output or failure evidence')
            observed.add(sample_id)
            failed += sample['status'] == 'failed'
        need(observed == expected and failed == row[platform + '_failed']
             and len(samples) - failed == row[platform + '_scored'],
             'held-out sample inventory or counts differ from the declared plan')


def effect(rows, expected, provenance, verify_raw):
    need(rows.keys() == expected.keys(), 'held-out metric set differs')
    regressions = {}
    common_checkpoints = None
    for metric, direction in expected.items():
        row = rows[metric]
        reference, target = row['A100'], row['Hygon']
        need(all(nonnegative(value) for value in (reference, target, row['Hygon_before']))
             and sha(row['raw_scores']['SHA256']) and sha(row['paired_checkpoint_binding']['SHA256']),
             'invalid held-out scores or missing evidence')
        scores = {k: v for k, v in row.items() if k not in ('raw_scores', 'paired_checkpoint_binding')}
        binding = verify_raw(row['paired_checkpoint_binding'])
        need(all(binding.get(key) == value for key, value in provenance.items()),
             'paired checkpoints belong to different held-out evidence')
        platforms = ('A100', 'Hygon', 'Hygon_before')
        need(all(sha(binding[key]) for key in platforms), 'missing paired checkpoint identities')
        checkpoints = {key: binding[key] for key in platforms}
        need(common_checkpoints is None or checkpoints == common_checkpoints,
             'held-out metrics use different paired checkpoints')
        common_checkpoints = checkpoints
        receipt = dict(scores, metric=metric, **provenance,
                       checkpoint_SHA256=checkpoints)
        need(verify_raw(row['raw_scores']) == receipt,
             'held-out scores differ from the pinned case, scorer, dataset or checkpoints')
        need(type(row['planned_samples']) is int and row['planned_samples'] > 0,
             'missing held-out sample denominator')
        for platform in ('A100', 'Hygon', 'Hygon_before'):
            scored, failed = row[platform + '_scored'], row[platform + '_failed']
            need(type(scored) is int and type(failed) is int and scored > 0 and failed >= 0
                 and scored + failed == row['planned_samples'],
                 'held-out score and failure counts do not close')
        verify_sample_inventory(row, metric, provenance, checkpoints, verify_raw)
        delta = reference - target if direction == 'higher' else target - reference
        regressions[metric] = {'absolute_degradation': delta,
                               'relative_degradation': None if reference == 0 else delta / reference}
    return regressions


def stage_three(case, framework_commit, evidence, verify_raw):
    need(evidence['case'] == case['case'] and evidence['framework_commit'] == framework_commit,
         'formal evidence belongs to a different case or framework')
    cards = evidence['cards']
    need(type(cards) is int and cards > 0 and cards == case['original_cards'],
         'formal timing does not use the registered recipe card count')
    recipe, boundary = evidence['recipe_SHA256'], evidence['boundary_SHA256']
    need(sha(recipe) and sha(boundary) and evidence['A100_original_launch_SHA256'] == case['launch_SHA256'],
         'missing recipe, timing boundary or A100 launch identity')
    runs = evidence['timing']
    need(len({run['raw_receipt']['SHA256'] for arm in ('A100', 'Hygon') for run in runs[arm]}) == 6,
         'formal timing receipts are not distinct')
    shared_device_clock = case['case'] == 'T09.sage-grpo'
    reference_commit = evidence.get('A100_framework_commit')
    need(isinstance(reference_commit, str) and len(reference_commit) == 40
         and all(c in '0123456789abcdef' for c in reference_commit),
         'missing or invalid A100 framework revision')
    a100 = timing(runs['A100'], cards, recipe, boundary, verify_raw,
                  provenance={'case': case['case'], 'platform': 'A100',
                              'framework_commit': reference_commit,
                              'original_launch_SHA256': case['launch_SHA256']},
                  shared_device_clock=shared_device_clock)
    hygon = timing(runs['Hygon'], cards, recipe, boundary, verify_raw,
                   provenance={'case': case['case'], 'platform': 'Hygon',
                               'framework_commit': framework_commit},
                   shared_device_clock=shared_device_clock)
    need(not ({r['run_id'] for r in evidence['timing']['A100']}
              & {r['run_id'] for r in evidence['timing']['Hygon']}), 'run IDs shared across platforms')
    holdout = evidence['holdout']
    need(sha(holdout['dataset_SHA256']) and sha(holdout['scorer_SHA256']),
         'missing held-out dataset or scorer identity')
    model = case['case'].split('.')[0]
    expected = METRICS.get(model)
    if model == 'T09':
        vbench = verify_raw(holdout['registered_vbench_binding'])
        need(vbench['case'] == case['case'] and vbench['original_launch_SHA256'] == case['launch_SHA256'],
             'VBench binding does not match the original recipe')
        directions = vbench['metric_directions']
        need(isinstance(directions, dict) and directions
             and all(name.startswith('VBench/') and len(name) > 7 and direction in ('higher', 'lower')
                     for name, direction in directions.items()), 'missing registered VBench metrics')
        expected = {'Video CLIP': 'higher', **directions}
    need(expected is not None, 'unknown training case')
    provenance = {'case': case['case'], 'framework_commit': framework_commit,
                  'dataset_SHA256': holdout['dataset_SHA256'], 'scorer_SHA256': holdout['scorer_SHA256']}
    regressions = effect(holdout['metrics'], expected, provenance, verify_raw)
    ratio = hygon / a100
    return {'A100_median_updates_per_second': a100, 'Hygon_median_updates_per_second': hygon,
            'throughput_ratio': ratio, 'effect_regressions': regressions,
            'passed': ratio >= .90 and all(row['relative_degradation'] is not None and row['relative_degradation'] <= .05
                                         for row in regressions.values())}
