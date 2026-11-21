"""Recompute formal paired training acceptance from pinned scalar evidence."""
import math
from pathlib import PurePosixPath
import re
import statistics

from textvqa_effect import SCORING as TEXTVQA_SCORING, VALID_CREDITS as TEXTVQA_CREDITS
from coco_cider_effect import (
    METRIC as CIDER_METRIC, POLICY as CIDER_POLICY, SCORE_TOLERANCE as CIDER_TOLERANCE,
    ORIGINAL_TRAINING_SHA256 as CIDER_TRAINING_SHA256, WHEEL_SHA256 as CIDER_WHEEL_SHA256,
    validate_holdout_image_ids, verify_image_bytes,
)
from vggt_co3d_effect import (
    CASE as CO3D_CASE, POLICY as CO3D_POLICY, THRESHOLDS as CO3D_THRESHOLDS,
    UPSTREAM as CO3D_POSE_EVALUATOR, credit as co3d_credit, scorer_sha256 as co3d_scorer_sha256,
)

METRICS = {
    'T01': {'TextVQA': 'higher', 'COCO CIDEr': 'higher'},
    'T06': {'CLIP Score': 'higher', 'FID': 'lower'},
    'T07': {'CLIP Score': 'higher', 'FID': 'lower'},
    'T08': {'R@1': 'higher', 'MotionFD': 'lower'},
    'T10': {'Chamfer-L1': 'lower', 'F-score@1%': 'higher'},
    'T11': {'LPIPS': 'lower', 'albedo MAE': 'lower', 'roughness MAE': 'lower', 'metallic MAE': 'lower'},
    'T12': {'AUC@5': 'higher', 'AUC@15': 'higher', 'AUC@30': 'higher'},
}
VLM_MODELS = frozenset({'T01', 'T02', 'T03', 'T04', 'T05'})
for model in sorted(VLM_MODELS - {'T01'}):
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
DP2_TIMING_BOUNDARY = (
    'before first measured update through completed final two-rank checkpoint commit and barrier; '
    'includes replicated-state checks and checkpoint IO; excludes loading and warmup'
)


def verify_hunyuan_a100_recorded_updates(case, references, sage_audit=None):
    """Verify the archived engineering windows against their case and original artifact pins."""
    case_id = case['case']
    sage = case_id == 'T09.sage-grpo'
    need(case_id in ('T09.sft', 'T09.sage-grpo') and case['original_cards'] == (2 if sage else 1),
         'unexpected Hunyuan A100 recorded case or card count')
    source = references['cases'][case_id]
    config = source['launch']['config']
    need(config['warmup_steps'] == WARMUP_UPDATES and config['max_steps'] == MEASURED_UPDATES
         and config['checkpoint_steps'] == [10, 20] and config['transformer_version'] == '480p_t2v',
         'Hunyuan original launch uses a different training window')
    repeats = source['runs']
    need([run['repeat'] for run in repeats] == [1, 2, 3], 'Hunyuan A100 needs three original repeats')
    prefix = 'hunyuan-cuda-rl-baseline-r1' if sage else 'hunyuan-cuda-sft-baseline-r2-walltime'
    if sage:
        need(sage_audit is not None and sage_audit['case'] == case_id and sage_audit['rc'] == 0
             and [row['repeat'] for row in sage_audit['audit']['runs']] == [1, 2, 3],
             'missing original SAGE step audit')
    rates, results_seen, step_logs_seen = [], set(), set()
    for run in repeats:
        root = f"node/results/{prefix}-repeat{run['repeat']}"
        launch, result = run['launch'], run['result']
        for pin, name in ((launch, 'launch.json'), (result, 'result.json')):
            need(pin['path'] == f'{root}/{name}' and sha(pin['sha256'])
                 and type(pin['bytes']) is int and pin['bytes'] > 0,
                 'Hunyuan recorded window has an unbound repeat artifact')
        need(launch['sha256'] == case['launch_SHA256'] and result['sha256'] not in results_seen,
             'Hunyuan recorded window uses a different launch or a duplicate result')
        results_seen.add(result['sha256'])
        summary = run['summary']
        metrics = summary['metrics']
        seconds = metrics['measurement_wall_seconds']
        need(nonnegative(seconds) and seconds > 0 and metrics['steps'] == MEASURED_UPDATES
             and metrics['measurement_steps'] == MEASURED_UPDATES and metrics['warmup_steps'] == WARMUP_UPDATES,
             'Hunyuan recorded duration or measured update count is invalid')
        rate = MEASURED_UPDATES / seconds
        need(rate == metrics['measurement_updates_per_second'], 'Hunyuan rate differs from the complete window')
        if sage:
            pins = sage_audit['pins']
            steps = pins[f'{root}/steps.ndjson']
            need(pins.get(launch['path']) == launch and pins.get(result['path']) == result
                 and steps['path'] == f'{root}/steps.ndjson' and sha(steps['sha256'])
                 and steps['sha256'] not in step_logs_seen and type(steps['bytes']) is int and steps['bytes'] > 0,
                 'SAGE original step log is unbound or reused')
            step_logs_seen.add(steps['sha256'])
            row = sage_audit['audit']['runs'][run['repeat'] - 1]
            need(metrics['num_generations'] == metrics['sampling_steps'] == row['rewards_per_component_per_update'] == 4
                 and row['reward_components'] == ['MQ', 'Overall', 'TA', 'VQ']
                 and row['warmup_updates'] == WARMUP_UPDATES and row['measured_updates'] == MEASURED_UPDATES
                 and row['recorded_adapter_changes'] == WARMUP_UPDATES + MEASURED_UPDATES
                 and row['measurement_wall_seconds'] == seconds and row['measurement_updates_per_second'] == rate
                 and nonnegative(row['sum_measured_step_seconds']) and 0 < row['sum_measured_step_seconds'] <= seconds,
                 'SAGE complete wall clock differs from the independently audited policy updates')
        else:
            need(all(metrics.get(key) == value for key, value in dict(first_global_step=1,
                     completed_global_step_at_start=0, executed_warmup_steps=WARMUP_UPDATES,
                     executed_measured_steps=MEASURED_UPDATES, executed_global_steps=WARMUP_UPDATES+MEASURED_UPDATES).items()),
                 'SFT recorded window is not a full training run from step zero')
            load, training = (summary['stage_timings'][field] for field in ('model_load_seconds', 'training_seconds'))
            need(nonnegative(load) and load > 0 and nonnegative(training) and seconds <= training
                 and nonnegative(summary['invocation_seconds']) and load + training <= summary['invocation_seconds'],
                 'SFT startup and training windows are inconsistent')
        rates.append(rate)
    median = statistics.median(rates)
    need(median == source['A100_median_measurement_updates_per_second'] == case['a100_median_updates_per_second']
         and (not sage or median == sage_audit['audit']['median_updates_per_second']),
         'Hunyuan recorded A100 median differs from its three windows')
    return median


def verify_hunyuan_dp2_a100_windows(case, audit, verify_raw):
    """Recompute the engineering reference from the independently audited original windows."""
    need(case['case'] == audit['case'] == 'T09.dp2' and case['original_cards'] == 2,
         'DP2 timing evidence belongs to a different case')
    report = audit['audit']
    need(report['passed'] is True and audit['formal_A100_performance_accepted'] is False,
         'incorrect state for the original DP2 engineering reference')
    original = verify_raw(audit['reference'])['cases'][case['case']]
    config = original['launch']['config']
    need(all(config.get(k) == v for k, v in dict(world_size=2, sp=1, warmup_steps=WARMUP_UPDATES,
                                               max_steps=MEASURED_UPDATES,
                                               checkpoint_steps=[10, 20, 25]).items()),
         'DP2 original launch uses a different geometry or training window')
    repeats = original['runs']
    need([r['repeat'] for r in repeats] == [1, 2, 3], 'DP2 original needs three reference repeats')
    baselines = {name: run for name, run in report['runs'].items() if run['start'] == 0}
    need(len(baselines) == len(repeats), 'DP2 reference includes missing or extra baseline windows')
    artifacts_seen = set()
    windows_seen = set()
    rates = []
    for original_run in repeats:
        launch = original_run['launch']
        result = original_run['result']
        launch_path = PurePosixPath(launch['path'])
        run_name = launch_path.parent.name
        need(run_name in baselines and run_name not in windows_seen, 'DP2 reference run identity differs')
        windows_seen.add(run_name)
        run = baselines[run_name]
        artifacts = run['original_artifacts']
        need(run['run'] == run_name and run['configured_resume'] is None
             and launch['sha256'] == case['launch_SHA256']
             and artifacts['launch.json'] == launch and artifacts['result.json'] == result,
             'DP2 audit and original result artifacts differ')
        rows = run['windows']
        need([row['rank'] for row in rows] == [0, 1]
             and all(type(row['rank']) is int and row['steps'] == MEASURED_UPDATES
                     and row['boundary'] == DP2_TIMING_BOUNDARY
                     and nonnegative(row['seconds']) and row['seconds'] > 0 for row in rows),
             'DP2 timing must use both complete rank windows at the original boundary')
        final_ckpt = run['checkpoints']['25']
        need([rank['rank'] for rank in final_ckpt['ranks']] == [0, 1]
             and final_ckpt['marker_SHA256'] == artifacts['checkpoint-25/COMMITTED.json']['sha256'],
             'DP2 timed window lacks the final pair checkpoint commit')
        # Deterministic step traces may have identical bytes across repeats.
        for name in ('result.json', 'rank-0/summary.json', 'rank-0/steps.ndjson',
                     'rank-1/summary.json', 'rank-1/steps.ndjson', 'checkpoint-25/COMMITTED.json'):
            pin = artifacts[name]
            need(sha(pin['sha256']) and pin['path'] == (launch_path.parent / name).as_posix()
                 and type(pin['bytes']) is int and pin['bytes'] > 0, 'DP2 timed artifact is unbound or reused')
            if name in ('result.json', 'rank-0/summary.json', 'rank-1/summary.json'):
                need(pin['sha256'] not in artifacts_seen, 'DP2 duplicated a timed summary across repeats')
                artifacts_seen.add(pin['sha256'])
        seconds = max(row['seconds'] for row in rows)
        metrics = original_run['summary']['metrics']
        rate = MEASURED_UPDATES / seconds
        need(metrics['measurement_steps'] == MEASURED_UPDATES
             and metrics['executed_warmup_steps'] == WARMUP_UPDATES
             and metrics['performance_accepted'] is False and run['performance_accepted'] is False
             and metrics['measurement_wall_seconds'] == seconds
             and metrics['measurement_updates_per_second'] == run['measurement_updates_per_second'] == rate,
             'DP2 reported duration or throughput differs from the slowest rank')
        rates.append(rate)
    median = statistics.median(rates)
    need(median == original['A100_median_measurement_updates_per_second']
         == report['baseline_median_updates_per_second'] == case['a100_median_updates_per_second'],
         'DP2 matrix median differs from the three independently audited windows')
    return median


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


def verified_execution(run, cards, provenance, window_seconds, verify_raw, *, shared_device_clock=False):
    """Link the timing receipt hash to a separately pinned execution summary.

    Summaries use the original Slurm step identity or a launcher process identity
    that includes process start time. Device ranks use node-local PCI addresses.
    """
    attachment = run.get('execution_evidence')
    need(isinstance(attachment, dict) and sha(attachment.get('SHA256')), 'missing pinned execution evidence')
    record = verify_raw(attachment)
    need(all(record.get(key) == value for key, value in provenance.items())
         and record['raw_timing_SHA256'] == run['raw_receipt']['SHA256'],
         'execution evidence and original timing receipt differ')
    progress = record.get('training_progress')
    need(isinstance(progress, dict) and 'configured_resume' in progress and progress['configured_resume'] is None
         and type(progress.get('start_global_step')) is int and progress['start_global_step'] == 0,
         'formal timing needs execution evidence of a fresh run from global step zero')
    # SAGE has one training actor and a separate VAE device. The VAE has no optimizer steps.
    training_ranks = 1 if shared_device_clock else cards
    rank_steps = progress.get('rank_steps')
    need(isinstance(rank_steps, list) and all(isinstance(row, dict) and type(row.get('rank')) is int for row in rank_steps)
         and sorted(row['rank'] for row in rank_steps) == list(range(training_ranks)),
         'missing or invalid execution progress for training ranks')
    expected_steps = list(range(1, WARMUP_UPDATES + MEASURED_UPDATES + 1))
    for row in rank_steps:
        steps = row.get('global_update_ordinals')
        need(isinstance(steps, list) and all(type(step) is int for step in steps) and steps == expected_steps,
             'execution evidence does not cover global optimizer updates 1 through 25')
    terminal = record['terminal']
    method, handle = terminal['method'], terminal['handle']
    need(method in ('slurm-step', 'launcher-process') and isinstance(handle, str)
         and bool(handle) and handle == handle.strip()
         and terminal['state'] == 'COMPLETED' and terminal['exit_code'] == '0:0'
         and nonnegative(terminal['elapsed_seconds']) and terminal['elapsed_seconds'] > 0,
         'formal timing requires a successful terminal execution')
    need(provenance['platform'] != 'Hygon' or method == 'slurm-step', 'Hygon timing needs its original Slurm step')
    if method == 'slurm-step':
        need(re.fullmatch(r'[0-9]+\.[0-9]+', handle) is not None, 'expected a numbered Slurm step')
    # Accounting elapsed may be integer seconds while measured rank durations are fractional.
    need(window_seconds <= terminal['elapsed_seconds'] + (1 if method == 'slurm-step' else 0),
         'measured timing exceeds the actual execution window')
    devices = record['devices']
    need(all(type(device['rank']) is int for device in devices)
         and sorted(device['rank'] for device in devices) == list(range(cards)),
         'physical device inventory differs from the measured rank count')
    addresses = set()
    node_ranks = {}
    models = []
    for device in sorted(devices, key=lambda device: device['rank']):
        node, pci, model = device['node'], device['pci'], device['model']
        need(isinstance(node, str) and bool(node.strip().rstrip('.')) and isinstance(model, str)
             and bool(model.strip()) and isinstance(pci, str)
             and re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', pci),
             'invalid device model, node or PCI identity')
        normalized_node = node.strip().rstrip('.').lower()
        domain, address = pci.lower().split(':', 1)
        addresses.add((normalized_node, int(domain, 16), address))
        node_ranks.setdefault(normalized_node, []).append(device['rank'])
        models.append(' '.join(model.casefold().split()))
    need(len(addresses) == cards, 'multiple ranks were attributed to the same physical device')
    if provenance['platform'] == 'A100':
        need(all(re.search(r'\ba100\b', model) for model in models), 'A100 timing used a different accelerator model')
    runtime = record['runtime']
    fields = ['driver', 'framework', 'operator_library']
    if cards > 1:
        fields.append('interconnect')
    need(all(isinstance(runtime.get(key), str) and runtime[key].strip() for key in fields),
         'missing hardware and runtime configuration evidence')
    hardware_scope = dict(device_models_by_rank=models, rank_groups_by_node=sorted(node_ranks.values()),
                          runtime={key: runtime[key].strip() for key in fields})
    return handle, hardware_scope


def timing(runs, cards, recipe, boundary, verify_raw, *, provenance, shared_device_clock=False, return_scope=False):
    need(len(runs) == 3 and len({r['run_id'] for r in runs}) == 3
         and all(isinstance(r['run_id'], str) and r['run_id'] for r in runs),
         'formal timing requires three distinct runs per platform')
    rates = []
    executions = set()
    reference_scope = None
    for run in runs:
        need(all(run.get(key) == value for key, value in provenance.items())
             and run['recipe_SHA256'] == recipe and run['boundary_SHA256'] == boundary
             and type(run['warmup_updates']) is int and run['warmup_updates'] == WARMUP_UPDATES
             and type(run['measured_updates']) is int and run['measured_updates'] == MEASURED_UPDATES
             and run['profiler'] is False and run['completed'] is True
             and sha(run['raw_receipt']['SHA256']), 'unqualified timing window')
        need(verify_raw(run['raw_receipt']) == {k: v for k, v in run.items() if k not in ('raw_receipt', 'execution_evidence')},
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
        identity, scope = verified_execution(
            run, cards, provenance, seconds, verify_raw, shared_device_clock=shared_device_clock,
        )
        need(identity not in executions, 'three timing runs reuse one actual execution')
        executions.add(identity)
        if reference_scope is None:
            reference_scope = scope
        else:
            need(scope == reference_scope, 'formal timing repeats differ in hardware, rank placement or runtime')
        for field in ('admission_seconds', 'load_seconds', 'cache_preparation_seconds'):
            need(nonnegative(run[field]), 'invalid separately reported startup timing')
        rates.append(MEASURED_UPDATES / seconds)
    median = statistics.median(rates)
    return (median, reference_scope) if return_scope else median


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
    textvqa = metric == 'TextVQA'
    cider = metric == CIDER_METRIC
    co3d_limit = next((limit for limit in CO3D_THRESHOLDS if metric == f'AUC@{limit}'), None)
    if co3d_limit is not None:
        files = inventory.get('prediction_file_SHA256')
        category_auc = inventory.get('category_auc')
        need(provenance.get('case') == CO3D_CASE and provenance.get('scorer_SHA256') == co3d_scorer_sha256()
             and inventory.get('policy') == CO3D_POLICY
             and inventory.get('pose_evaluator_commit') == CO3D_POSE_EVALUATOR
             and sha(planned.get('evaluation_plan_SHA256')) and sha(inventory.get('prediction_manifest_SHA256'))
             and isinstance(files, dict) and files.keys() == checkpoints.keys() and all(map(sha, files.values()))
             and isinstance(category_auc, dict) and category_auc.keys() == checkpoints.keys(),
             'missing or incompatible VGGT CO3D evaluation inputs')
        co3d_categories, unordered_pairs = {}, set()
        for sample_id in sample_ids:
            pair = re.fullmatch(r'([a-z0-9_]+)/([A-Za-z0-9_]+)/(0|[1-9][0-9]*)/(0|[1-9][0-9]*)', sample_id)
            need(pair is not None and pair[3] != pair[4], 'invalid VGGT CO3D planned pair')
            pair_key = (pair[1], pair[2], *sorted((pair[3], pair[4])))
            need(pair_key not in unordered_pairs, 'duplicate VGGT CO3D planned frame pair')
            unordered_pairs.add(pair_key)
            co3d_categories[sample_id] = pair[1]
    if textvqa or cider:
        files = inventory.get('prediction_file_SHA256', {})
        need(isinstance(files, dict) and files.keys() == checkpoints.keys() and all(map(sha, files.values())),
             'missing pinned prediction files')
    if textvqa:
        need(inventory.get('textvqa_scoring') == TEXTVQA_SCORING, 'missing official TextVQA scoring evidence')
    if cider:
        validate_holdout_image_ids(sample_ids)
        scorer = inventory.get('scorer_binding', {})
        need(inventory.get('cider_scoring') == CIDER_POLICY and sha(provenance.get('scorer_SHA256'))
             and scorer.get('SHA256') == provenance['scorer_SHA256'], 'missing pinned COCO CIDEr scoring evidence')
        runtime = verify_raw(scorer)
        need(runtime.get('wheel_SHA256') == CIDER_WHEEL_SHA256 and runtime.get('policy') == CIDER_POLICY
             and all(sha(runtime.get(field)) for field in ('implementation_SHA256', 'java_SHA256'))
             and all(isinstance(runtime.get(field), str) and runtime[field].strip()
                     for field in ('java_version', 'numpy_version')), 'invalid COCO CIDEr scorer runtime')
        media_ref = inventory.get('image_inventory')
        need(isinstance(media_ref, dict) and sha(media_ref.get('SHA256')), 'missing COCO image byte inventory')
        media = verify_raw(media_ref)
        need(all(media.get(key) == value for key, value in identity.items())
             and media.get('original_training_SHA256') == CIDER_TRAINING_SHA256
             and media.get('checkpoint_SHA256') == checkpoints
             and media.get('sample_plan_SHA256') == row['sample_plan']['SHA256'],
             'COCO image byte inventory belongs to a different plan or checkpoint')
        image_hashes = verify_image_bytes(media['images'], sample_ids)
    for platform, samples in inventory['arms'].items():
        observed = set()
        failed = 0
        credit_sum = 0
        cider_scores = []
        co3d_pairs_by_category = {}
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
            if textvqa:
                credit = sample.get('textvqa_credit_x30')
                need(type(credit) is int and credit in TEXTVQA_CREDITS
                     and (sample['status'] == 'scored' or credit == 0), 'invalid per-question TextVQA credit')
                credit_sum += credit
            if cider:
                need(sample.get('image_SHA256') == image_hashes[sample_id],
                     'scored COCO image differs from the frozen media inventory')
                score = sample.get('cider_score')
                need(nonnegative(score) and score <= 10 + CIDER_TOLERANCE
                     and (sample['status'] == 'scored' or score == 0), 'invalid per-image CIDEr score')
                cider_scores.append(score)
            if co3d_limit is not None:
                category = co3d_categories[sample_id]
                error = sample.get('maximum_error_deg')
                need(sample.get('category') == category and 'maximum_error_deg' in sample
                     and ((sample['status'] == 'failed' and error is None)
                          or (sample['status'] == 'scored' and nonnegative(error) and error <= 180)),
                     'invalid VGGT CO3D pair, category or error')
                co3d_pairs_by_category.setdefault(category, []).append(co3d_credit(error, co3d_limit))
        need(observed == expected and failed == row[platform + '_failed']
             and len(samples) - failed == row[platform + '_scored'],
             'held-out sample inventory or counts differ from the declared plan')
        if textvqa:
            accuracy = credit_sum / (30 * len(sample_ids))
            need(0 <= row[platform] <= 1 and math.isclose(row[platform], accuracy, rel_tol=0, abs_tol=1e-12),
                 'TextVQA accuracy differs from the full question denominator')
        if cider:
            score = math.fsum(cider_scores) / len(sample_ids)
            need(math.isclose(row[platform], score, rel_tol=0, abs_tol=CIDER_TOLERANCE),
                 'COCO CIDEr differs from the full image denominator')
        if co3d_limit is not None:
            per_category = {category: sum(credits) / (len(credits) * co3d_limit)
                            for category, credits in co3d_pairs_by_category.items()}
            reported = category_auc[platform]
            need(isinstance(reported, dict) and reported.keys() == per_category.keys()
                 and all(nonnegative(reported[name]) and math.isclose(reported[name], result, rel_tol=0, abs_tol=1e-12)
                         for name, result in per_category.items())
                 and math.isclose(row[platform], math.fsum(per_category.values()) / len(per_category),
                                  rel_tol=0, abs_tol=1e-12),
                 'VGGT CO3D AUC differs from the full per-category pair denominator')
    if co3d_limit is not None:
        return (planned['evaluation_plan_SHA256'], inventory['prediction_manifest_SHA256'], files, inventory['arms'])


def effect(rows, expected, provenance, verify_raw, *, metric_bindings=None, trained_checkpoints=None):
    need(rows.keys() == expected.keys(), 'held-out metric set differs')
    if metric_bindings is not None:
        fields = {'dataset_SHA256', 'scorer_SHA256'}
        need(isinstance(metric_bindings, dict) and metric_bindings.keys() == expected.keys()
             and all(isinstance(binding, dict) and binding.keys() == fields and all(map(sha, binding.values()))
                     for binding in metric_bindings.values()), 'invalid per-metric held-out dataset or scorer bindings')
    regressions = {}
    common_checkpoints = None
    co3d_pair_binding = None
    for metric, direction in expected.items():
        row = rows[metric]
        identity = provenance if metric_bindings is None else {**provenance, **metric_bindings[metric]}
        reference, target = row['A100'], row['Hygon']
        need(all(nonnegative(value) for value in (reference, target, row['Hygon_before']))
             and sha(row['raw_scores']['SHA256']) and sha(row['paired_checkpoint_binding']['SHA256']),
             'invalid held-out scores or missing evidence')
        scores = {k: v for k, v in row.items() if k not in ('raw_scores', 'paired_checkpoint_binding')}
        binding = verify_raw(row['paired_checkpoint_binding'])
        need(all(binding.get(key) == value for key, value in identity.items()),
             'paired checkpoints belong to different held-out evidence')
        platforms = ('A100', 'Hygon', 'Hygon_before')
        need(all(sha(binding[key]) for key in platforms), 'missing paired checkpoint identities')
        checkpoints = {key: binding[key] for key in platforms}
        need(common_checkpoints is None or checkpoints == common_checkpoints,
             'held-out metrics use different paired checkpoints')
        need(trained_checkpoints is None or checkpoints == trained_checkpoints,
             'held-out effect uses different paired checkpoints from the verified training runs')
        common_checkpoints = checkpoints
        receipt = dict(scores, metric=metric, **identity,
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
        pair_binding = verify_sample_inventory(row, metric, identity, checkpoints, verify_raw)
        if pair_binding is not None:
            need(co3d_pair_binding is None or pair_binding == co3d_pair_binding,
                 'VGGT CO3D AUC thresholds use different plans, predictions or pair errors')
            co3d_pair_binding = pair_binding
        delta = reference - target if direction == 'higher' else target - reference
        regressions[metric] = {'absolute_degradation': delta,
                               'relative_degradation': None if reference == 0 else delta / reference}
    return regressions


def effect_training_checkpoints(holdout, case, framework_commit, a100_commit, verify_raw):
    """Bind the scored checkpoints to a common schedule and actual completed training."""
    reference = holdout.get('training_protocol')
    need(isinstance(reference, dict) and sha(reference.get('SHA256')), 'missing paired effect training protocol')
    protocol = verify_raw(reference)
    identity = dict(case=case['case'], framework_commit=framework_commit, A100_framework_commit=a100_commit,
                    A100_original_launch_SHA256=case['launch_SHA256'])
    need(all(protocol.get(key) == value for key, value in identity.items()), 'effect training protocol identity differs')
    fields = ('training_recipe_SHA256', 'schedule_SHA256', 'input_order_SHA256')
    common = {key: protocol.get(key) for key in fields}
    need(all(map(sha, common.values())), 'missing frozen effect training recipe, schedule or input order')
    first, last = protocol.get('start_global_step'), protocol.get('final_global_step')
    need(type(first) is int and type(last) is int and 0 <= first < last, 'invalid common effect training updates')
    runs = protocol['runs']
    need(set(runs) == {'A100','Hygon'} and runs['A100']['SHA256'] != runs['Hygon']['SHA256'],
         'paired effect training needs two distinct original runs')
    checkpoints = {}
    for platform, commit in (('A100', a100_commit), ('Hygon', framework_commit)):
        attachment = runs[platform]
        need(sha(attachment.get('SHA256')), 'missing original effect training run evidence')
        run = verify_raw(attachment)
        binding = dict(common, case=case['case'], platform=platform, framework_commit=commit)
        if platform == 'A100':
            binding['original_launch_SHA256'] = case['launch_SHA256']
        need(all(run.get(key) == value for key,value in binding.items()), 'effect training run recipe or identity differs')
        progress = run['training_progress']
        expected = dict(start_global_step=first, final_global_step=last, executed_optimizer_updates=last-first)
        need(all(type(progress.get(key)) is int and progress[key] == value for key,value in expected.items()),
             'effect training run does not cover the frozen update window')
        terminal = run['terminal']
        method, handle = terminal['method'], terminal['handle']
        need(method in ('slurm-step','launcher-process') and isinstance(handle, str) and handle == handle.strip()
             and handle and terminal['state'] == 'COMPLETED' and terminal['exit_code'] == '0:0',
             'effect training lacks a successful terminal execution')
        need(platform != 'Hygon' or method == 'slurm-step', 'Hygon effect training needs an original Slurm step')
        if method == 'slurm-step':
            need(re.fullmatch(r'[0-9]+\.[0-9]+', handle) is not None, 'effect training needs a numbered Slurm step')
        checkpoint = run['final_checkpoint_SHA256']
        need(sha(checkpoint), 'effect training has no bound final checkpoint')
        checkpoints[platform] = checkpoint
        if platform == 'Hygon':
            start = run['start_checkpoint_SHA256']
            need(sha(start), 'Hygon before evaluation needs its original checkpoint evidence')
            checkpoints['Hygon_before'] = start
    return checkpoints


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
    a100, a100_scope = timing(runs['A100'], cards, recipe, boundary, verify_raw,
                  provenance={'case': case['case'], 'platform': 'A100',
                              'framework_commit': reference_commit,
                              'original_launch_SHA256': case['launch_SHA256']},
                  shared_device_clock=shared_device_clock, return_scope=True)
    hygon, hygon_scope = timing(runs['Hygon'], cards, recipe, boundary, verify_raw,
                   provenance={'case': case['case'], 'platform': 'Hygon',
                               'framework_commit': framework_commit},
                   shared_device_clock=shared_device_clock, return_scope=True)
    need(a100_scope['rank_groups_by_node'] == hygon_scope['rank_groups_by_node'],
         'formal paired timing uses different per-node rank placement')
    need(not ({r['run_id'] for r in evidence['timing']['A100']}
              & {r['run_id'] for r in evidence['timing']['Hygon']}), 'run IDs shared across platforms')
    holdout = evidence['holdout']
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
    provenance = {'case': case['case'], 'framework_commit': framework_commit}
    metric_bindings = holdout.get('metric_bindings')
    if metric_bindings is None:
        need(model not in VLM_MODELS, 'VLM effect requires independent TextVQA and COCO CIDEr bindings')
        shared = {field: holdout.get(field) for field in ('dataset_SHA256', 'scorer_SHA256')}
        need(all(map(sha, shared.values())), 'missing held-out dataset or scorer identity')
        provenance.update(shared)
    scored_checkpoints = effect_training_checkpoints(holdout, case, framework_commit, reference_commit, verify_raw)
    regressions = effect(holdout['metrics'], expected, provenance, verify_raw, metric_bindings=metric_bindings,
                         trained_checkpoints=scored_checkpoints)
    ratio = hygon / a100
    return {'A100_median_updates_per_second': a100, 'Hygon_median_updates_per_second': hygon,
            'throughput_ratio': ratio, 'effect_regressions': regressions,
            'timing_hardware': {'A100': a100_scope, 'Hygon': hygon_scope},
            'passed': ratio >= .90 and all(row['relative_degradation'] is not None and row['relative_degradation'] <= .05
                                         for row in regressions.values())}
