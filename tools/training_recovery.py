"""Recompute the formal recovery gate from the registered recipe and paired runs."""
import math

from training_formal_metrics import need, nonnegative, sha


ENGINEERING_UPDATES = 25
STATE_FIELDS = (
    'parameters_SHA256', 'optimizer_SHA256', 'scheduler_SHA256', 'rng_SHA256',
    'data_cursor_SHA256',
)


def recovery_plan(case, verify_raw):
    plan = case.get('recovery_plan')
    need(isinstance(plan, dict) and sha(case['launch_SHA256'])
         and plan['original_launch_SHA256'] == case['launch_SHA256'],
         'recovery plan differs from the registered original recipe')
    checkpoints, resumes = plan['checkpoint_steps'], plan['resume_steps']
    for steps, last in ((checkpoints, ENGINEERING_UPDATES), (resumes, ENGINEERING_UPDATES - 1)):
        need(isinstance(steps, list) and steps
             and all(type(step) is int and 1 <= step <= last for step in steps)
             and steps == sorted(set(steps)), 'invalid registered checkpoint or resume steps')
    need(set(resumes) <= set(checkpoints), 'resume step is missing from the checkpoint plan')
    # SAGE has one training process for the actor and a separate VAE device.
    ranks = 1 if case['case'] == 'T09.sage-grpo' else case['original_cards']
    need(type(plan['training_ranks']) is int and plan['training_ranks'] == ranks,
         'recovery plan uses the wrong training rank count')
    limits = plan['comparison_limits']
    need(set(limits) == {'loss_absolute', 'parameter_relative_l2'}
         and all(nonnegative(limit) for limit in limits.values()), 'invalid recovery comparison limits')
    references = plan['original_references']
    need(isinstance(references, list) and references
         and len({p['SHA256'] for p in references}) == len(references),
         'recovery plan requires distinct original reference evidence')
    for ref in references:
        verify_raw(ref)
    return plan


def rank_values(rows, ranks):
    need(isinstance(rows, list) and len(rows) == ranks
         and all(type(row['rank']) is int for row in rows)
         and [row['rank'] for row in rows] == list(range(ranks)), 'incomplete recovery rank inventory')
    return rows


def state(rows, ranks):
    for row in rank_values(rows, ranks):
        need(set(row) == {'rank', 'accumulation_boundary', *STATE_FIELDS}
             and all(sha(row[name]) for name in STATE_FIELDS)
             and type(row['accumulation_boundary']) is int and row['accumulation_boundary'] >= 0,
             'incomplete checkpoint parameters, optimizer, scheduler, RNG, data cursor or accumulation boundary')
    return rows


def execution(receipt, binding, first, ranks):
    need(all(receipt.get(key) == value for key, value in binding.items())
         and receipt['terminal_status'] == 'COMPLETED'
         and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and sha(receipt['terminal_receipt_SHA256']) and sha(receipt['run_id'])
         and type(receipt['start_global_step']) is int and receipt['start_global_step'] == first
         and type(receipt['final_global_step']) is int and receipt['final_global_step'] == ENGINEERING_UPDATES,
         'recovery run differs in case, source, global steps or terminal status')
    processes = {row['process_identity_SHA256'] for row in rank_values(receipt['rank_processes'], ranks)}
    need(len(processes) == ranks and all(sha(pid) for pid in processes),
         'missing or reused rank process identity')
    updates = receipt['updates']
    need(isinstance(updates, list) and all(type(row['global_step']) is int for row in updates)
         and [row['global_step'] for row in updates] == list(range(first + 1, ENGINEERING_UPDATES + 1)),
         'incomplete or unordered recovery updates')
    for update in updates:
        for row in rank_values(update['ranks'], ranks):
            loss = row['loss']
            need(sha(row['input_SHA256']) and type(loss) in (float, int) and math.isfinite(loss),
                 'missing or nonfinite rank input or loss')
    final = state(receipt['final_state'], ranks)
    return processes, updates, final


def verify_recovery(case, framework_commit, evidence, verify_raw):
    plan = recovery_plan(case, verify_raw)
    need('continuous' in evidence and isinstance(evidence.get('resumed'), list),
         'recovery evidence requires continuous and resumed receipts')
    ranks = plan['training_ranks']
    binding = {'case': case['case'], 'framework_commit': framework_commit,
               'original_launch_SHA256': case['launch_SHA256'], 'platform': 'Hygon'}
    continuous_ref = evidence['continuous']
    continuous = verify_raw(continuous_ref)
    processes, baseline, original_final = execution(continuous, binding, 0, ranks)
    runs = {continuous['run_id']}
    terminals = {continuous['terminal_receipt_SHA256']}
    checkpoints = continuous['checkpoints']
    need(isinstance(checkpoints, list) and all(type(row['global_step']) is int for row in checkpoints)
         and [row['global_step'] for row in checkpoints] == plan['checkpoint_steps'],
         'incomplete continuous checkpoint inventory')
    sources = {}
    for item in checkpoints:
        need(sha(item['manifest_SHA256']), 'missing original checkpoint manifest')
        sources[item['global_step']] = item['manifest_SHA256'], state(item['state'], ranks)
    if ENGINEERING_UPDATES in sources:
        need(sources[ENGINEERING_UPDATES][1] == original_final, 'final state differs from the last checkpoint')
    need(len({digest for digest, _ in sources.values()}) == len(sources),
         'a checkpoint manifest was reused for multiple global steps')

    resumed = [verify_raw(item) for item in evidence['resumed']]
    need(all(type(doc['start_global_step']) is int for doc in resumed)
         and [doc['start_global_step'] for doc in resumed] == plan['resume_steps'],
         'missing or extra fresh-process recovery at a registered checkpoint')
    limits = plan['comparison_limits']
    comparisons = []
    for receipt_ref, receipt in zip(evidence['resumed'], resumed, strict=True):
        step = receipt['start_global_step']
        fresh, suffix, final = execution(receipt, binding, step, ranks)
        need(not (fresh & processes) and receipt['run_id'] not in runs
             and receipt['terminal_receipt_SHA256'] not in terminals,
             'recovery did not use a distinct process run')
        processes.update(fresh)
        runs.add(receipt['run_id'])
        terminals.add(receipt['terminal_receipt_SHA256'])
        manifest, original_state = sources[step]
        need(receipt['source_checkpoint_SHA256'] == manifest
             and state(receipt['restored_state'], ranks) == original_state,
             'restored checkpoint state differs from the continuous run')

        absolute_loss = 0.0
        for left, right in zip(baseline[step:], suffix, strict=True):
            for a, b in zip(left['ranks'], right['ranks'], strict=True):
                need(a['input_SHA256'] == b['input_SHA256'], 'resumed update used different input')
                absolute_loss = max(absolute_loss, abs(a['loss'] - b['loss']))
        need(absolute_loss <= limits['loss_absolute'], 'recovered loss exceeds the original bound')

        continuous_params = [row['parameters_SHA256'] for row in original_final]
        resumed_params = [row['parameters_SHA256'] for row in final]
        relative_l2 = 0.0
        if continuous_params != resumed_params:
            difference = verify_raw(evidence['parameter_comparisons'][str(step)])
            expected = dict(binding, global_step=ENGINEERING_UPDATES,
                            continuous_run_SHA256=continuous_ref['SHA256'],
                            resumed_run_SHA256=receipt_ref['SHA256'],
                            continuous_parameters_SHA256=continuous_params,
                            resumed_parameters_SHA256=resumed_params)
            need(all(difference.get(key) == value for key, value in expected.items())
                 and nonnegative(difference['maximum_rank_relative_l2']),
                 'parameter comparison differs from the paired final state')
            relative_l2 = difference['maximum_rank_relative_l2']
            # A different logical parameter digest cannot establish exact recovery.
            need(limits['parameter_relative_l2'] > 0 and relative_l2 <= limits['parameter_relative_l2'],
                 'recovered parameters exceed the original bound')
        comparisons.append(dict(resume_step=step, reproduced_updates=len(suffix), training_ranks=ranks,
                                max_absolute_loss_difference=absolute_loss, max_parameter_relative_l2=relative_l2,
                                resumed_receipt_SHA256=receipt_ref['SHA256']))
    return comparisons
