"""Recompute the formal recovery gate from the registered recipe and paired runs."""
import math
from pathlib import PurePosixPath

from read_training_slurm_membership import allocated_dcu, completed_numbered_step
from training_formal_metrics import need, nonnegative, sha


ENGINEERING_UPDATES = 25
STATE_FIELDS = (
    'parameters_SHA256', 'optimizer_SHA256', 'scheduler_SHA256', 'rng_SHA256',
    'data_cursor_SHA256',
)
ZIMAGE_FINAL_CHECKPOINT_FIELDS = {
    'buffers', 'integrity_sha256', 'lr_scheduler', 'metadata', 'modes', 'optimizer',
    'optimizer_implementation', 'optimizer_parameter_names', 'optimizer_placements',
    'optimizer_progress', 'rng', 'scheduler', 'trainable',
}


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


def verify_registered_zimage_recovery(report, source, source_root, registry, verify_bytes):
    """Bind the recorded HIP recovery to its registered source, jobs and checkpoint."""
    install = report['target_source_installation']
    proof = source['source_proof']
    framework = proof['candidate_commit']
    need(report['case'] == source['case'] == 'T07.sft'
         and report['candidate'] == install['candidate_commit'] == framework
         and install['request'] == source['request']
         and install['publication'] == report['published_scope']
         and install['registered_case_count'] == len(registry) == 17
         and proof['registry'] == registry
         and install['source_files'] == proof['source_members']
         and install['target_source_rehash_verified'] is True
         and install['fresh_resume_without_checkpoint_refused'] is True,
         'Z-Image recovery source or registered case identity differs')
    for path, pin in proof['changed_members'].items():
        raw = verify_bytes(dict(path=f'{source_root}/{path}', SHA256=pin['SHA256']))
        need(len(raw) == pin['bytes'], 'Z-Image recovery source member length differs')
    entry_sha = proof['changed_members']['tools/hygon_training_recipe.py']['SHA256']

    execution = report['actual_execution_and_cleanup']
    files = execution['files']
    columns = ['JobIDRaw', 'JobName', 'State', 'ExitCode', 'ElapsedRaw', 'NodeList', 'AllocTRES']

    def fields(name):
        member = files[name]
        need(sha(member['SHA256']) and member['bytes'] > 0, 'unbound Z-Image recovery record')
        return member['fields']

    def terminal(step, rows, expected_dcu):
        matches = [row for row in rows if row[0] == step]
        need(len(matches) == 1, 'missing or duplicate Z-Image recovery step')
        row = completed_numbered_step(columns, matches[0], step, scope='Z-Image engineering recovery')
        need(allocated_dcu(row['AllocTRES']) == expected_dcu, 'wrong DCU scope in Z-Image recovery')
        return row

    runs = {}
    steps = []
    labels = []
    for mode, expected in (('continuous', list(range(1, 26))), ('resume20', list(range(21, 26)))):
        invoked = fields(mode + '-registered-invocation.json')
        result = fields(mode + '/result.json')
        owner = fields(mode + '-worker-result.json')
        progress = fields(mode + '/progress.json')
        step = owner['slurm']
        row = terminal(step, execution['accounting'], 1)
        need(invoked['registered_case'] == report['case'] and invoked['candidate'] == framework
             and invoked['entry_SHA256'] == entry_sha
             and invoked['binding_SHA256'] == source['retained_real_binding']['SHA256']
             and invoked['mode'] == owner['phase'] == result['mode'] == mode
             and invoked['planned_global_steps'] == owner['completed_global_steps']
             == result['completed_global_steps'] == progress['completed_global_steps'] == expected
             and result['actual_optimizer_updates'] == owner['actual_optimizer_updates']
             == report['actual_updates'][mode] == len(expected)
             and result['passed'] is owner['passed'] is result['frozen_parameters_unchanged'] is True,
             'Z-Image recovery optimizer steps, registered input binding or training result differs')
        steps.append(step)
        labels.append(row['NodeList'])
        runs[mode] = result['checkpoints']
    continuous, resume = runs['continuous'], runs['resume20']
    need([item['global_step'] for item in continuous] == [10, 20, 25]
         and [item['global_step'] for item in resume] == [25],
         'Z-Image recovery checkpoint schedule differs')
    restore = fields('resume20/restore.json')
    after20 = continuous[1]
    need(restore['restored_boundary_exact'] is True
         and restore['checkpoint_SHA256'] == after20['SHA256'] == fields('resume20-worker-result.json')['checkpoint20_SHA256']
         and restore['next']['global_step'] == 21
         and restore['next']['sample_cursor'] == after20['next_data_cursor']
         and restore['rng_devices'] == after20['RNG_devices'] == [0]
         and all(sha(continuous[-1][key]) and continuous[-1][key] == resume[-1][key]
                 for key in ('SHA256', 'adapter_SHA256', 'integrity_SHA256')),
         'Z-Image recovery restored checkpoint20 or final checkpoint25 differs')

    audit = report['independent_numeric_and_state_comparison']
    comparison = audit['comparison']
    post = fields('post.json')
    cleaned = terminal(post['step'], execution['accounting'], 0)
    audit_step, = report['independent_audit_accounting']['accounting']
    terminal(audit_step[0], [audit_step], 0)
    need(len(set([*steps, post['step'], audit_step[0]])) == 4
         and labels == [post['node'], post['node']] and cleaned['NodeList'] == post['node']
         and audit['accounting'][0][0] == audit_step[0]
         and audit['producer_outcome_SHA256'] == files['outcome.json']['SHA256']
         and fields('outcome.json')['post_clean'] is True
         and comparison['actual_slurm_steps'] == steps
         and comparison['same_pci'] is comparison['paired_input_fingerprints_equal'] is True
         and comparison['paired_steps'] == list(range(21, 26))
         and comparison['paired_loss_abs'] == [0] * 5
         and comparison['max_adapter_abs'] == comparison['adapter_tensors_with_differences'] == 0
         and set(comparison['exact_checkpoint25_contents']) == ZIMAGE_FINAL_CHECKPOINT_FIELDS
         and all(value is True for value in comparison['exact_checkpoint25_contents'].values())
         and comparison['A100_comparison_executed'] is False,
         'Z-Image independent same-platform recovery audit differs')
    return dict(framework_commit=framework, registered_cases=len(registry),
                actual_steps=steps, audit_step=audit_step[0], cleanup_step=post['step'],
                actual_updates=report['actual_updates'], final_checkpoint_SHA256=resume[-1]['SHA256'],
                cross_platform_numerics=False, formal_performance=False)


def state(rows, ranks):
    for row in rank_values(rows, ranks):
        need(set(row) == {'rank', 'accumulation_boundary', *STATE_FIELDS}
             and all(sha(row[name]) for name in STATE_FIELDS)
             and type(row['accumulation_boundary']) is int and row['accumulation_boundary'] >= 0,
             'incomplete checkpoint parameters, optimizer, scheduler, RNG, data cursor or accumulation boundary')
    return rows


def execution(receipt, binding, first, ranks, cards, verify_raw):
    need(all(receipt.get(key) == value for key, value in binding.items())
         and receipt['terminal_status'] == 'COMPLETED'
         and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and sha(receipt['terminal_receipt_SHA256']) and sha(receipt['run_id'])
         and type(receipt['start_global_step']) is int and receipt['start_global_step'] == first
         and type(receipt['final_global_step']) is int and receipt['final_global_step'] == ENGINEERING_UPDATES,
         'recovery run differs in case, source, global steps or terminal status')
    terminal_ref = receipt.get('terminal_evidence')
    need(isinstance(terminal_ref, dict) and terminal_ref.get('SHA256') == receipt['terminal_receipt_SHA256'],
         'recovery run is missing independently pinned terminal evidence')
    terminal = verify_raw(terminal_ref)
    terminal_binding = dict(binding, run_id=receipt['run_id'], start_global_step=first,
                            final_global_step=ENGINEERING_UPDATES)
    need(all(terminal.get(key) == value for key, value in terminal_binding.items()),
         'recovery terminal evidence belongs to a different execution')
    row = completed_numbered_step(terminal.get('accounting_columns'), terminal.get('accounting_row'),
                                  receipt.get('slurm_step'), scope='recovery')
    dcu = allocated_dcu(row['AllocTRES']) if isinstance(row['AllocTRES'], str) else None
    need(dcu == cards, 'recovery Slurm step differs from the registered DCU allocation')
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


def checkpoint_state(item, binding, run_id, ranks, verify_raw):
    reference = item.get('manifest')
    need(sha(item['manifest_SHA256']) and isinstance(reference, dict),
         'missing independently pinned checkpoint inventory')
    manifest = verify_raw(reference)
    expected = dict(binding, run_id=run_id, global_step=item['global_step'],
                    original_checkpoint_SHA256=item['manifest_SHA256'])
    need(all(manifest.get(key) == value for key, value in expected.items())
         and type(manifest['global_step']) is int,
         'checkpoint inventory belongs to a different execution or global step')
    files = manifest['files']
    need(isinstance(files, list) and bool(files), 'checkpoint inventory contains no persisted files')
    paths = set()
    for file in files:
        value = file['relative_path']
        need(isinstance(value, str) and value and '\\' not in value and ':' not in value,
             'invalid checkpoint inventory path')
        path = PurePosixPath(value)
        need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == value and value != '.'
             and value not in paths and type(file['bytes']) is int and file['bytes'] > 0
             and sha(file['SHA256']), 'missing or duplicated checkpoint file identity')
        paths.add(value)
    actual = state(manifest['state'], ranks)
    need(actual == state(item['state'], ranks), 'checkpoint inventory disagrees with recorded rank state')
    return actual


def verify_recovery(case, framework_commit, evidence, verify_raw):
    plan = recovery_plan(case, verify_raw)
    need('continuous' in evidence and isinstance(evidence.get('resumed'), list),
         'recovery evidence requires continuous and resumed receipts')
    ranks = plan['training_ranks']
    cards = case['original_cards']
    binding = {'case': case['case'], 'framework_commit': framework_commit,
               'original_launch_SHA256': case['launch_SHA256'], 'platform': 'Hygon'}
    continuous_ref = evidence['continuous']
    continuous = verify_raw(continuous_ref)
    processes, baseline, original_final = execution(continuous, binding, 0, ranks, cards, verify_raw)
    runs = {continuous['run_id']}
    terminals = {continuous['terminal_receipt_SHA256']}
    checkpoints = continuous['checkpoints']
    need(isinstance(checkpoints, list) and all(type(row['global_step']) is int for row in checkpoints)
         and [row['global_step'] for row in checkpoints] == plan['checkpoint_steps'],
         'incomplete continuous checkpoint inventory')
    sources = {}
    for item in checkpoints:
        actual = checkpoint_state(item, binding, continuous['run_id'], ranks, verify_raw)
        sources[item['global_step']] = item['manifest_SHA256'], actual
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
        fresh, suffix, final = execution(receipt, binding, step, ranks, cards, verify_raw)
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

        discrete_fields = ('scheduler_SHA256', 'data_cursor_SHA256', 'accumulation_boundary')
        need(all(a[field] == b[field] for a, b in zip(original_final, final, strict=True) for field in discrete_fields),
             'recovery final scheduler, data cursor or accumulation boundary differs')
        need(all(a['rng_SHA256'] == b['rng_SHA256'] for a, b in zip(original_final, final, strict=True)),
             'recovery final per-rank RNG differs')
        optimizer_exact = all(a['optimizer_SHA256'] == b['optimizer_SHA256']
                              for a, b in zip(original_final, final, strict=True))
        need(limits['parameter_relative_l2'] > 0 or optimizer_exact,
             'recovery final optimizer differs under the registered exact-state comparison')

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
                                final_optimizer_exact=optimizer_exact, final_rank_RNG_exact=True,
                                resumed_receipt_SHA256=receipt_ref['SHA256']))
    return comparisons
