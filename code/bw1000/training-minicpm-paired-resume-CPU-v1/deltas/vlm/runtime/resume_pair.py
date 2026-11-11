"""Compare complete recorded native continuation and fresh-process resume tails."""
from __future__ import annotations


def _validate(record, start):
    plan = record['plan']
    if (record.get('native_resume_window_complete') is not True
            or plan['checkpoint_global_step'] != start or plan['unchanged_max_steps'] != 25
            or plan['gradient_accumulation_steps'] != 4
            or record['saved_checkpoints'] != ([20, 25] if start == 10 else [25])
            or len(record['RNG_loads']) != 1
            or record['RNG_loads'][0]['saved_equal'] is not True):
        raise ValueError('incomplete native recovery record')
    count = (25 - start) * 4
    inputs, outputs, steps = record['inputs'], record['outputs'], record['steps']
    if (len(inputs) != count or len(outputs) != count
            or [row['global_step'] for row in steps] != list(range(start + 1, 26))):
        raise ValueError('incomplete forward or optimizer denominator')
    for row in steps:
        if set(row['state']) != {'adapter', 'optimizer', 'scheduler'}:
            raise ValueError('incomplete recorded full training state')
        for item in row['state'].values():
            if (item.get('finite') is not True or len(item['SHA256']) != 64
                    or any(c not in '0123456789abcdef' for c in item['SHA256'])):
                raise ValueError('missing complete training-state identity')
    for ordinal, (before, after) in enumerate(zip(inputs, outputs), 1):
        if (before['ordinal'] != ordinal or after['ordinal'] != ordinal
                or before['global_step'] != start + (ordinal - 1) // 4):
            raise ValueError('input, output or accumulation occurrence differs')
        for value in (before['full_input'], after['loss']):
            digest = value['SHA256']
            if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                    or value.get('finite') is not True):
                raise ValueError('missing full-value input/loss identity')
    return inputs[-20:], outputs[-20:], steps[-5:]


def compare_resume_pair(continuous, restored):
    """Exact identities are diagnostics; no model-specific tolerance is invented.

    Complete final checkpoint RNG and state need their independent on-host file
    audit in addition to this recorded-tail comparison. Hence this helper alone
    never declares overall recovery qualification.
    """
    left, left_loss, left_steps = _validate(continuous, 10)
    right, right_loss, right_steps = _validate(restored, 20)
    for key in ('rank', 'world_size'):
        if continuous[key] != restored[key]:
            raise ValueError('paired rank or topology differs')
    for key in ('world_size_to_verify', 'per_device_batch', 'gradient_accumulation_steps',
                'dataloader_batches_per_rank_to_verify', 'effective_batch'):
        if continuous['plan'][key] != restored['plan'][key]:
            raise ValueError('paired data or accumulation recipe differs')
    if (continuous['world_size'] != continuous['plan']['world_size_to_verify']
            or type(continuous['rank']) is not int
            or not 0 <= continuous['rank'] < continuous['world_size']):
        raise ValueError('invalid paired rank identity')
    rows = []
    for ordinal, (a, b, loss_a, loss_b) in enumerate(zip(left, right, left_loss, right_loss), 1):
        rows.append(dict(tail_ordinal=ordinal, global_step=a['global_step'],
                         full_input_equal=a['full_input'] == b['full_input'],
                         loss_exact=loss_a['loss'] == loss_b['loss'],
                         runtime_cursor_equal=(a['runtime_cursor'] == b['runtime_cursor']
                             if 'runtime_cursor' in a and 'runtime_cursor' in b else None)))
    updates = [dict(global_step=a['global_step'], full_state_exact=a['state'] == b['state'])
               for a, b in zip(left_steps, right_steps)]
    return dict(rank=continuous['rank'], world_size=continuous['world_size'],
                planned_tail_forwards=20, compared_tail_forwards=len(rows),
                input_equal_forwards=sum(row['full_input_equal'] for row in rows),
                exact_loss_forwards=sum(row['loss_exact'] for row in rows),
                exact_observed_runtime_cursors=sum(row['runtime_cursor_equal'] is True for row in rows),
                compared_optimizer_updates=5,
                exact_full_state_updates=sum(row['full_state_exact'] for row in updates),
                occurrences=rows, optimizer_updates=updates,
                all_recorded_tail_inputs_equal=all(row['full_input_equal'] for row in rows),
                all_recorded_tail_losses_exact=all(row['loss_exact'] for row in rows),
                all_recorded_tail_states_exact=all(row['full_state_exact'] for row in updates),
                final_checkpoint_RNG_independently_compared=False,
                recovery_qualified=False, formal_performance=False)
