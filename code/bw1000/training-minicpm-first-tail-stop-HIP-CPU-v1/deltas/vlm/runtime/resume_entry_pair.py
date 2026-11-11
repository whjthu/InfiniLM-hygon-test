"""Compare bounded, SHA-bound first-tail metadata without claiming recovery."""
from __future__ import annotations

import hashlib
import json


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _validate(record, raw, start):
    plan = record['plan']
    count = (21 - start) * 4
    if (record.get('native_resume_window_complete') is not False
            or record.get('declared_observation_window_complete') is not True
            or record.get('diagnostic_stop_requested') is not True
            or record.get('diagnostic_stop_step') != 21 or record.get('completed_global_step') != 21
            or record.get('entry_state_step') != 20 or record.get('entry_state_captured') is not True
            or plan['checkpoint_global_step'] != start or plan['unchanged_max_steps'] != 25
            or plan['gradient_accumulation_steps'] != 4
            or plan['declared_observation_micro_forwards'] != count
            or plan['planned_micro_forwards_per_rank'] != (25 - start) * 4
            or record['saved_checkpoints'] != ([20] if start == 10 else [])
            or len(record['RNG_loads']) != 1 or record['RNG_loads'][0]['saved_equal'] is not True
            or len(record['inputs']) != count or len(record['outputs']) != count
            or [x['global_step'] for x in record['steps']] != list(range(start + 1, 22))):
        raise ValueError('incomplete declared first-tail observation window')
    for i, (before, after) in enumerate(zip(record['inputs'], record['outputs']), 1):
        if (before['ordinal'] != i or after['ordinal'] != i
                or before['global_step'] != start + (i - 1) // 4):
            raise ValueError('first-tail input/output occurrence differs')
        for value in (before['full_input'], after['loss']):
            digest = value['SHA256']
            if (value.get('finite') is not True or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('missing first-tail value identity')
    for step in record['steps']:
        if set(step['state']) != {'adapter', 'optimizer', 'scheduler'}:
            raise ValueError('incomplete first-tail full training state')
        for value in step['state'].values():
            digest = value['SHA256']
            if (value.get('finite') is not True or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('missing first-tail training-state identity')
    if (not isinstance(raw, bytes) or not 0 < len(raw) <= 8 * 1024**2
            or hashlib.sha256(raw).hexdigest() != record['entry_state_SHA256']):
        raise ValueError('forward-entry bytes differ from witness identity')
    entry = json.loads(raw)
    if (entry['schema'] != 1 or entry['global_step'] != 20
            or entry['forward_ordinal'] != (20 - start) * 4 + 1
            or entry['all_registered_parameters_and_buffers_observed'] is not True
            or entry['observer_RNG_unchanged'] is not True
            or _identity(entry['registered_state']) != entry['registered_state_SHA256']):
        raise ValueError('incomplete first-tail entry state')
    for key in ('RNG', 'context', 'input_layouts', 'input_alignments', 'selected_module_runtime',
                'loader_rng_types', 'accelerator_rng_types', 'runtime_cursor'):
        if key not in entry:
            raise ValueError('missing first-tail runtime field')
    registrations = entry['registered_state']
    if not registrations or len({(r['kind'], r['name']) for r in registrations}) != len(registrations):
        raise ValueError('duplicate registered entry-state identity')
    seen = set()
    for row in registrations:
        if (row['kind'] not in ('parameter', 'buffer') or not isinstance(row['name'], str)
                or type(row['present']) is not bool):
            raise ValueError('invalid registered entry-state identity')
        if row['present']:
            if 'alias_of' in row:
                if row['alias_of'] not in seen:
                    raise ValueError('unbound registered tensor alias')
            else:
                digest = row.get('SHA256', '')
                if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('missing registered tensor value identity')
        seen.add(row['kind'] + ':' + row['name'])
    return entry, record['inputs'][-4:], record['outputs'][-4:]


def compare_forward_entry_pair(continuous, restored, continuous_entry_bytes, restored_entry_bytes):
    left, inputs_a, outputs_a = _validate(continuous, continuous_entry_bytes, 10)
    right, inputs_b, outputs_b = _validate(restored, restored_entry_bytes, 20)
    if (continuous['rank'] != restored['rank'] or continuous['world_size'] != restored['world_size']
            or type(continuous['rank']) is not int
            or not 0 <= continuous['rank'] < continuous['world_size']):
        raise ValueError('paired rank or world size differs')
    for key in ('world_size_to_verify', 'per_device_batch', 'gradient_accumulation_steps',
                'dataloader_batches_per_rank_to_verify', 'effective_batch'):
        if continuous['plan'][key] != restored['plan'][key]:
            raise ValueError('paired data or accumulation recipe differs')
    if continuous['plan']['world_size_to_verify'] != continuous['world_size']:
        raise ValueError('declared topology differs from record')
    fields = {key + '_equal': left[key] == right[key] for key in
              ('RNG', 'context', 'input_layouts', 'input_alignments', 'selected_module_runtime',
               'loader_rng_types', 'accelerator_rng_types', 'runtime_cursor')}
    a = {(r['kind'], r['name']): r for r in left['registered_state']}
    b = {(r['kind'], r['name']): r for r in right['registered_state']}
    changed = sorted(set(a) ^ set(b) | {key for key in set(a) & set(b) if a[key] != b[key]})
    rows = [dict(tail_ordinal=i, full_input_equal=x['full_input'] == y['full_input'],
                 loss_exact=u['loss'] == v['loss']) for i, (x, y, u, v) in
            enumerate(zip(inputs_a, inputs_b, outputs_a, outputs_b), 1)]
    return dict(rank=continuous['rank'], world_size=continuous['world_size'],
                compared_tail_forwards=4, input_equal_forwards=sum(r['full_input_equal'] for r in rows),
                exact_loss_forwards=sum(r['loss_exact'] for r in rows), occurrences=rows,
                entry_fields=fields, registered_state_equal=not changed,
                changed_registrations=[dict(kind=kind, name=name) for kind, name in changed],
                first_tail_input_equal=rows[0]['full_input_equal'], first_tail_loss_exact=rows[0]['loss_exact'],
                final_observed_training_state_equal=continuous['steps'][-1]['state'] == restored['steps'][-1]['state'],
                full25_native_recovery_compared=False, unregistered_python_state_compared=False,
                inner_kernel_context_compared=False, numerical_cause_attributed=False,
                recovery_qualified=False, formal_performance=False)
