"""Check first-forward records from three independent processes per rank."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rng_treatment import ARMS, SEED_BASE, check_transition, need


ORIGINAL_ROUTE_SHA = 'db7bcb1609050b843b8260b018e61849f683a91af5991fdb52149141e5156b4a'
KNOWN_ROUTE = {'selected_SDPA': {'aten::_scaled_dot_product_flash_attention': 839},
               'SDPA_wrappers': {'aten::scaled_dot_product_attention': 839},
               'native_dropout': {'aten::dropout': 186, 'aten::native_dropout': 186},
               'SDPA_route_observed': True}


def read(path, limit=2*1024**2, digest=None):
    need(path.resolve(strict=True) == path.absolute(), 'aliased diagnostic input')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and (digest is None or sha256(raw).hexdigest() == digest),
         'diagnostic evidence identity differs: ' + path.name)
    return json.loads(raw)


def check_arm(arm, rank, run_step, observed, applied, audit, stopped, ready):
    match = applied['run']
    need(match == audit['run'] and match['arm'] == arm and match['rank'] == rank and match['world_size'] == 8
         and match['checkpoint_step'] == 20 and match['actual_step'] == run_step
         and type(match['pid']) is int and match['pid'] > 1 and type(match['startticks']) is int and match['startticks'] > 0
         and match['boot_id'] and ready['run']['actual_step'] == run_step
         and ready['run']['config_SHA256'] == match['config_SHA256']
         and ready['state'] == 'FORWARD_RETURNED' and ready['rank'] == rank and ready['world_size'] == 8
         and ready['checkpoint_step'] == 20, 'wrong rank, Slurm step, config or native process')
    delta = applied['treatment']
    need(delta['arm'] == arm and delta['rank'] == rank and delta['seed'] == (SEED_BASE+rank if arm == 'b' else None),
         'wrong declared per-rank treatment')
    check_transition(delta['before'], delta['after'], arm, rank, delta['requested_device_SHA256'])
    need(observed['rank'] == rank and observed['world_size'] == 8
         and observed['checkpoint_step'] == observed['global_step'] == 20
         and observed['completed_forwards'] == observed['planned_forwards'] == 1
         and observed['backward_calls'] == observed['optimizer_updates'] == audit['backward_calls'] == audit['optimizer_updates'] == 0
         and observed['state'] == 'STOP_BEFORE_BACKWARD'
         and stopped == dict(rank=rank, state='ALL_RANKS_STOPPED', global_step=20, backward_calls=0, optimizer_updates=0)
         and audit['parameter_identity_and_versions_unchanged'] is True
         and audit['root_input_identity_and_versions_unchanged'] is True and audit['training_modes_unchanged'] is True,
         'native forward denominator or mutation gate differs')
    need(observed['RNG_after_native_restore'] == delta['before']['full']
         and observed['RNG_before_compute_loss'] == observed['root_before_forward']['RNG'] == delta['after']['full'],
         'RNG changed between intervention and native root')
    need(observed['actual_input']['ordinal'] == observed['actual_loss']['ordinal'] == 1
         and observed['actual_input']['global_step'] == 20
         and observed['operators'] == KNOWN_ROUTE, 'original microbatch or flash/dropout route differs')
    value = observed['actual_loss']['loss_scalar']
    need(type(value) is float and math.isfinite(value), 'invalid actual loss')
    return dict(owner=(match['boot_id'], match['pid'], match['startticks']), run=ready['run'],
                original_rng=delta['before']['full'], ending_rng=observed['RNG_after_forward'],
                root_input=observed['actual_input'], restored=observed['restored_state'],
                context=dict(ddp=observed['ddp'], language=observed['language_before_forward'],
                             inner_linear=observed['inner_linear'], compute_loss_source=observed['compute_loss_source'],
                             root={k: v for k, v in observed['root_before_forward'].items() if k != 'RNG'}),
                state_before=audit['before'], state_after=audit['after'],
                opaque=audit['opaque_cache_attrs_before'] + audit['opaque_cache_attrs_after'],
                cache_changed=audit['observed_buffer_rotary_changed'],
                loss=(observed['actual_loss']['loss'], value))


def compare_rank(rows, historical):
    need(set(rows) == set(ARMS), 'incomplete per-rank A/B/A bracket')
    a1, b, a2 = (rows[arm] for arm in ARMS)
    need(len({row['owner'] for row in rows.values()}) == 3 and len({row['owner'][0] for row in rows.values()}) == 1,
         'arms are reused processes or different physical boots')
    held = ('original_rng', 'restored', 'root_input', 'context', 'state_before')
    need(all(a1[key] == b[key] == a2[key] for key in held), 'uncontrolled initial state, batch or software path')
    need(a1['root_input']['full_input'] == historical['input'] and a1['restored'] == historical['state']
         and a1['original_rng'] == historical['RNG'], 'native first microbatch/checkpoint does not match .1171')
    baseline = a1['loss'] == a2['loss']
    contrast = b['loss'] != a1['loss'] and b['loss'] != a2['loss']
    return dict(A_loss_bitwise_equal=baseline, A_ending_rng_equal=a1['ending_rng'] == a2['ending_rng'],
                B_loss_differs_from_both_A=contrast, A_abs_loss_delta=abs(a1['loss'][1]-a2['loss'][1]),
                B_minus_A1=b['loss'][1]-a1['loss'][1], B_minus_A2=b['loss'][1]-a2['loss'][1],
                observed_cache_change_by_arm={arm: row['cache_changed'] for arm, row in rows.items()},
                A_end_model_state_equal=a1['state_after'] == a2['state_after'],
                opaque_cache_counts_by_arm={arm: len(row['opaque']) for arm, row in rows.items()})


def consume(root, step, original_route, original_history):
    route = read(original_route, 8192, ORIGINAL_ROUTE_SHA)
    counts = {operator: count for category in ('selected_SDPA', 'SDPA_wrappers', 'native_dropout')
              for operator, count in KNOWN_ROUTE[category].items()}
    need(route['gpu_step'] == '1103606.1189' and route['optimizer_updates'] == 0
         and route['identical_CPU_profiler_ATen_counts_in_each_rank'] == counts,
         'wrong actual flash/dropout baseline')
    need(re.fullmatch(r'1103606\.\d+', step) and step not in ('1103606.1171', '1103606.1189'),
         'expected a new exact eight-DCU accounting step')
    records = {rank: {} for rank in range(8)}
    errors = []
    for arm in ARMS:
        for rank in range(8):
            folder = root / arm / 'train/minicpm-usgs-witness' / f'rank-{rank:02d}'
            try:
                records[rank][arm] = check_arm(arm, rank, step, read(folder / 'first-forward-state.json'),
                    read(folder / 'rng-treatment.json', 8192), read(folder / 'rng-state-audit.json', 16384),
                    read(folder / 'first-forward-stop.json', 4096), read(folder / 'first-forward-ready.json', 4096))
            except (OSError, KeyError, TypeError, ValueError) as error:
                errors.append(dict(arm=arm, rank=rank, error=type(error).__name__, detail=str(error)[:180]))
    owners = [item['owner'] for rows in records.values() for item in rows.values()]
    contexts = [item['run'] for rows in records.values() for item in rows.values()]
    valid = []
    if len(owners) != len(set(owners)):
        errors.append(dict(error='OWNED_RANK_PROCESS_REUSED_ACROSS_ARMS'))
    elif len({owner[0] for owner in owners}) > 1:
        errors.append(dict(error='PHYSICAL_BOOT_CHANGED'))
    elif contexts and any(context != contexts[0] for context in contexts):
        errors.append(dict(error='SHARED_NATIVE_EPOCH_DEADLINE_OR_CHECKPOINT_CONFIG_CHANGED'))
    else:
        for rank, rows in records.items():
            try:
                valid.append(dict(rank=rank, **compare_rank(rows, original_history[rank])))
            except (KeyError, TypeError, ValueError) as error:
                errors.append(dict(rank=rank, error=type(error).__name__, detail=str(error)[:180]))
    comparable = len(valid) == 8 and not errors
    return dict(scope='first restored microbatch for native MiniCPM, three fresh processes per rank',
                actual_step=step, planned_forwards=24, verified_individual_forwards=len(owners), comparable_ranks=len(valid),
                complete=comparable, rows=valid, errors=errors, historical_cause_proven=False,
                GPU_nondeterminism_proven=False, formal_recovery_qualified=False,
                interpretation=('CONTROL_REPEAT_DIFFERS' if any(
                    not (row['A_loss_bitwise_equal'] and row['A_ending_rng_equal'] and row['A_end_model_state_equal']) for row in valid)
                    else 'OBSERVED_RNG_SENSITIVITY' if any(row['B_loss_differs_from_both_A'] for row in valid)
                    else 'NO_OBSERVED_LOSS_RESPONSE_TO_THIS_RNG_STATE') if comparable else 'INCOMPLETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'original-route', 'original-root', 'original-public', 'original-observer'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--gpu-step', required=True)
    args = parser.parse_args()
    observer = args.original_observer.resolve(strict=True)
    with (observer / 'compare_first.py').open('rb') as stream:
        raw = stream.read(6594)
    need(len(raw) == 6593 and sha256(raw).hexdigest() ==
         'e11c8b7b0955be3ea4b3e35dc8d747cb7edfbf51f62a0416ffad765b406200dd', 'wrong original comparator')
    with (observer / 'probe.py').open('rb') as stream:
        raw = stream.read(10996)
    need(len(raw) == 10995 and sha256(raw).hexdigest() ==
         '1d340abb7659f30cb625fd21164f47168b16267fccf86f9cb8da73edcc6bfdc4', 'wrong original probe dependency')
    sys.path.insert(0, str(observer))
    from compare_first import original_rows
    history = original_rows(args.original_root, args.original_public)
    result = consume(args.root.resolve(), args.gpu_step, args.original_route.resolve(), history)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return int(not result['complete'])


if __name__ == '__main__':
    raise SystemExit(main())
