"""Observe the frozen VGGT dev24 backfill staging and original non-submission."""

import json
import os
import re
import subprocess

from bind_vggt_tum_cross_camera_heads import ARMS, digest, is_sha, read_bounded
from bundle_vggt_tum_cross_camera_dev import MANIFEST
from control_vggt_tum_cross_camera_dev import (
    BUNDLE_SHA, COHORT, COLUMNS, INPUTS_SHA, LABEL, NODE, PARENT, REPO, RESULTS, SOURCE_FILES,
    STAGE_NAME, STAGES, membership, pin, require,
)


MANIFEST_PATH = 'coordination/training/vggt-tum-cross-camera-controller-files-v1.json'
MANIFEST_SHA = 'f224f502dc9b5755182f458591e47cf5214f7484dc0c0480d4028d5dade0a910'
ACCOUNTING = ('sacct', '-n', '-P', '-j', PARENT, '-S', '2026-11-19T00:00:00', '-o', COLUMNS)
LABELS = {LABEL + '-' + phase for phase in STAGES}
ORIGINAL_ARTIFACTS = ('score-intent.json', 'score-outcome.json', 'assignment.json', 'before.json', 'after.json',
                      'score', *(phase + suffix for phase in STAGES for suffix in ('.log', '-accounting.json')))


def present(path):
    return path.exists() or path.is_symlink()


def original_observation(root, read_accounting):
    markers = [name for name in ORIGINAL_ARTIFACTS if present(root / name)]
    raw = read_accounting()
    require(len(raw) <= 2 * 1024**2, 'oversized Slurm accounting observation')
    rows = [line.split('|') for line in raw.decode('utf-8').splitlines() if line.strip()]
    require(all(len(row) == 7 for row in rows), 'expected seven explicit Slurm accounting columns')
    require(sum(row[0] == PARENT for row in rows) == 1, 'original scoring parent missing or duplicated in accounting')
    matches = [row for row in rows if row[1] in LABELS]
    require(all(re.fullmatch(PARENT + r'\.[0-9]+', row[0]) and row[5] == NODE for row in matches),
            'original scoring label has foreign step or node')
    return dict(markers=markers, rows=matches, accounting_SHA256=digest(raw))


def verify_controller_source(repo):
    manifest = json.loads(pin(repo / MANIFEST_PATH, MANIFEST_SHA))
    require(set(manifest) == set(SOURCE_FILES), 'controller manifest does not bind its exact source files')
    for name, wanted in manifest.items():
        pin(repo / name, wanted)


def staged_assets(repo, root):
    require(root.resolve(strict=True) == root.absolute(), 'aliased original staging directory')
    verify_controller_source(repo)
    raw = read_bounded(root / 'stage-complete.json')
    stage = json.loads(raw)
    inputs = stage['input_staging']
    require(stage['state'] == 'INPUTS_STAGED_ORIGINAL_TRAINING_RECEIPTS_MATCHED' and
            stage['new_slurm_steps'] == stage['optimizer_updates'] == stage['actual_checkpoint_bytes_read'] == 0 and
            stage['GPU_device_admitted'] is False and stage['scorer_permission'] is False and
            inputs['inputs_SHA256'] == INPUTS_SHA and inputs['bundle_SHA256'] == BUNDLE_SHA and
            tuple(inputs[name] for name in ('input_pairs', 'planned_windows', 'verified_members')) == COHORT,
            'original staged input or no-submission boundary differs')
    pin(root / 'original-training-accounting.psv', stage['accounting_SHA256'])
    pin(root / 'inputs' / MANIFEST, INPUTS_SHA)
    heads = json.loads(pin(root / 'head-authority.json', stage['head_authority_SHA256']))
    require(set(heads) == set(ARMS), 'expected both original trained heads')
    for arm, (gpu, request, clean, prior) in ARMS.items():
        head = heads[arm]
        require((head['gpu_step'], head['request'], head['clean_post'], head['training_receipt_SHA256']) ==
                (gpu, request, clean, prior) and head['global_step'] == 25 and head['actual_checkpoint_bytes_read'] == 0
                and is_sha(head['file_SHA256']) and is_sha(head['module_SHA256']), 'original head authority differs')
    return dict(stage_SHA256=digest(raw), manifest_SHA256=MANIFEST_SHA,
                head_authority_SHA256=stage['head_authority_SHA256'], original_heads=len(heads))


def preflight(repo, results, read_accounting, observe_node):
    root = results / STAGE_NAME
    result = dict(case='T12.domain-tum-cross-camera-dev24', label=LABEL, parent=PARENT, node=NODE,
                  requested_DCU=1, planned_windows=24, optimizer_updates=0, controller_seconds=1500,
                  counts_toward_training_reserve=False, actual_checkpoint_bytes_read=0, new_slurm_steps=0,
                  device_admitted=False, approved_for_dispatch=False)
    first = original_observation(root, read_accounting)
    result['original_observations'] = [first]
    if first['markers'] or first['rows']:
        return dict(result, state='RECONCILE_ORIGINAL_SCORING')
    if not present(root):
        return dict(result, state='TARGET_STAGING_REQUIRED')
    if not present(root / 'stage-complete.json'):
        return dict(result, state='RECONCILE_ORIGINAL_STAGING')
    assets = staged_assets(repo, root)
    membership_state = observe_node()
    last = original_observation(root, read_accounting)
    result['original_observations'].append(last)
    result.update(assets, slurm_membership_state=membership_state)
    if last['markers'] or last['rows']:
        return dict(result, state='RECONCILE_ORIGINAL_SCORING')
    if membership_state != 'NO_NUMBERED_STEP':
        return dict(result, state='WAITING_FOR_NUMBERED_STEP_CLEARANCE')
    return dict(result, state='ORIGINAL_STAGING_VERIFIED_DEVICE_AND_HOLDS_PENDING')


def observe_node():
    ledger = json.loads(read_bounded(REPO / 'coordination/training/training80-active-node-task-map-v1.json'))
    groups = {job['JobId']: {zone: entry['allocations'][job['JobId']] for zone, entry in ledger['groups'].items()}
              for job in ledger['jobs']}
    pool = membership(groups, owner_uid=os.getuid())
    require(pool['membership_qualified'], 'current Slurm ownership or steps are unavailable')
    return pool['parents'][PARENT]['coverage_by_group_and_node']['T12'][NODE]['state']


def main():
    result = preflight(REPO, RESULTS, lambda: subprocess.check_output(ACCOUNTING, timeout=8), observe_node)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
