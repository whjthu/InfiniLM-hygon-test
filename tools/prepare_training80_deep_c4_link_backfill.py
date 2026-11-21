"""Expose the frozen I03 CPU link validation to the existing 640-card planner."""

import argparse
import json
from pathlib import Path

from report_training80_backlog import (
    BACKLOG, QUEUE_DB, REPO, TRAINING80_LEDGER, has_original_intent, invocation, need, pin_file, read_original_holds,
    read_queue_holds,
)
from verify_training80_manifest import manifest_members


TASK = dict(path='coordination/tasks/deep-c4-direct-history-guarded-link-cpu-v1.json',
            SHA256='233252c9ab5ece032eafa14224b3303827724a70ba37a48f4ce0286c865d5939')
ORIGINAL_BUILD = dict(path='results/bw1000/2026-11-20/deep-c4-direct-history-original487-compile-terminal.json',
                      SHA256='cf535eeb59b6d4e25ccc3ca6548214a3dd27fc77d5103257ec20ba097a0d5775')
PARENT, NODE = '1103606', 'f13r2n01'
LABEL = 'DeepC4-T1-history-guarded-probe-link-v1'
REQUEST = 'ae85c97293ae410483e55d94383a6c48'
IDENT = 'deep-c4-direct-history-guarded-link-cpu-v1'
INTENT = f'local-only/{LABEL}/intent.json'
SCOPE = dict(parent=PARENT, node=NODE, cpus=4, memory_MiB=8192, GPU_cards=0, original_deadline_seconds=300)


def prepare(repo, backlog, ledger, holds):
    repo = repo.resolve()
    need(NODE in ledger['groups']['I03']['allocations'][PARENT], 'original I03 node or parent differs')
    task = json.loads(pin_file(repo, TASK, ('coordination/tasks/',)))
    need(task['request'] == REQUEST and task['label'] == LABEL and task['placement'] == SCOPE and
         task['CPU_GO'] is True and task['GPU_GO'] is False and task['Model_GO'] is False and
         task['replay_allowed'] is False and task['new_translation_units'] == 0 and
         task['upstream_terminal_steps'] == [f'{PARENT}.156', f'{PARENT}.487'], 'original frozen CPU scope differs')
    manifest_ref = dict(path=task['source'] + '/SOURCE_FILES.json', SHA256=task['manifest_SHA256'])
    manifest = json.loads(pin_file(repo, manifest_ref, ('code/bw1000/',)))
    need((manifest['request'], manifest['label'], manifest['parent'], manifest['node']) ==
         (REQUEST, LABEL, PARENT, NODE), 'source manifest does not bind original request and placement')
    members = manifest_members(manifest)
    for ref in members.values():
        pin_file(repo, ref, ('code/bw1000/', 'tools/', 'results/bw1000/'))

    probe_ref = manifest['files']['repo/results/bw1000/2026-11-17/Deep483-guarded-CPU-v2-original-1103606.156-terminal.json']
    probe = json.loads(pin_file(repo, probe_ref, ('results/bw1000/',)))['independent_original_build']['result']
    accounting = probe['accounting']
    original = json.loads(pin_file(repo, ORIGINAL_BUILD, ('results/bw1000/',)))
    rows = [row.split('|') for row in original['slurm_steps']]
    need((accounting['job'], accounting['state'], accounting['exit_code'], accounting['node']) ==
         (f'{PARENT}.156', 'COMPLETED', '0:0', NODE) and probe['passive_ELF_verified'] is True and
         probe['object_SHA256'] == task['previous_guarded_probe_object_SHA256'] and
         original['result']['SHA256'] == task['original_candidate_receipt_SHA256'] and
         original['result']['scalars']['passed'] is True and original['actual_GPU'] == 0 and
         len(rows) == 1 and (rows[0][0], rows[0][2], rows[0][3]) == (f'{PARENT}.487', 'COMPLETED', '0:0'),
         'original probe or candidate terminal evidence differs')

    launcher = manifest['files']['launch.py']
    entry = invocation(repo, dict(launch=dict(launcher=launcher, publication_commit='<verified_publication>')))
    need(entry is not None and entry['kind'] == 'coordinator_callback', 'original bounded launcher interface differs')
    row = dict(id=IDENT, case='I03.deep-c4-guarded-probe-link', kind='validation', state='blocked', priority=900,
               placement={NODE: 0}, cpu_prerequisite=True, attempt_receipts=[INTENT], evidence=TASK,
               next_action='Use the frozen CPU receiver once after original Slurm and namespace checks. Reconcile any prior intent.')
    registered = [item for item in backlog['items'] if item['id'] == IDENT]
    need(not registered or registered == [row], 'registered Deep CPU backfill scope differs')
    items = backlog['items'] if registered else [*backlog['items'], row]
    attempted = has_original_intent(repo, [INTENT])
    ledger_nodes = {node for group in ledger['groups'].values() for nodes in group['allocations'].values() for node in nodes}
    direct_holds, _, _ = read_original_holds(repo, items, ledger_nodes, holds)
    queue_runs = sorted(hold['run'] for hold in holds if NODE in hold['nodes'])
    blocker_ids = sorted(ident for ident, hold in direct_holds.items() if ident != IDENT and NODE in hold['nodes'])
    occupied = queue_runs or blocker_ids
    state = ('RECONCILE_ORIGINAL_ATTEMPT' if attempted else 'WAITING_FOR_QUEUE_HOLD' if occupied else
             'WAITING_AUTHENTICATED_CPU_ADMISSION')
    details = dict(request=REQUEST, state=state, placement=SCOPE, original_steps=task['upstream_terminal_steps'],
                   source_manifest=manifest_ref, verified_source_members=len(members), launcher=launcher,
                   invocation_after_existing_checks=entry if not attempted and not occupied else None,
                   local_attempt_marker=INTENT, before_run_queue_holds=queue_runs, before_run_direct_holds=blocker_ids,
                   counts_toward_training_reserve=False,
                   actual_slurm_submissions=0, actual_DCU_started=0, gpu_eligible=False)
    return dict(backlog, items=items, deep_c4_link_backfill=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backlog', type=Path, default=BACKLOG)
    parser.add_argument('--queue-db', type=Path, default=QUEUE_DB)
    args = parser.parse_args()
    result = prepare(REPO, json.loads(args.backlog.read_bytes()), json.loads(TRAINING80_LEDGER.read_bytes()),
                     read_queue_holds(args.queue_db))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
