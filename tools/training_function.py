"""Bind the functional gate to a live registry, a shared entry and a real first update."""
import re
from pathlib import PurePosixPath

from read_training_slurm_membership import allocated_dcu, completed_numbered_step
from training_formal_metrics import need, sha


def verify_function(case, framework_commit, entry, registry, evidence, numerics_target, verify_raw):
    """Check that the Hygon numerical receipt came from the registered native entry."""
    source = registry.get('source', {}) if registry is not None else {}
    entry_sha = registry.get('entry_SHA256') if registry is not None else None
    tree = source.get('framework_tree')
    manifest = source.get('manifest_SHA256')
    need(source.get('framework_commit') == framework_commit and sha(entry_sha) and sha(manifest)
         and isinstance(tree, str) and re.fullmatch(r'[0-9a-f]{40}', tree),
         'functional acceptance requires a live registry for the exact framework commit')
    need(isinstance(entry, str) and entry.endswith('.py') and '\\' not in entry and ':' not in entry
         and not PurePosixPath(entry).is_absolute() and '..' not in PurePosixPath(entry).parts
         and PurePosixPath(entry).as_posix() == entry, 'missing shared registered entry path')

    need(evidence.get('hygon_execution') == numerics_target and sha(numerics_target.get('SHA256')),
         'function and numerics must reference the same Hygon execution')
    numerical = verify_raw(numerics_target)
    run_id, step = numerical.get('run_id'), numerical.get('slurm_step')
    binding = {'case': case['case'], 'framework_commit': framework_commit,
               'original_launch_SHA256': case['launch_SHA256'], 'platform': 'Hygon', 'run_id': run_id}
    need(sha(run_id) and all(numerical.get(key) == value for key, value in binding.items())
         and numerical.get('terminal_status') == 'COMPLETED' and type(numerical.get('exit_code')) is int
         and numerical['exit_code'] == 0 and type(numerical.get('cards')) is int
         and numerical['cards'] == case['original_cards'], 'functional numerical execution identity differs')
    updates = numerical.get('optimizer_update_counts', {})
    need(type(updates.get('zero_forward')) is int and updates['zero_forward'] == 0
         and type(updates.get('first_update')) is int and updates['first_update'] == 1,
         'functional execution lacks a zero-update forward and its first optimizer update')

    launch_ref, terminal_ref = evidence.get('launch_evidence'), evidence.get('terminal_evidence')
    need(isinstance(launch_ref, dict) and sha(launch_ref.get('SHA256'))
         and isinstance(terminal_ref, dict) and sha(terminal_ref.get('SHA256'))
         and len({launch_ref['SHA256'], terminal_ref['SHA256'], numerics_target['SHA256']}) == 3
         and terminal_ref['SHA256'] == numerical.get('terminal_receipt_SHA256'),
         'function needs distinct launcher, original terminal and numerical receipts')
    launch, terminal = verify_raw(launch_ref), verify_raw(terminal_ref)
    deployed = {'entry_path': entry, 'entry_SHA256': entry_sha,
                'framework_tree': tree, 'source_manifest_SHA256': manifest}
    need(all(launch.get(key) == value for key, value in (binding | deployed).items())
         and launch.get('slurm_step') == step and launch.get('selected_case') == case['case']
         and launch.get('run_requested') is True and launch.get('dry_run') is False,
         'actual execution did not use the verified case entry in training mode')
    need(all(terminal.get(key) == value for key, value in binding.items())
         and terminal.get('launch_receipt_SHA256') == launch_ref['SHA256'],
         'functional terminal does not bind the launcher and original numerical run')
    row = completed_numbered_step(terminal.get('accounting_columns'), terminal.get('accounting_row'),
                                  step, scope='functional')
    dcu = allocated_dcu(row['AllocTRES']) if isinstance(row['AllocTRES'], str) else None
    need(dcu == case['original_cards'], 'functional Slurm step does not have the original exact DCU count')
    return {'passed': True, 'slurm_step': step, 'cards': dcu}
