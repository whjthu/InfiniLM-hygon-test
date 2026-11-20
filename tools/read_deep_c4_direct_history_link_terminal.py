"""Independently reconcile the one published guarded-probe CPU link."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re

from verify_deep_c4_direct_history_link import (CANDIDATE_DIR, CANDIDATE_NATIVE, CANDIDATE_RECEIPT_SHA, PLAN,
    PLAN_SHA, PROBE_REPORT, PROBE_REPORT_SHA, RELINKED_PROBE, REPO, digest, link_argv,
    parse_candidate_receipt, read_pinned_json, require)


TASK = 'coordination/tasks/deep-c4-direct-history-guarded-link-cpu-v1.json'
COLUMNS = 'JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}


def hex_pin(value, digits):
    return isinstance(value, str) and re.fullmatch(f'[0-9a-f]{{{digits}}}', value) is not None


def read_json(path):
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    actual_sha = sha256(raw).hexdigest()
    require(0 < len(raw) <= 1024 * 1024, f'{path.name} length outside the reader bound')
    document = json.loads(raw)
    require(isinstance(document, dict), f'{path.name} must be a JSON object')
    return document, actual_sha


def accounting(raw, claim):
    """Consume one exact label and numbered step. Empty/nonterminal remains unknown."""
    placement = claim['placement']
    parent, label = placement['parent'], claim['label']
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    require(all(len(row) == 7 for row in rows), 'accounting columns do not match the collector contract')
    exact = [row for row in rows if row[1] == label]
    require(len(exact) <= 1, 'ambiguous accounting: repeated exact CPU relink label')
    if not exact:
        return None
    step, _, raw_state, code, elapsed, node, tres_raw = exact[0]
    require(re.fullmatch(re.escape(parent) + r'\.[1-9][0-9]*', step) and step not in claim['upstream_terminal_steps'],
            'CPU relink accounting does not identify a distinct numbered step')
    state = raw_state.split(' ', 1)[0]
    result = dict(step=step, state=state, raw_state=raw_state, exit_code=code, terminal=state in TERMINAL,
                  node=node, AllocTRES=tres_raw)
    if not result['terminal']:
        return result
    require(re.fullmatch(r'[0-9]+:[0-9]+', code) and elapsed.isdecimal(), 'invalid terminal accounting counter')
    elapsed = int(elapsed)
    require(elapsed <= placement['original_deadline_seconds'], 'CPU accounting exceeded original budget')
    result.update(elapsed_s=elapsed, successful=state == 'COMPLETED' and code == '0:0')
    if result['successful']:
        tres = dict(field.split('=', 1) for field in tres_raw.split(','))
        require(node == placement['node'] and tres.get('cpu') == str(placement['cpus'])
                and tres.get('mem') in ('8G', '8192M') and tres.get('node') == '1'
                and not any(key.startswith('gres') or key in ('gpu', 'dcu') for key in tres),
                'successful step did not use the recorded zero-GPU CPU placement')
    return result


def bind_host(claim, intent, launch, publication):
    placement = claim['placement']
    identity = dict(request=claim['request'], publication=publication, manifest_SHA256=claim['manifest_SHA256'],
                    parent=placement['parent'], node=placement['node'], GPU=False)
    for evidence in (intent, launch):
        require(all(type(evidence[key]) is type(value) and evidence[key] == value for key, value in identity.items()),
                'original HOST intent or launch identity differs')
    epoch, deadline = intent['epoch'], intent['deadline']
    require(all(type(x) in (float, int) and math.isfinite(x) for x in (epoch, deadline)) and epoch > 0
            and deadline - epoch == placement['original_deadline_seconds']
            and (epoch, deadline) == (launch['epoch'], launch['deadline'])
            and intent['replay_allowed'] is False, 'original HOST epoch, deadline or replay flag differs')
    require(all(type(launch[field]) is int and launch[field] > 0 for field in ('pid', 'pgid', 'startticks'))
            and launch['pid'] == launch['pgid'] and isinstance(launch['host'], str) and bool(launch['host']),
            'owned launcher process identity incomplete')
    return {**identity, 'source_manifest_SHA256': identity['manifest_SHA256'], 'epoch': epoch, 'deadline': deadline}


def read_completed(terminal, binding, read, artifact_sha, upstream_candidate_sha, original, plan):
    """Keep scheduler, application and artifact success independent."""
    result = dict(accounting=terminal, output_consumed=False, cpu_link_artifact_qualified=False,
                  replay_allowed=False, GPU_GO=False, Model_GO=False, runtime_provider_verified=False)
    if terminal is None or not terminal['terminal']:
        return dict(result, state='ORIGINAL_TERMINAL_NOT_OBSERVED')
    try:
        receipt, receipt_sha = read('CPU-result.json')
    except FileNotFoundError:
        if terminal['successful']:
            raise ValueError('completed CPU step has no original result bytes') from None
        return dict(result, state='ORIGINAL_SLURM_FAILED_RESULT_MISSING')
    expected = {key: binding[key] for key in ('request', 'publication', 'source_manifest_SHA256', 'node',
                                               'epoch', 'deadline')}
    expected.update(step=terminal['step'], completed=True, new_translation_units=0,
                    actual_GPU_calls=0, checkpoint_weight_reads=0, GPU_GO=False, Model_GO=False,
                    runtime_provider_verified=False)
    require(all(type(receipt.get(key)) is type(value) and receipt[key] == value for key, value in expected.items()),
            'CPU result does not bind to the original Slurm and HOST identity')
    result.update(output_consumed=True, CPU_result_SHA256=receipt_sha, linker_attempts=receipt['linker_attempts'])
    if not terminal['successful'] or receipt.get('passed') is not True:
        return dict(result, state='ORIGINAL_CPU_LINK_FAILED', failed_check=receipt.get('failed_check'))

    worker_intent, worker_intent_sha = read('intent.json')
    require(all(type(worker_intent.get(key)) is type(value) and worker_intent[key] == value
                for key, value in expected.items() if key != 'completed')
            and worker_intent['guarded_probe_object_reuse_planned'] is True
            and worker_intent['replay_allowed'] is False, 'worker intent does not bind the original CPU attempt')
    proof, proof_sha = read('link-plan.json')
    attempt, attempt_sha = read('link-attempt.json')
    link, = plan['links']
    baseline = link['required_NEEDED']
    candidate_sha = upstream_candidate_sha()
    expected_inputs = {
        plan['candidate'] + '/probe.o': original['object_SHA256'],
        plan['candidate'] + '/' + link['output']: original['probe_SHA256'],
        baseline: original['project_provider_SHA256'][baseline],
        CANDIDATE_NATIVE: candidate_sha,
    }
    expected_needed = [CANDIDATE_NATIVE if item == baseline else item for item in original['needed']]
    require(receipt['linker_attempts'] == 1 and receipt['linker_returncode'] == 0
            and receipt['guarded_probe_object_reused'] is True
            and proof['linker_argv'] == link_argv(plan, link)
            and proof['candidate_receipt_SHA256'] == CANDIDATE_RECEIPT_SHA
            and proof['input_SHA256'] == expected_inputs
            and proof['expected_DT_NEEDED'] == receipt['expected_DT_NEEDED'] == expected_needed,
            'successful CPU result did not preserve original object or linkage pins')
    seconds = attempt['linker_timeout_seconds']
    require(attempt['request'] == binding['request'] and attempt['replay_allowed'] is False
            and attempt['original_deadline'] == binding['deadline']
            and type(seconds) in (int, float) and 2 <= seconds <= 90, 'original bounded link attempt differs')
    actual_probe_sha = artifact_sha()
    require(hex_pin(actual_probe_sha, 64) and actual_probe_sha == receipt['relinked_probe_SHA256'],
            'relinked probe bytes differ from the original CPU result')
    return dict(result, cpu_link_artifact_qualified=True, state='ORIGINAL_CPU_LINK_COMPLETED_ARTIFACT_MATCHED',
                worker_intent_SHA256=worker_intent_sha, link_plan_SHA256=proof_sha,
                link_attempt_SHA256=attempt_sha, probe_SHA256=actual_probe_sha,
                candidate_native_SHA256=candidate_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='original HOST project directory')
    parser.add_argument('--publication', required=True, help='verified published commit used for submission')
    parser.add_argument('--accounting', type=Path, required=True, help=f'original sacct -nP -o {COLUMNS} output')
    args = parser.parse_args()
    require(hex_pin(args.publication, 40), 'expected exact original publication')
    claim, _ = read_json(REPO / TASK)
    label = claim['label']
    scope = args.root / 'candidates' / label
    intent, intent_sha = read_json(args.root / 'results' / (label + '-intent.json'))
    launch, launch_sha = read_json(args.root / 'results' / (label + '-launch.json'))
    require(launch['intent'] == str((args.root / 'results' / (label + '-intent.json')).as_posix()),
            'launch did not bind the exact original intent path')
    binding = bind_host(claim, intent, launch, args.publication)
    with args.accounting.open('rb') as stream:
        raw = stream.read(400001)
    require(len(raw) <= 400000, 'original accounting response is oversized')
    terminal = accounting(raw.decode('utf-8'), claim)
    plan = read_pinned_json(REPO / PLAN, PLAN_SHA)
    original = read_pinned_json(REPO / PROBE_REPORT, PROBE_REPORT_SHA)['independent_original_build']['result']
    upstream = args.root.joinpath(*PurePosixPath(CANDIDATE_DIR).relative_to('/refactor').parts, 'CPU-result.json')

    def upstream_candidate_sha():
        return parse_candidate_receipt(read_pinned_json(upstream, CANDIDATE_RECEIPT_SHA))

    result = read_completed(terminal, binding, lambda name: read_json(scope / name),
                            lambda: digest(scope / PurePosixPath(RELINKED_PROBE).name),
                            upstream_candidate_sha, original, plan)
    result.update(intent_SHA256=intent_sha, launch_SHA256=launch_sha, accounting_SHA256=sha256(raw).hexdigest())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
