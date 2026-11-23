"""Verify DP2 rank timing and fresh-process recovery from sealed native results."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


BOUNDARY = ('before first measured update through completed final two-rank checkpoint commit and barrier; '
            'includes replicated-state checks and checkpoint IO; excludes loading and warmup')
SCHEMA = 'hunyuanvideo-dp2-replicated-full-step-v2'
SLURM_COLUMNS = 'JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def checked_metadata(path, expected):
    require(path.is_file() and not path.is_symlink(), 'missing or linked evidence file')
    with path.open('rb') as stream:
        raw = stream.read((2 << 20) + 1)
    require(0 < len(raw) <= 2 << 20 and sha256(raw).hexdigest() == expected,
            'sealed metadata hash or length differs')
    return raw


def sealed_by_inventory(path, inventory):
    pins = [row['sha256'] for row in inventory if Path(row['path']).resolve() == path.resolve()]
    require(len(pins) == 1 and is_sha(pins[0]), 'native metadata inventory is missing or ambiguous')
    return pins[0], checked_metadata(path, pins[0])


def verify_bytes(path, expected, *, minimum=1):
    require(path.is_file() and not path.is_symlink(), 'missing sealed native file')
    digest = sha256()
    total = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            total += len(block)
    require(total >= minimum and digest.hexdigest() == expected, 'sealed native file bytes differ')


def audit_run(root, evidence_sha, backend_sha, start):
    evidence = json.loads(checked_metadata(root / 'dp2-evidence.json', evidence_sha))
    require(evidence['schema'] == 'hunyuanvideo-dp2-post-torchrun-evidence-v1'
            and type(evidence['torchrun_exit_code']) is int and evidence['torchrun_exit_code'] == 0
            and is_sha(backend_sha) and evidence['backend_SHA256'] == backend_sha,
            'wrong backend or missing successful torchrun exit')
    verify_bytes(root / 'torchrun.log', evidence['torchrun_log_SHA256'], minimum=0)
    request = json.loads(checked_metadata(root / 'dp2-request.json', evidence['request_SHA256']))
    result = json.loads(checked_metadata(root / 'dp2-result.json', evidence['result_SHA256']))
    config = request['config']
    require(config['world_size'] == 2 and config['sp'] == 1 and config['warmup_steps'] == 5
            and config['max_steps'] == 20 and len(set(config['cards'])) == 2
            and result['checks']['replicated_state_agreement'] is True
            and (config.get('resume_from_checkpoint') is not None) == (start > 0), 'DP2 recipe or run kind differs')

    ranks, streams, restored, contracts = [], [], [], []
    for rank in (0, 1):
        _, raw_summary = sealed_by_inventory(root / f'rank-{rank}' / 'summary.json', result['logs'])
        _, raw_steps = sealed_by_inventory(root / f'rank-{rank}' / 'steps.ndjson', result['logs'])
        summary = json.loads(raw_summary)
        seconds = summary['measurement_wall_seconds']
        require(summary['rank'] == rank and is_sha(summary['process_identity_SHA256'])
                and summary['completed_global_step_at_start'] == start
                and summary['timing_boundary'] == BOUNDARY and summary['measurement_steps'] == 25 - max(start, 5)
                and finite(seconds) and seconds > 0
                and finite(summary['model_load_seconds']) and finite(summary['cache_preparation_seconds']),
                'rank identity or timed boundary differs')
        records = [json.loads(line) for line in raw_steps.splitlines()]
        require([row['global_step'] for row in records] == list(range(start + 1, 26))
                and all(row['rank'] == rank and finite(row['loss']) and finite(row['grad_norm_before_clip'])
                        and row['grad_norm_before_clip'] > 0
                        and all(is_sha(row[field]) for field in ('replica_sha256', 'rank_state_sha256',
                                                               'rng_after_sha256', 'flow_input_sha256'))
                        for row in records), 'rank update denominator or values differ')
        boundary = summary['restored_boundary']
        require((boundary is None and start == 0) or (start > 0 and is_sha(boundary['source_marker_SHA256'])
                and is_sha(boundary['restored_rank_state_SHA256'])), 'rank restore evidence differs')
        ranks.append(dict(rank=rank, seconds=seconds, process_identity_SHA256=summary['process_identity_SHA256'],
                          slurm_device=summary.get('slurm_device'),
                          model_load_seconds=summary['model_load_seconds'],
                          cache_preparation_seconds=summary['cache_preparation_seconds']))
        restored.append(boundary)
        streams.append(records)
        contracts.append(summary['contract'])

    require(all(a['replica_sha256'] == b['replica_sha256'] for a, b in zip(*streams, strict=True)),
            'replica state differs between ranks')
    require(isinstance(contracts[0], dict) and contracts[0] and contracts[0] == contracts[1],
            'rank native training contracts differ')
    seconds = [row['seconds'] for row in ranks]
    rate = (25 - max(start, 5)) / max(seconds)
    require(evidence['rank_window_metrics'] == dict(rank_measurement_wall_seconds=seconds,
            slowest_rank_measurement_wall_seconds=max(seconds), measurement_updates_per_second=rate,
            timing_boundary=BOUNDARY, complete_5_warmup_20_measured_window=start == 0),
            'sealed result does not use the slowest complete rank window')

    markers = {}
    for step in (step for step in (10, 20, 25) if step > start):
        path = root / f'checkpoint-{step}' / 'COMMITTED.json'
        pin, raw_marker = sealed_by_inventory(path, result['artifacts'])
        marker = json.loads(raw_marker)
        require(marker['schema'] == SCHEMA and marker['global_step'] == step
                and [row['rank'] for row in marker['ranks']] == [0, 1]
                and all(row['replica_sha256'] == streams[row['rank']][step-start-1]['replica_sha256']
                        and is_sha(row['sha256']) for row in marker['ranks']), 'checkpoint logical replica differs')
        for row in marker['ranks']:
            verify_bytes(path.with_name(f"rank-{row['rank']}.pt"), row['sha256'])
        markers[step] = pin
    phase_name = {0: 'continuous', 10: 'resume10', 20: 'resume20'}[start]
    require(config.get('run_name') == phase_name and root.name == phase_name,
            'DP2 run name does not identify the frozen phase')
    unchanged_config = {key: value for key, value in config.items()
                        if key not in ('run_name', 'resume_from_checkpoint')}
    return dict(evidence_SHA256=evidence_sha, ranks=ranks, measurement_updates_per_second=rate, markers=markers,
                restored=restored, streams=streams, source_checkpoint=config.get('resume_from_checkpoint'),
                training_binding=(unchanged_config, contracts[0], request['weights'], request['source_root']))


def audit_slurm(runs, accounting):
    path, pin = accounting
    raw = checked_metadata(path, pin)
    # Compare rank placement with independent sacct -n -P -j <steps> --format=<SLURM_COLUMNS>.
    rows = {}
    for line in raw.decode('utf-8').splitlines():
        require(len(line.split('|')) == 6, 'unexpected sacct columns')
        step, state, rc, elapsed, node, tres = line.split('|')
        require(re.fullmatch(r'[0-9]+\.[0-9]+', step) and step not in rows, 'invalid or duplicate sacct step')
        require(state == 'COMPLETED' and rc == '0:0' and elapsed.isascii() and elapsed.isdecimal(),
                'Slurm training step has no successful terminal row')
        resources = [item.split('=', 1) for item in tres.split(',')]
        require(all(len(item) == 2 and item[0] and item[1] for item in resources)
                and len({item[0] for item in resources}) == len(resources), 'invalid Slurm resources')
        allocation = dict(resources)
        generic = allocation.get('gres/dcu')
        typed = [value for name, value in resources if name.startswith('gres/dcu:')]
        require(all(value.isascii() and value.isdecimal() for value in typed), 'invalid typed Slurm DCU count')
        require(allocation.get('node') == '1' and generic in (None, '2')
                and (sum(map(int, typed)) == 2 if typed else generic == '2'),
                'Slurm did not assign exactly two DCUs on one node')
        rows[step] = dict(node=node, elapsed_seconds=int(elapsed), accounted_measured_seconds=0.0)

    observed = {}
    for start, ranks in runs.items():
        placement = [rank['slurm_device'] for rank in ranks]
        require(all(isinstance(row, dict) and set(row) == {'step', 'node', 'pci'} for row in placement),
                'rank physical Slurm placement is missing')
        step, node = placement[0]['step'], placement[0]['node']
        require(step in rows and node == rows[step]['node']
                and all(row['node'] == node and row['step'] == step for row in placement),
                'rank location does not match Slurm step')
        pci = [row['pci'] for row in placement]
        require(all(isinstance(p, str) and re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', p)
                    for p in pci), 'invalid physical device address')
        canonical_pci = [f'{int(p.split(":", 1)[0], 16):04x}:{p.split(":", 1)[1]}' for p in pci]
        require(len(set(canonical_pci)) == 2, 'ranks do not identify two distinct physical devices')
        rows[step]['accounted_measured_seconds'] += max(rank['seconds'] for rank in ranks)
        observed[start] = dict(step=step, node=node, rank_pci=canonical_pci)
    require({row['step'] for row in observed.values()} == set(rows), 'unexpected accounted Slurm steps')
    for row in rows.values():
        require(0 < row['accounted_measured_seconds'] <= row['elapsed_seconds'] + 1,
                'DP2 measured windows exceed accounted Slurm duration')
    return dict(accounting_SHA256=pin, accounting_columns=SLURM_COLUMNS.split(','), steps=rows, runs=observed)


def audit_recovery(continuous_root, resumes, backend_sha, *, accounting=None):
    root, pin = continuous_root
    require(set(resumes) == {10, 20}, 'DP2 must restore both original checkpoint steps')
    initial = audit_run(root, pin, backend_sha, 0)
    processes = {rank['process_identity_SHA256'] for rank in initial['ranks']}
    require(len(processes) == 2, 'continuous DP2 rank processes are not distinct')
    bindings = []
    runs = {0: initial['ranks']}
    for step, (resume_root, resume_sha) in sorted(resumes.items()):
        resumed = audit_run(resume_root, resume_sha, backend_sha, step)
        require(resumed['training_binding'] == initial['training_binding'], 'DP2 resumed training recipe differs')
        identities = {rank['process_identity_SHA256'] for rank in resumed['ranks']}
        require(len(identities) == 2 and not identities & processes, 'DP2 recovery reused a rank process')
        processes.update(identities)
        runs[step] = resumed['ranks']
        require(Path(resumed['source_checkpoint']).resolve() == (root / f'checkpoint-{step}').resolve(),
                'DP2 recovery opened a different checkpoint directory')
        for rank, (baseline, suffix) in enumerate(zip(initial['streams'], resumed['streams'], strict=True)):
            expected = dict(source_marker_SHA256=initial['markers'][step],
                            restored_rank_state_SHA256=baseline[step-1]['rank_state_sha256'])
            require(resumed['restored'][rank] == expected, 'DP2 actual restored rank boundary differs')
            require(baseline[step:] == suffix, 'DP2 recovered rank trajectory differs')
        bindings.append(dict(start=step, matched_updates_per_rank=25-step, evidence_SHA256=resume_sha,
                             final_marker_SHA256=resumed['markers'][25]))
    slurm = audit_slurm(runs, accounting) if accounting is not None else None
    return dict(case='T09.dp2', backend_SHA256=backend_sha, checked_rank_processes=len(processes),
                continuous_evidence_SHA256=pin, continuous_rank_windows=initial['ranks'],
                continuous_updates_per_second=initial['measurement_updates_per_second'],
                recoveries=bindings, formal_performance=False, checked_external_slurm_accounting=slurm is not None,
                slurm=slurm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-sha', required=True)
    for name in ('continuous', 'resume10', 'resume20'):
        parser.add_argument('--' + name, nargs=2, metavar=('DIRECTORY', 'SEALED_SHA256'), required=True)
    parser.add_argument('--slurm-accounting', nargs=2, metavar=('SACCT_OUTPUT', 'SEALED_SHA256'))
    args = parser.parse_args()
    binding = lambda pair: (Path(pair[0]), pair[1])
    print(json.dumps(audit_recovery(binding(args.continuous), {10: binding(args.resume10), 20: binding(args.resume20)},
                                   args.backend_sha, accounting=binding(args.slurm_accounting) if args.slurm_accounting
                                   else None), allow_nan=False))
