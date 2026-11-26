"""Prepare one new f12r3n17 standby after independently closing the old CPU probe."""

import ast
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import runpy
import stat
import subprocess
from tempfile import TemporaryDirectory
import textwrap
from types import SimpleNamespace
from uuid import UUID, uuid5


WRAPPER = Path(__file__).resolve()
ROOT = WRAPPER.parents[1]
LOCAL = ROOT/'local-only'
STEM = 'training80-legacy-n17-warning-closed-bf16-v3'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
NODE = 'f12r3n17'
V2 = ('tools/run_training80_legacy_three_fresh_owner_bf16_v2_once.py',
      '16eb4c737c78270bcd89866a38c8feaf38d44471e5a9289b3ee2c97255433fc6')
CLAIM_V2 = ('coordination/tasks/training80-legacy-three-fresh-owner-bf16-v2.json',
            'd338899e57ab281b36d31446a740e1f83db9edf5cfb6b24639b8fb563f0053d2')
WARNING = ('tools/run_training80_r54_three_0113_guarded_successor_once.py',
           '3e72de0c7e52ebdd0eba65e199e934a0fab37a4d6d447c815c0dd275e0f9c941')
RECON = ('local-only/training80-r82-v2-n17-original-zero-gres-warning-terminal-707e7850fb3e4194bd78473b821267df/receipt.json',
         'af5db9e3d8f0c523c6a099740b98ebb7872141412fb59bc773e1843e6e1ae521')
OLD_REQUEST = 'e436284f83ff5eb6b1dcbaa88780ed80'
OLD_SCOPE = f'training80-legacy-three-fresh-owner-bf16-v2-{NODE}-{OLD_REQUEST}'
MARKERS = {
    'intent': '9ca88cd2e53716394dc8835c051c7a9678919518be3ba1adb2b05e6a28c4db5e',
    'send': '0cb66be83adbc26fe5807ebe747723b20f73902bb523cb2b8d559bff088a6807',
    'reply': '9b4b377868b84a6577117b15b8b48611060c32dfa5b9a59c2114a6985b5bb2ab',
}
CPU = ['1194900.1633','training80-r25-fresh-e436284f83ff','COMPLETED','0:0',
       NODE,'cpu=1,mem=256M,node=1','2026-11-26T00:45:58']

PRIVATE_STDERR = r'''
def save_probe_stderr(audit, stderr, outcome):
    limit = 8192
    outcome["probe_stderr_bytes"] = len(stderr)
    outcome["probe_stderr_SHA256"] = sha256(stderr).hexdigest()
    temp = audit / ".fresh-probe-stderr.tmp"
    final = audit / "fresh-probe-stderr.private"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(stderr[:limit])
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, final)
    finally:
        temp.unlink()


def run_fresh_probe(argv, timeout, audit, outcome):
    try:
        observation = run(argv, timeout)
    except subprocess.TimeoutExpired as error:
        save_probe_stderr(audit, error.stderr or b"", outcome)
        raise
    save_probe_stderr(audit, observation.stderr, outcome)
    return observation
'''


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def pin(spec):
    path = ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True) == path, 'noncanonical frozen path')
    with path.open('rb') as stream:
        raw = stream.read(262145)
    require(0 < len(raw) <= 262144 and sha256(raw).hexdigest() == spec[1], 'frozen bytes differ')
    return raw


def proof():
    markers = {
        kind: json.loads(pin((f'local-only/{OLD_SCOPE}-{kind}.json', digest)))
        for kind, digest in MARKERS.items()
    }
    reply = markers['reply']['transport']
    outcome = reply['response']
    packet = json.loads(pin(RECON))
    read = packet['response']
    require(all(value['request'] == OLD_REQUEST for value in markers.values())
            and markers['send']['payload_sha256'] == reply['payload_sha256']
            and reply['authority_verified'] is reply['payload_flushed'] is reply['remote_received'] is True
            and reply['rc'] == 0 and outcome['request'] == OLD_REQUEST and outcome['node'] == NODE
            and outcome['parent'] == '1194900' and outcome['state'] == 'GPU_NOT_SENT'
            and outcome['mutex_acquired'] is outcome['zero_GRES_call_attempted'] is True
            and outcome['GPU_srun_attempted'] is outcome['replay_allowed'] is False
            and outcome['probe_rc'] == 0
            and outcome['probe_stderr_SHA256'] == '819487f21be5e7eeffc86fdd45d5d28f64adbecc596ef82ba176c8a5bc4b6799'
            and outcome['detail_SHA256'] == '65f3a77095fdcab49b34702824d24e5e44299f56107c36f5985aec601edb2c5b',
            'old guarded warning boundary differs')
    require(packet['rc'] == 0 and packet['authority_verified'] is packet['remote_received'] is True
            and read['original_request'] == OLD_REQUEST and read['state'] == 'ORIGINAL_CPU_C0_GPU_ABSENT'
            and read['durable_outcome_matches'] is read['predecessor_exact_C0'] is read['probe_C0_zero_GRES'] is True
            and read['replay_allowed'] is read['physical_observed'] is read['usable_as_launch_receipt'] is False
            and read['original_probe_accounting'] == [CPU] and read['cpu_label_steps'] == [CPU[0]]
            and read['gpu_label_steps'] == read['unexpected_artifacts'] == []
            and read['current_target_steps'] == {'controller': [], 'queue': []}
            and set(read['artifact_SHA256']) == {'intent', 'probe_intent', 'outcome'}
            and all(re.fullmatch('[a-f0-9]{64}', digest) for digest in read['artifact_SHA256'].values()),
            'independent warning CPU closure differs')
    return dict(request=OLD_REQUEST, scope=OLD_SCOPE, CPU=CPU, outcome=outcome,
                outcome_SHA256=read['artifact_SHA256']['outcome'])


def with_private_stderr(launch, unique):
    launch = unique(launch, '\ndef run_guarded():',
                    '\n'+PRIVATE_STDERR+'\n\ndef run_guarded():')
    launch = unique(launch, '            observation = run(["srun",',
                    '            observation = run_fresh_probe(["srun",')
    launch = unique(launch, 'REQUEST, PARENT, NODE, json.dumps(PORTS)], 18)\n',
                    'REQUEST, PARENT, NODE, json.dumps(PORTS)], 18, audit, outcome)\n')
    compile(launch, '<n17 bounded private probe evidence>', 'exec')
    return launch


def stderr_fixture(launch):
    tree = ast.parse(launch)
    defs = [stmt for stmt in tree.body if isinstance(stmt, ast.FunctionDef)
            and stmt.name in {'save_probe_stderr', 'run_fresh_probe'}]
    worker = next(stmt for stmt in tree.body if isinstance(stmt, ast.FunctionDef) and stmt.name == 'run_guarded')
    locked = next(stmt for stmt in ast.walk(worker) if isinstance(stmt, ast.With)
                  and any(isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == 'lock'
                          for item in stmt.items))
    run_at = next(i for i, stmt in enumerate(locked.body) if 'observation = run_fresh_probe(' in ast.unparse(stmt))
    gate_at = next(i for i, stmt in enumerate(locked.body) if 'and (not observation.stderr)' in ast.unparse(stmt))
    require(run_at < gate_at, 'private capture ordering differs')
    code = compile(ast.Module(body=defs, type_ignores=[]), '<actual private stderr functions>', 'exec')
    gate = compile(ast.Module(body=[locked.body[gate_at]], type_ignores=[]), '<actual warning-free gate>', 'exec')
    for data, timeout in ((b'', False), (b'private warning', False), (b'X'*9000, False),
                          (b'captured before timeout', True)):
        with TemporaryDirectory() as folder:
            audit = Path(folder)
            outcome = {}
            observation = SimpleNamespace(returncode=0, stdout=b'{}', stderr=data)
            modes = []

            def run(argv, seconds):
                if timeout:
                    raise subprocess.TimeoutExpired(argv, seconds, stderr=data)
                return observation

            def open_private(path, flags, mode):
                modes.append(mode)
                return os.open(path, flags, mode)

            io = SimpleNamespace(open=open_private, fdopen=os.fdopen, fsync=os.fsync, link=os.link,
                                 O_WRONLY=os.O_WRONLY, O_CREAT=os.O_CREAT, O_EXCL=os.O_EXCL)
            env = dict(os=io, sha256=sha256, subprocess=subprocess, run=run, require=require,
                       observation=observation)
            exec(code, env)
            try:
                seen = env['run_fresh_probe'](['fixture'], 18, audit, outcome)
            except subprocess.TimeoutExpired:
                require(timeout, 'unexpected timeout in fixture')
            else:
                require(not timeout and seen is observation, 'probe result changed')
                try:
                    exec(gate, env)
                except ValueError:
                    require(bool(data), 'empty stderr failed the actual gate')
                else:
                    require(not data, 'nonempty stderr passed the actual gate')
            artifact = audit/'fresh-probe-stderr.private'
            require(artifact.read_bytes() == data[:8192] and list(audit.iterdir()) == [artifact]
                    and outcome == {'probe_stderr_bytes': len(data), 'probe_stderr_SHA256': sha256(data).hexdigest()}
                    and modes == [0o600] and (os.name != 'posix' or stat.S_IMODE(artifact.stat().st_mode) == 0o600),
                    'private raw file, permissions, size cap or public projection differs')
            try:
                env['save_probe_stderr'](audit, b'changed', {})
            except FileExistsError:
                require(artifact.read_bytes() == data[:8192] and list(audit.iterdir()) == [artifact],
                        'existing evidence was overwritten')
            else:
                raise ValueError('existing evidence did not block a second publication')


def entry():
    pin(V2)
    pin(WARNING)
    v2 = runpy.run_path(str(ROOT/V2[0]))
    original = runpy.run_path(str(ROOT/v2['V1'][0]))
    shadow = runpy.run_path(str(ROOT/v2['R45'][0]))
    warning = runpy.run_path(str(ROOT/WARNING[0]))
    clone = v2['clone']
    load_v2, _ = v2['entry']()
    namespace = dict(original, WRAPPER=WRAPPER, STEM=STEM, CLAIM=CLAIM, NODES=(NODE,),
                     __file__=__file__, __doc__=__doc__)

    def load():
        base, reader, budget, ctx, rows, old_claim = load_v2()
        old = rows[NODE]
        require(json.loads(pin(CLAIM_V2)) == old_claim and old['new_request'] == OLD_REQUEST
                and old['cgroup_SHA256'] is None, 'old v2 claim or historical identity differs')
        closure = proof()
        request = uuid5(UUID('be92ecb4-e9ce-4bf9-a289-8c88a7bf7919'), f'{NODE}/{OLD_REQUEST}').hex
        seconds = ctx[1][7](NODE, 10800, request)
        require(request not in {OLD_REQUEST, old['request'], old['previous_v1']['request']}
                and 10500 <= seconds <= 10800, 'new request or actual controller duration differs')
        row = dict(old, new_request=request, renewal_seconds=seconds)
        target = dict(old_claim['targets'][NODE], new_request=request, worker_seconds=seconds,
                      prior_v2_request=OLD_REQUEST, prior_v2_CPU=CPU[0],
                      prior_v2_independent_reconciliation_SHA256=RECON[1],
                      prior_v2_gpu_srun_attempted=False,
                      prior_v2_durable_outcome_SHA256=closure['outcome_SHA256'])
        claim = dict(old_claim, scope=STEM, launcher=WRAPPER.relative_to(ROOT).as_posix(), targets={NODE: target},
                     source_pins=dict(old_claim['source_pins'], **dict((V2, CLAIM_V2, WARNING))),
                     v2_replay_allowed=False, warning_policy='strict empty fresh probe stderr, no allowed digest',
                     warning_capture=dict(
                         cluster_file='fresh-probe-stderr.private', file_mode='0600', max_bytes=8192,
                         original_stderr_prefix_if_over_limit=True,
                         public_fields=['probe_stderr_bytes','probe_stderr_SHA256']))
        return base, reader, budget, ctx, {NODE: row}, claim

    def marker_set(names):
        expected = {f'{OLD_SCOPE}-{kind}.json' for kind in MARKERS}
        require(set(names) == expected, 'new or missing singleton artifact; reconcile without dispatch')

    def booking(base, ordinary, material, row):
        v2['no_v1_marker'](row['previous_v1']['scope'])
        proof()
        exact = base['exact_no_other_claims'](ordinary, material, row)
        expected = {(v['parent'], v['ancestor'], v['claim'], v['wave'], v['group']) for v in row['ledger_rows']}
        require(len(exact) == 2 and set(exact) == expected, 'exact old booking history has an extra writer')
        journal = LOCAL/'training80-1738-ordinary-renewal-dispatch'
        require(journal.is_dir() and not journal.is_symlink(), 'unverified old ordinary journal')
        for step in row['journal_steps']:
            require(not any(os.path.lexists(journal/f'{NODE}-{step}-{slot}.json') for slot in range(3)),
                    'old ordinary slot still has a ticket')
        select = re.compile(r'-(intent|send|reply)\.json$').search
        marker_set(p.name for p in LOCAL.glob(f'training80-*-{NODE}-*.json') if select(p.name))

    def runtime(base, reader, budget, ctx, revision):
        inherited = clone(original['runtime'], namespace, local_booking=booking)
        source, read, old_render, dispatch, observe, old_publication = inherited(base, reader, budget, ctx, revision)
        closure = proof()

        def render(ctx, row, account, terminal, commit):
            program = old_render(ctx, row, account, terminal, commit)
            launch = shadow['guarded_shadow'](program['launch'], row['previous_v1'], NODE, base['unique'])
            launch = warning['with_locked_warning'](launch, closure, NODE, base['unique'])
            program['launch'] = with_private_stderr(launch, base['unique'])
            return program

        def publication(ctx, commit):
            old_publication(ctx, commit)
            _, publisher = ctx[3][-2]['modules']()
            for spec in (v2['V1'], v2['R45'], V2, CLAIM_V2, WARNING):
                require(publisher['git']('cat-file', 'blob', commit+':'+spec[0]) == pin(spec),
                        'published inherited warning or custody bytes differ')

        return (source, read, render, clone(dispatch, publication=publication, render=render),
                clone(observe, render=render), publication)

    def check_warning(launch, row):
        shadow['shadow_fixture'](launch, row, row['previous_v1'])
        warning['warning_fixture'](launch, proof(), row)
        require('and not observation.stderr and' in launch, 'fresh stderr gate differs')
        stderr_fixture(launch)
        valid = {f'{OLD_SCOPE}-{kind}.json' for kind in MARKERS}
        marker_set(valid)
        for bad in (set(), valid | {STEM+'-intent.json'}):
            try:
                marker_set(bad)
            except ValueError:
                pass
            else:
                raise ValueError('changed marker set accepted')

    namespace.update(load=load, runtime=runtime, local_booking=booking, check_warning=check_warning)
    cli = textwrap.dedent(inspect.getsource(original['main']))
    anchor = '            check_locked_gates(launch,row)\n'
    require(cli.count(anchor) == 1, 'inherited CLI fixture boundary differs')
    cli = cli.replace(anchor, anchor+'            check_warning(launch,row)\n', 1)
    cli = cli.replace("launch.index('gpu-send-intent.json')",
                      "launch.index('save(audit / \"gpu-send-intent.json\"')")
    cli = cli.replace('LOCAL_BOOKINGS_CLEAR_3', 'LOCAL_BOOKINGS_CLEAR_N17')
    cli = cli.replace("targets=result,old_writer_and_boot_vetoes=12,",
                      "targets=result,private_stderr_fixtures=4,replacement_blocked=True,")
    exec(compile(cli, '<n17 warning closed CLI>', 'exec'), namespace)
    return load, namespace['main']


if __name__ == '__main__':
    _, main = entry()
    main()
