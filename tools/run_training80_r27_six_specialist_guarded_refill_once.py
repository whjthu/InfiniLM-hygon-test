"""Resume one of six exact returned specialist BF16 standby workers."""

import argparse
import ast
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
from tempfile import TemporaryDirectory
import time
from types import FunctionType


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r27-six-specialist-guarded-bf16-v1'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
BASE = ('tools/run_training80_prebridge_five_guarded_refill_once.py',
        '7b88d30a4af7eff4f25fbcc663de674af328a03de74e364c517752bd599d306c')
BASE_CLAIM = ('coordination/tasks/training80-prebridge-five-guarded-bf16-v1.json',
              'c4c18f644e6ebc2a99ea0e19d2261a9cbbe47efe743e79dd5ae8f79fea4ce974')
T07_CLAIM = ('coordination/tasks/training80-t07-3081-guarded-bf16-v1.json',
             'a13e131d2823a85626092149dd05dfe0767f32fe80a100ea787c0ef6ec40dc04')
ORDINARY = ('tools/run_training80_ordinary58_1140_refill_once.py',
            '9cdf04cf946058c837ff9b5101f9eff56e969c595ae3b2397fe064834f2c4ba2')
TERMINAL_READ = 'ce1972e8558e4e8297cf0a2df5ccf201'
TERMINAL = (f'local-only/training80-r27-six-gap-exact-original-terminal-{TERMINAL_READ}.json',
            '3698389d84c704b8c9ded30049e90309f689c09dc08a545fa1955d5a805ddf10')
CLEARANCE = ('local-only/training80-r27-six-terminal-predecessor-independent-zero-gres-utc.json',
             '9387f34f76773cf9b3e702a508f616d3c55093cc6e914451dbd93835ef28b99d')
PARENT = '1103606'
GPU_TRES = 'cpu=8,gres/dcu=8,mem=16G,node=1'
CPU_TRES = 'cpu=1,mem=256M,node=1'
SACCT_COLUMNS = 'JobIDRaw,JobName%110,State,ExitCode,ElapsedRaw,NodeList%60,AllocTRES%150,Start,End'
OLD_FILES = ('worker', 'worker-cgroup', 'expected')
HASH_PATTERN = r'[a-f0-9]{64}'
SPEC = {
    'f13r1n10': ('3179', '3178', '0ab8e682ab29495581ea18b168cf7898', '3858a5ecc4594696be012360d386c2f7',
                '310f7625a48c0d38c401b750258b4450f1cc8ebbccaef8bfe7f0660c7e8c7489'),
    'f13r1n11': ('3181', '3180', '92bd0f5e82394a02afffa1020679e2dc', 'ef0485b209d647ab902cfb5e4558fb49',
                '365aca86f3a1ff89deabbb141eff311755312e957e333dc626bc0a68f2d8ae79'),
    'f13r1n17': ('3183', '3182', 'af7fe81583494929b68838b6057f75ce', 'aac14e185d1e46819f4dfce92cba8bd8',
                '0728124b6fc0d1d952df4b571b627e343b753bc852ac40c63c7193004ee0895c'),
    'f13r2n03': ('3193', '3192', 'ceb75ea0b3ce449795780445e743c99d', 'f88b267bfa7d49c58866b4650abc849d',
                'e29bbd59aeb1e19532efd981e5d5a082175555f2cc0af2841b96e03128da98f1'),
    'f13r2n07': ('3203', '3202', 'df982d1cc29d4ab39515b752eea2c164', 'c2acb1b023c34240af5826d9db100064',
                'd5c4a4cb0c570d9b5ba780a45bc67ba25d119b5022a14ccef8d26e5d3e5c9f68'),
    'f13r2n10': ('3195', '3194', 'bdda5fbee0b54935b262113fba4f897e', 'df47b5ea06e34ca7959ac828d8c90de0',
                'c5ac31eb6f910fda2308fa7d7dbc25d17b12ecb27a7427c336ad7fb3ae50e5e2'),
}


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(spec, limit=262144):
    relative, expected = spec
    path = ROOT / relative
    require(path.resolve(strict=True) == path and path.is_relative_to(ROOT), 'noncanonical pinned file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and digest(raw) == expected, 'pinned bytes changed')
    return raw


def numbered(number):
    return f'{PARENT}.{number}'


def reply_stem(node):
    if node == 'f13r2n07':
        return 'training80-t07-3081-guarded-bf16-v1'
    return ('training80-prebridge-two-r2-guarded-bf16-v2' if node in ('f13r2n03', 'f13r2n10')
            else 'training80-prebridge-five-guarded-bf16-v1')


def previous_clearance(node):
    _, zero, old, _, expected_sha = SPEC[node]
    name = f'local-only/{reply_stem(node)}-{node}-{old}-reply.json'
    reply = json.loads(pin((name, expected_sha)))
    transfer = reply['transport']
    response = transfer['response']
    row = response['fresh_zero_GRES_accounting']
    expected = [numbered(zero), 'training80-g4-fresh-' + old[:12], 'COMPLETED', '0:0', node, CPU_TRES]
    require(reply['node'] == response['node'] == node and reply['request'] == transfer['request'] == response['request'] == old
            and transfer['rc'] == 0 and transfer['remote_outcome'] == 'RESPONSE_OBSERVED'
            and transfer['authority_verified'] is transfer['remote_received'] is True
            and response['state'] == 'SRUN_PROCESS_STARTED_AWAIT_SCHEDULER_ACCOUNTING'
            and response['zero_GRES_call_attempted'] is response['GPU_srun_attempted'] is response['mutex_acquired'] is True
            and response['probe_step'] == numbered(zero) and len(row) == 7 and row[:6] == expected,
            'historical exact zero-GRES receipt differs')
    zone = timezone.utc if node == 'f13r2n07' else timezone(timedelta(hours=8))
    end = datetime.fromisoformat(row[6]).replace(tzinfo=zone).astimezone(timezone.utc).replace(tzinfo=None).isoformat()
    return row[:6] + [end]


def clearance_rows():
    proof = json.loads(pin(CLEARANCE))
    data = proof['body']
    accounting = data['rows']
    require(proof['rc'] == 0 and data['new_slurm'] == 0 and data['accounting_columns'] == SACCT_COLUMNS
            and set(accounting) == {numbered(spec[1]) for spec in SPEC.values()}, 'six independent zero-GRES rows differ')
    rows = {}
    for node, spec in SPEC.items():
        row = accounting[numbered(spec[1])]
        require(len(row) == 9 and row[:4] + row[5:7] + row[8:] == previous_clearance(node)
                and row[4].isdecimal() and datetime.fromisoformat(row[7]) <= datetime.fromisoformat(row[8]),
                'direct prior zero-GRES receipt and independent UTC accounting disagree')
        rows[node] = row
    return rows


def terminal_rows():
    proof = json.loads(pin(TERMINAL))
    read = proof['observed']
    result = read['result']
    require(proof['rc'] == 0 and proof['replay_allowed'] is False and proof['read'] == read['read'] == TERMINAL_READ
            and read['accounting_columns'] == SACCT_COLUMNS and read['Slurm_mutations'] == 0 and read['device_admission'] is False
            and set(result) == set(SPEC), 'original exact six-node terminal evidence differs')
    growth = {}
    for name, sha in proof['original_inputs'].items():
        raw = json.loads(pin(('local-only/' + name, sha)))
        require(raw['rc'] == 0 and raw['new_Slurm_operations'] == 0, 'original independent growth evidence differs')
        for node, point in raw['result']['nodes'].items():
            require(node not in growth, 'duplicate specialist identity in original evidence')
            growth[node] = point
    require(set(growth) == set(result), 'original independent growth set differs')
    for node, row in result.items():
        predecessor, _, old, _, _ = SPEC[node]
        step = numbered(predecessor)
        original = growth[node]
        account = row['accounting']
        require(row['step'] == original['step'] == step and row['request'] == original['request'] == old
                and original['growth_cards'] == 8 and original['worker_start_ticks'] > 0
                and row['file_pins'] == original['file_pins'] and len(row['file_pins']) == len(OLD_FILES)
                and all(re.fullmatch(HASH_PATTERN, value) for value in [*row['file_pins'], row['original_return_SHA256']])
                and len(set(row['original_PCI'])) == 8 and set(row['original_PCI']) == set(original['cards'])
                and row['controller'] == row['queue'] == [] and row['return_validated'] is True
                and len(account) == 9 and account[:4] == [step, 'training80-dcu-standby-' + old[:12] + '-' + node, 'COMPLETED', '0:0']
                and account[5:7] == [node, GPU_TRES] and 10500 <= row['worker_seconds'] <= int(account[4]) <= 11100
                and row['worker_start_epoch'] + row['worker_seconds'] <= row['return_epoch']
                and datetime.fromisoformat(account[7]) <= datetime.fromisoformat(account[8]),
                'exact C0 worker/cgroup/PCI return or dual views differ')
    return result


def load():
    pin(BASE)
    base = runpy.run_path(str(ROOT / BASE[0]))
    config = base['prepared'](json.loads(pin(BASE_CLAIM)))
    pin(ORDINARY)
    ordinary = runpy.run_path(str(ROOT / ORDINARY[0]))['plan']()[-1]
    active_roster = {row['node'] for group in ordinary for row in group}
    require(len(active_roster) == 58 and not (set(SPEC) | {'f13r1n02'}) & active_roster,
            'specialist or n02 overlaps frozen 58-node ordinary roster')
    t07 = json.loads(pin(T07_CLAIM))
    require((t07['node'], t07['request'], t07['previous_gpu_step'], t07['maximum_invocations'], t07['replay_allowed'])
            == ('f13r2n07', SPEC['f13r2n07'][2], '1103606.3081', 1, False), 'original T07 dedicated scope changed')
    raw = base['pin'](CLAIM)
    claim = json.loads(raw)
    prior = terminal_rows()
    clearance = clearance_rows()
    for node, (old, zero, old_request, request, reply_pin) in SPEC.items():
        require(claim['targets'][node] == dict(original_GPU=numbered(old), original_zero_GRES=numbered(zero),
                original_request=old_request, original_reply_SHA256=reply_pin, request=request), 'fixed singleton claim differs')
    require(set(claim['targets']) == set(SPEC) and len({detail['request'] for detail in claim['targets'].values()}) == 6
            and (claim['scope'], claim['state'], claim['parent'], claim['GPU_per_selected_node'], claim['max_invocations_per_node'],
                 claim['terminal_proof_SHA256'], claim['clearance_proof_SHA256'], claim['ordinary_roster_count'],
                 claim['excluded_n02'], claim['replay_allowed'])
            == (STEM, 'ORIGINAL_TERMINALS_BOUND_FRESH_DEVICE_ADMISSION_PENDING', PARENT, 8, 1, TERMINAL[1], CLEARANCE[1],
                58, 'f13r1n02', False),
            'fixed six-target scope differs')
    legacy_steps = {node: status[0] for node, status in base['FIVE'].items()}
    legacy_steps[t07['node']] = t07['previous_gpu_step']
    return base, config, claim, digest(raw), prior, clearance, legacy_steps


def require_no_slots(journal, node, step, slots):
    require(journal.resolve(strict=True) == journal and journal.is_dir(), 'original reservation directory unavailable')
    for slot in slots:
        path = journal / f'{node}-{step}-{slot}.json'
        try:
            os.lstat(path)
        except FileNotFoundError:
            continue
        raise ValueError('original or current target reservation remains, requires reconciliation')


def bound(base, name, node, spec):
    function = base[name]
    paths = FunctionType(base['paths'].__code__, dict(base['paths'].__globals__, STEM=STEM))
    targets = {node: (spec['original_GPU'], spec['original_zero_GRES'])}
    return FunctionType(function.__code__, dict(function.__globals__, STEM=STEM, FIVE=targets, paths=paths))


def render(base, config, claim, prior, clearance, node, publication):
    spec = claim['targets'][node]
    gpu = prior[node]['accounting']
    config[3][node] = (gpu, clearance[node], sorted(prior[node]['original_PCI']))
    output = bound(base, 'render', node, spec)(node, claim, config, publication)
    pins = dict(zip(OLD_FILES, prior[node]['file_pins'], strict=True), terminal=prior[node]['original_return_SHA256'])
    directory = 'training80-dcu-standby-' + spec['original_request']
    anchor = '            check_previous(account(PREVIOUS))'
    allow = '({OBSERVER_STEP, own} if own else {OBSERVER_STEP})'
    source = output['launch']
    require(source.count(anchor) == source.count(allow) == 1
            and source.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)') < source.index(anchor)
            < source.index('outcome["zero_GRES_call_attempted"] = True'), 'shared locked gate or order changed')
    gate = (f'            for member, expected_sha in {pins!r}.items():\n'
            f"                old_file = RESULTS / {directory!r} / NODE / (member + '.json')\n"
            "                require(old_file.is_file() and old_file.resolve(strict=True) == old_file, 'original artifact path changed')\n"
            "                with old_file.open('rb') as handle:\n"
            "                    raw_pin = handle.read(8193)\n"
            "                require(0 < len(raw_pin) <= 8192 and sha256(raw_pin).hexdigest() == expected_sha, 'old terminal artifact differs')\n")
    prefix = "import os\nos.environ.update(TZ='UTC', SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')\n"
    output['launch'] = prefix + source.replace(anchor, gate + anchor, 1).replace(allow, '({own} if own else set())', 1)
    output['observe'] = prefix + output['observe']
    for key in ('launch', 'observe'):
        compile(output[key], f'<R27 guarded {node} {key}>', 'exec')
    return output


def offline_fixture(source, node, old_gpu):
    module = ast.parse(source)
    eligible = {'account', 'check_previous', 'check_first_clearance'}
    constants = {'FMT', 'PREVIOUS', 'PREVIOUS_LABEL', 'PREVIOUS_END', 'CLEAR_STEP', 'CLEAR_LABEL', 'CLEAR_END', 'NODE', 'ZERO'}
    selected = [obj for obj in module.body if (isinstance(obj, ast.FunctionDef) and obj.name in eligible)
                or (isinstance(obj, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in obj.targets))]
    cpu = previous_clearance(node)
    gpu = old_gpu[:4] + old_gpu[5:7] + old_gpu[8:]
    fixed = {gpu[0]: gpu, cpu[0]: cpu}

    def query(*argv):
        require(argv[:3] == ('sacct', '-nP', '-j') and argv[4:] == ('-o', 'JobIDRaw,JobName%110,State,ExitCode,NodeList,AllocTRES,End'),
                'observed Slurm accounting arity changed')
        return ['|'.join(fixed[argv[3]])]

    scope = dict(require=require, query=query)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<exact seven-column guard fixture>', 'exec'), scope)
    for step, action in ((gpu[0], 'check_previous'), (cpu[0], 'check_first_clearance')):
        scope[action](scope['account'](step))
        terminal = fixed[step][-1]
        fixed[step][-1] = '2026-11-25T00:00:00'
        try:
            scope[action](scope['account'](step))
        except ValueError:
            fixed[step][-1] = terminal
        else:
            raise ValueError('rendered guard accepted a different End')


def offline_staging_fixture(source, standby):
    module = ast.parse(source)
    guard = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'run_guarded')
    locked = next(node for node in ast.walk(guard) if isinstance(node, ast.With)
                  and any(isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == 'lock' for item in node.items))
    original = next(node for node in locked.body if isinstance(node, ast.For) and ast.unparse(node.target) == '(member, expected_sha)')
    stage = next(index for index, node in enumerate(locked.body) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'root.mkdir')
    payload = locked.body[stage:stage + 3]
    require(isinstance(payload[-1], ast.With) and 'stream.write(raw[\'worker.py\'])' in ast.unparse(payload[-1]),
            'frozen worker staging block differs')
    write = next(node for node in ast.parse(standby['controller.py']).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'write')
    writer = dict(os=os, json=json)
    exec(compile(ast.Module(body=[write], type_ignores=[]), '<frozen controller write>', 'exec'), writer)
    pins = {name: digest(name.encode()) for name in (*OLD_FILES, 'terminal')}
    original.iter.func.value = ast.parse(repr(pins), mode='eval').body

    def run_fixture(mode):
        with TemporaryDirectory() as temporary:
            results = Path(temporary).resolve()
            node, request = 'f13r1n10', 'a' * 32
            prior_path = next(obj.value for obj in ast.walk(original) if isinstance(obj, ast.Constant)
                              and isinstance(obj.value, str) and obj.value.startswith('training80-dcu-standby-'))
            directory = results / prior_path / node
            directory.mkdir(parents=True)
            for name in pins:
                (directory / f'{name}.json').write_bytes(b'drift' if mode == 'sha_drift' and name == 'terminal' else name.encode())
            root = results / 'new-worker'
            raw = dict(standby)
            loop = original
            if mode == 'legacy_shadow':
                loop = ast.parse(ast.unparse(original).replace('raw_pin', 'raw')).body[0]
            test = ast.Module(body=[loop, *payload], type_ignores=[])
            scope = dict(RESULTS=results, NODE=node, REQUEST=request, sha256=sha256, require=require, root=root,
                         controller={'write': writer['write']}, PUBLICATION=None, outcome={'probe_stdout_SHA256': 'b' * 64},
                         time=time, CLEAR_STEP=numbered('3178'), step=numbered('99999'), raw=raw)
            try:
                exec(compile(ast.fix_missing_locations(test), '<actual locked pin and staging fixture>', 'exec'), scope)
            except (ValueError, TypeError) as error:
                if mode == 'sha_drift':
                    require(type(error) is ValueError and str(error) == 'old terminal artifact differs' and not root.exists(),
                            'drift was not rejected before staging')
                elif mode == 'legacy_shadow':
                    require(type(error) is TypeError and (root / 'worker.py').stat().st_size == 0,
                            'fixture did not detect the previous source-map shadow')
                else:
                    raise
            else:
                require(mode == 'positive' and scope['raw'] is raw and (root / 'worker.py').read_bytes() == standby['worker.py'],
                        'rendered pin gate did not preserve the staging source map')

    for mode in ('positive', 'sha_drift', 'legacy_shadow'):
        run_fixture(mode)


def verify_publication(base, config, revision):
    base['verify_publication'](revision, config)
    git = ['git', '--no-optional-locks', '-c', 'safe.directory=' + ROOT.as_posix()]
    paths = [Path(__file__).resolve(), CLAIM, ROOT / T07_CLAIM[0], ROOT / ORDINARY[0]]
    for path in paths:
        ref = revision + ':' + path.relative_to(ROOT).as_posix()
        published = subprocess.check_output([*git, 'cat-file', 'blob', ref], cwd=ROOT, stderr=subprocess.PIPE, timeout=10)
        require(published == base['pin'](path), 'published specialist launcher, claim or authority bytes differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original', action='store_true')
    parser.add_argument('--node', choices=SPEC)
    args = parser.parse_args()
    require(args.check or args.node, 'each live invocation must select exactly one specialist')
    base, config, claim, claim_pin, prior, clearance, legacy = load()
    selected = [args.node] if args.node else list(SPEC)
    reports = []
    for node in selected:
        spec = claim['targets'][node]
        program = render(base, config, claim, prior, clearance, node, args.publication)
        if args.observe_original:
            reports.append(bound(base, 'observe', node, spec)(node, program, config))
            continue
        require_no_slots(base['JOURNAL'], node, legacy[node], (1, 2))
        require_no_slots(base['JOURNAL'], node, spec['original_GPU'], (0, 1, 2))
        offline_fixture(program['launch'], node, prior[node]['accounting'])
        offline_staging_fixture(program['launch'], config[6])
        if args.check:
            reports.append(dict(node=node, prior=spec['original_GPU'], clear=spec['original_zero_GRES'], request=spec['request'],
                                worker_seconds=program['worker_seconds'], launch_SHA256=digest(program['launch'].encode()),
                                account_fixtures=4, locked_staging_fixtures=3))
        else:
            verify_publication(base, config, args.publication)
            ledger_nodes = base['live_authority'](config)
            require_no_slots(base['JOURNAL'], node, legacy[node], (1, 2))
            require_no_slots(base['JOURNAL'], node, spec['original_GPU'], (0, 1, 2))
            reports.append(bound(base, 'dispatch', node, spec)(node, program, args.publication, claim_pin, config, ledger_nodes))
    print(json.dumps(dict(claim_SHA256=claim_pin, reports=reports, replay_allowed=False), sort_keys=True))


if __name__ == '__main__':
    main()
