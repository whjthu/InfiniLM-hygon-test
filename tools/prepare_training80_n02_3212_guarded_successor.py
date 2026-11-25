"""Prepare the n02 3212 successor locally. This module performs no transport or dispatch."""

import argparse
import ast
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory
from types import FunctionType


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NODE, PARENT = 'f13r1n02', '1103606'
PREVIOUS, CLEARANCE = '1103606.3212', '1103606.3211'
OLD_REQUEST = 'fb95a1f66e744a34924e3d56a212f1b8'
STEM = 'training80-n02-3212-guarded-standby-renewal-v1'
GPU_TRES = 'cpu=8,gres/dcu=8,mem=16G,node=1'
OLD = ('tools/run_training80_n02_3092_guarded_refill_once.py',
       'fe56212d737913a890589252e405a3b3abdb056b72df68398569a04f6c4016fd')
R25 = ('local-only/read-training80-r25-next-incumbents-once.py',
       '0462a5247bb78a0b4f61095d3be8bf1f66c06d73339043654dbc135f68f4a789')
PUBLIC = ('results/bw1000/2026-11-25/training80-n02-3212-actual-eight-card-counter-growth.json',
          '546a71636a59aba2c7ea9cd283129c860f602dbcd56df7b698670ec343b36a12')
PREP, BUDGET = 1795588674.9674673, 1795589574.9674673
CAP = 45


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(relative, expected):
    path = ROOT / relative
    require(path.resolve(strict=True) == path and path.is_relative_to(ROOT), 'noncanonical evidence path')
    with path.open('rb') as handle:
        raw = handle.read(262145)
    require(0 < len(raw) <= 262144 and digest(raw) == expected, 'pinned source or evidence changed')
    return raw


def load():
    pin(*OLD)
    previous = runpy.run_path(str(ROOT / OLD[0]))
    base, prior_claim, _, previous_claim, *_ = previous['load']()
    pin(*R25)
    read = runpy.run_path(str(ROOT / R25[0]))
    [scope] = read['rows_for']('n02')
    audited = json.loads(pin(*PUBLIC))
    require((audited['step'], audited['request'], audited['verified_positive_growth_DCU'], audited['worker_and_cgroup_verified'])
            == (PREVIOUS, OLD_REQUEST, 8, True) and audited['lifecycle_terminal'] is False
            and scope['previous'] == PREVIOUS and scope['request'] == OLD_REQUEST
            and scope['prep_Beijing'].endswith('14:37:54.967467+08:00')
            and (scope['worker_seconds'], scope['planned_end_epoch']) == (10627, round(BUDGET, 6))
            and previous_claim['request'] == OLD_REQUEST, 'actual n02 incumbent pins differ')
    original = json.loads(pin(*read['N02']))['observed']
    gate = original['original_guard_outcome'][0]
    clear = gate['fresh_zero_GRES_accounting']
    require(gate['request'] == OLD_REQUEST and gate['probe_step'] == CLEARANCE and gate['GPU_srun_attempted'] is True
            and clear[:6] == [CLEARANCE, 'training80-n02-fresh-' + OLD_REQUEST[:12], 'COMPLETED', '0:0', NODE,
                               'cpu=1,mem=256M,node=1'], 'actual .3211 zero-GRES predecessor differs')
    # The original bound n02 collector reported Slurm time in Beijing.
    utc = datetime.fromisoformat(clear[6]).replace(tzinfo=timezone(timedelta(hours=8))).astimezone(timezone.utc)
    clear = clear[:6] + [utc.replace(tzinfo=None).isoformat()]
    pins = [scope['worker_SHA256'], scope['cgroup_SHA256'], scope['expected_SHA256']]
    return previous, base, prior_claim, original, scope, clear, pins


def positive_token(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{32}', value) is not None and int(value, 16) > 0


def check_priority(inference, current_local_authority, programs):
    """Entry point for the owning coordinator's current inference handoff and existing local hold read."""
    require(inference.get('node') == NODE and inference.get('parent') == PARENT
            and inference.get('state') == 'RELEASED_OR_UNRESERVED' and inference.get('authority_verified') is True,
            'inference active, reserved, or unresolved on n02')
    current_local_authority(programs)


def bind_terminal(proof, read_id, request, pins, *, monotonic, wall):
    require(positive_token(read_id) and positive_token(request) and read_id != request and request != OLD_REQUEST,
            'distinct new request and read identities required')
    require(proof['rc'] == 0 and proof['replay_allowed'] is False
            and abs(wall - monotonic - proof['boot_epoch_anchor']) < 2
            and 0 <= monotonic - proof['received_monotonic'] <= CAP and 0 <= proof['roundtrip_s'] <= 25,
            'fresh exact monotonic terminal observation required')
    result = proof['result']
    require(isinstance(result, dict) and result['read_id'] == read_id and result['next_request'] == request
            and result['predecessor'] == PREVIOUS and result['new_Slurm_operations'] == 0
            and result['state'] == 'OLD_ACTUAL_TERMINAL_TARGET_EMPTY' and result['two_views_empty'] is True
            and result['worker_pins'] == pins and result['elapsed_seconds'] >= 10627
            and result['worker_return_epoch'] >= BUDGET
            and re.fullmatch(r'[a-f0-9]{64}', result['worker_terminal_SHA256']),
            'actual natural .3212 worker return and both empty target views required')
    account = result['accounting']
    require(len(account) == 7 and account[:6] == [PREVIOUS, f'training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}',
                                                  'COMPLETED', '0:0', NODE, GPU_TRES]
            and re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}', account[6]),
            'exact current .3212 C0 and End required')
    return result


def _render_programs(material, request, read_id, terminal=None, publication=None):
    """Render pure HOST text only. An absent terminal can render only the readback."""
    previous, base, prior_claim, start, scope, clear, pins = material
    require(positive_token(request) and positive_token(read_id) and request not in (OLD_REQUEST, read_id), 'new binder identity required')
    if terminal is not None:
        require(terminal['next_request'] == request and terminal['read_id'] == read_id and terminal['two_views_empty'] is True
                and isinstance(publication, str) and re.fullmatch(r'[a-f0-9]{40}', publication),
                'rendering a successor requires bound terminal and publication')
    material_for_source = (base, prior_claim, None, dict(request=request, old_worker_start=scope['worker_start_epoch'],
                                                       old_worker_budget_s=scope['worker_seconds']), start, clear, pins)
    render = previous['render']
    private = FunctionType(render.__code__, dict(render.__globals__, STEM=STEM, PREVIOUS=PREVIOUS, OLD_REQUEST=OLD_REQUEST))
    result = private(material_for_source, publication, terminal=terminal, read_id=read_id)
    utc = "import os\nos.environ.update(TZ='UTC', SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')\n"
    for key in ('read_predecessor', 'observe'):
        result[key] = utc + result[key]
        compile(result[key], f'<n02 successor {key}>', 'exec')
    # No launch payload is returned until the coordinator supplies actual validated terminal proof.
    if terminal is None:
        del result['launch']
        return result
    source = result['launch']
    anchor = '            root.mkdir(mode=0o700)'
    require(source.count(anchor) == 1 and source.count("outcome['previous_natural_terminal'] = check_previous(account(PREVIOUS))") == 1
            and source.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)')
                < source.index("outcome['previous_natural_terminal'] = check_previous(account(PREVIOUS))")
                < source.index('outcome["zero_GRES_call_attempted"] = True') < source.index(anchor), 'guard order changed')
    check_map = ("            require(isinstance(raw, dict) and set(raw) == set(PINS)\n"
                 "                    and {name: sha256(data).hexdigest() for name, data in raw.items()} == PINS, 'worker source map drift')\n")
    result['launch'] = utc + source.replace(anchor, check_map + anchor, 1)
    compile(result['launch'], '<n02 successor guarded payload>', 'exec')
    return result


def read_programs(material, request, read_id):
    return _render_programs(material, request, read_id)


def guarded_successor(material, request, read_id, proof, publication, inference, *, monotonic, wall):
    """Consume exact live read bytes and owner priority before generating a guarded host source."""
    expected = read_programs(material, request, read_id)
    require(proof['source_SHA256'] == digest(expected['read_predecessor'].encode()), 'terminal read program hash differs')
    terminal = bind_terminal(proof, read_id, request, material[-1], monotonic=monotonic, wall=wall)
    check_priority(inference, material[1]['current_local_authority'], expected)
    return _render_programs(material, request, read_id, terminal, publication)


def verify_growth(first, second, request, pci):
    """Join independently harvested raw card snapshots from the same numbered worker."""
    keys = ('node', 'parent', 'step', 'request', 'worker_SHA256', 'worker_cgroup_SHA256', 'expected_SHA256',
            'actual_worker_start_epoch', 'worker_budget_seconds')
    require(first['rc'] == second['rc'] == 0 and positive_token(request) and request != OLD_REQUEST, 'usable successor reads required')
    a, b = first['observed'], second['observed']
    step = a['step']
    label = f'training80-dcu-standby-{request[:12]}-{NODE}'
    require((a['node'], a['parent'], a['request']) == (NODE, PARENT, request)
            and re.fullmatch(PARENT + r'\.[0-9]+', step) and step not in (PREVIOUS, CLEARANCE)
            and all(a[k] == b[k] for k in keys)
            and all(re.fullmatch(r'[a-f0-9]{64}', a[k]) for k in keys[4:7])
            and all(point['accounting'] == [step, label, 'RUNNING', '0:0', NODE, GPU_TRES, 'Unknown'] for point in (a, b))
            and len(pci) == len(set(pci)) == 8 and set(a['first']) == set(b['first']) == set(pci), 'successor identity or eight PCI differs')
    delta = {}
    for card in pci:
        old, new = a['first'][card], b['first'][card]
        require(new['epoch'] > old['epoch'] and new['calls'] > old['calls'], 'successor evidence has a nonadvancing PCI')
        delta[card] = new['calls'] - old['calls']
    return dict(step=step, verified_cards=len(delta), counter_delta=delta)


def host_return_fixture(source, scope):
    """Execute the exact rendered predecessor guard with seven real accounting columns."""
    module = ast.parse(source)
    funcs = [obj for obj in module.body if isinstance(obj, ast.FunctionDef)
             and obj.name in ('account', 'check_previous', 'previous_file')]
    code = compile(ast.Module(body=funcs, type_ignores=[]), '<actual n02 HOST return guard>', 'exec')
    label = f'training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}'
    end = '2026-11-25T06:52:56'
    account = [PREVIOUS, label, 'COMPLETED', '0:0', NODE, GPU_TRES, end]
    fmt = 'JobIDRaw,JobName%110,State,ExitCode,NodeList,AllocTRES,End'

    def query(*args):
        require(args[:5] == ('sacct', '-nP', '-j', PREVIOUS, '-o'), 'different original scheduler query')
        if args[5] == fmt:
            return ['|'.join(account)]
        require(args[5] == 'JobIDRaw,ElapsedRaw,End', 'different original terminal elapsed query')
        return [f'{PREVIOUS}|10648|{end}']

    with TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        node = root / ('training80-dcu-standby-' + OLD_REQUEST) / NODE
        node.mkdir(parents=True)
        worker = dict(step=PREVIOUS, node=NODE, worker_pid=123, synthetic=True, epoch=scope['worker_start_epoch'], pci=scope['pci'])
        cgroup = dict(step=PREVIOUS, node=NODE, worker_pid=123, worker_start_ticks=456, same_UID=True, exact_job_and_step_cgroup=True)
        expected = dict(node=NODE, parent=PARENT, request=OLD_REQUEST, label=label, seconds=scope['worker_seconds'], pci=scope['pci'])
        terminal = dict(node=NODE, step=PREVIOUS, state='WORKER_RETURNED', epoch=BUDGET+1)

        def write(name, payload):
            raw = json.dumps(payload).encode()
            (node / f'{name}.json').write_bytes(raw)
            return digest(raw)

        pins = [write(name, obj) for name, obj in (('worker',worker), ('worker-cgroup',cgroup), ('expected',expected))]
        terminal_pin = write('terminal', terminal)
        host = dict(RESULTS=root, PREVIOUS=PREVIOUS, PREVIOUS_LABEL=label, NODE=NODE, PARENT=PARENT, FMT=fmt, GPU_TRES=GPU_TRES,
                    OLD_REQUEST=OLD_REQUEST, OLD_WORKER_PINS=pins, OLD_START=scope['worker_start_epoch'], OLD_SECONDS=scope['worker_seconds'],
                    PCI=scope['pci'], EXPECTED_TERMINAL=dict(accounting=account[:], worker_terminal_SHA256=terminal_pin),
                    query=query, require=require, re=re, json=json, sha256=sha256)
        exec(code, host)
        result = host['check_previous'](host['account'](PREVIOUS))
        require(result['worker_terminal_SHA256'] == terminal_pin and result['elapsed_seconds'] == 10648, 'actual HOST return fixture differs')
        write('terminal', dict(terminal, epoch=BUDGET-1))
        try:
            host['check_previous'](host['account'](PREVIOUS))
        except ValueError as error:
            require(str(error) == 'old exact worker did not return naturally', 'actual HOST returned wrong veto')
        else:
            raise ValueError('actual HOST incorrectly accepted early worker exit')


def offline_check():
    material = load()
    previous, _, _, startup, scope, _, pins = material
    request, read_id = '1' * 32, '2' * 32
    proof = dict(rc=0, replay_allowed=False, boot_epoch_anchor=1000, received_monotonic=77, roundtrip_s=.5,
                 result=dict(read_id=read_id, next_request=request, predecessor=PREVIOUS, new_Slurm_operations=0,
                 state='OLD_ACTUAL_TERMINAL_TARGET_EMPTY', two_views_empty=True, worker_pins=pins, elapsed_seconds=10648,
                 worker_return_epoch=BUDGET + 1, worker_terminal_SHA256='c' * 64,
                 accounting=startup['accounting'][:2] + ['COMPLETED', '0:0', NODE, GPU_TRES, '2026-11-25T06:52:56']))
    read_only = read_programs(material, request, read_id)
    require('launch' not in read_only, 'terminal-missing read exposed a launch payload')
    proof['source_SHA256'] = digest(read_only['read_predecessor'].encode())

    def bound(p):
        return bind_terminal(p, read_id, request, pins, monotonic=80, wall=1080)

    fixture_material = list(material)
    fixture_material[1] = dict(material[1], current_local_authority=lambda cfg:
                               require('launch' not in cfg, 'priority must be checked before rendering a dispatch payload'))
    inference = dict(node=NODE, parent=PARENT, state='RELEASED_OR_UNRESERVED', authority_verified=True)

    def successors(p, ownership=inference):
        return guarded_successor(fixture_material, request, read_id, p, 'd'*40, ownership, monotonic=80, wall=1080)

    result = successors(proof)
    host_return_fixture(result['launch'], scope)
    step = PARENT + '.99999'
    label = f'training80-dcu-standby-{request[:12]}-{NODE}'
    evidence = dict(startup, request=request, step=step, accounting=[step,label,'RUNNING','0:0',NODE,GPU_TRES,'Unknown'])
    first = dict(rc=0, observed=evidence)
    latest = {pci: dict(values, epoch=values['epoch']+1, calls=values['calls']+17) for pci, values in evidence['first'].items()}
    second = dict(rc=0, observed=dict(evidence, first=latest))
    matched = verify_growth(first, second, request, scope['pci'])
    require(matched['verified_cards'] == 8 and set(matched['counter_delta'].values()) == {17}, 'eight-card progress fixture differs')
    stopped = dict(latest, **{scope['pci'][0]: evidence['first'][scope['pci'][0]]})
    negative = [lambda: bound(dict(proof, received_monotonic=34)), lambda: bound(dict(proof, boot_epoch_anchor=1100)),
                lambda: bound(dict(proof, result=dict(proof['result'], two_views_empty=False))),
                lambda: bound(dict(proof, result=dict(proof['result'], worker_return_epoch=BUDGET-1))),
                lambda: bound(dict(proof, result=dict(proof['result'], worker_pins=['a' * 64] * 3))),
                lambda: bind_terminal(proof, read_id, OLD_REQUEST, pins, monotonic=80, wall=1080),
                lambda: successors(dict(proof, source_SHA256='f'*64)),
                lambda: verify_growth(first, dict(second, observed=dict(second['observed'], first=stopped)), request, scope['pci']),
                lambda: verify_growth(first, dict(second, observed=dict(second['observed'], worker_cgroup_SHA256='f'*64)),
                                      request, scope['pci'])]
    for status in ('ACTIVE', 'RESERVED', 'UNKNOWN'):
        negative.append(lambda status=status: successors(proof, dict(inference,state=status)))
    for run_case in negative:
        try:
            run_case()
        except ValueError:
            continue
        raise ValueError('unsafe synthetic history or local priority passed')
    return dict(scope=STEM, state='LOCAL_SOURCE_PREPARED_NO_ACTUAL_TERMINAL', actual_GPU=0, SCNet=0, Git=0, claim_written=False,
                predecessor=PREVIOUS, old_probe=CLEARANCE, exact_worker_seconds=scope['worker_seconds'],
                preparation_epoch=PREP, projected_budget_epoch=BUDGET,
                positive_cases=4, negative_cases=len(negative) + 1, inherited_terminal_RTT=previous['OBSERVE_RTT_SECONDS'],
                read_fixture_SHA256=digest(read_only['read_predecessor'].encode()),
                guarded_fixture_SHA256=digest(result['launch'].encode()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(offline_check(), sort_keys=True, allow_nan=False))
