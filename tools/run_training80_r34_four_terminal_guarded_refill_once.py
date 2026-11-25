"""Single-node guarded standby successors for four independently returned workers."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
from types import FunctionType
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r34-four-terminal-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r34_four_terminal_guarded_refill_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
DRAFT = LOCAL / f'{STEM}.draft.json'
TERMINAL = ('training80-r34-four-new-terminal-owned-workers-b6b49f6339ac4f4a9f0b0ed49b4d78b7.json',
            '102ea6aee031e2ca96d2a8f492e0896746a3afcb121a9530b78e6fd9c0d4278e')
R27 = ('tools/run_training80_r27_six_specialist_guarded_refill_once.py',
       '9f941399241fb83d968217e0e0e13df0683a0464bd169bd3c8daa6f787210a18')
O01 = ('tools/run_training80_o01_3087_guarded_refill_once.py',
       '97326a1adaee29a89966bf13676d7c6d7476e274fe5be43c41c0b854679468b2')
NODE_O01 = 'f12r2n11'
PARENT = '1103606'
NODES = {
    NODE_O01: ('3307', '3306', 'cce84178ab4a44c2bda5464fcf089526',
               '67d8f2a8f0441dbd529b52eb1b8a8add0dba10f9c3f31060ee871ae2f7f56e8c', 10755),
    'f13r1n10': ('3297', '3296', '3858a5ecc4594696be012360d386c2f7',
                'fc7c526875955e20864c177329f2d14ad17d17ae157da2894d693aff852626de', 10530),
    'f13r1n11': ('3299', '3298', 'ef0485b209d647ab902cfb5e4558fb49',
                '17438315ccc2b5c8fca5aee54c7a07cf36d755779bb20d88801665c01f2a69c4', 10605),
    'f13r2n10': ('3305', '3304', 'df47b5ea06e34ca7959ac828d8c90de0',
                'e7339d11f72b27ddeab9824cf924fafbf4e9fdaef1510186d1d4d1e106b6af46', 10680),
}


def digest(raw):
    return sha256(raw).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, expected=None):
    require(path.resolve(strict=True) == path, 'frozen evidence path differs')
    with path.open('rb') as stream:
        raw = stream.read(262145)
    require(0 < len(raw) <= 262144 and (expected is None or digest(raw) == expected), 'frozen evidence bytes differ')
    return raw


def load():
    path = ROOT / R27[0]
    pinned(path, R27[1])
    r27 = runpy.run_path(str(path))
    base, config, _, _, legacy_predecessors, _, legacy_ordinary = r27['load']()
    pinned(ROOT / O01[0], O01[1])
    o01 = runpy.run_path(str(ROOT / O01[0]))
    old_o01 = o01['load']()
    protected = o01['terminal_receipt'](old_o01, o01['OLD_CHECK']['read_id'], o01['OLD_CHECK']['receipt_SHA256'])[0][2]
    proof = json.loads(pinned(LOCAL / TERMINAL[0], TERMINAL[1]))
    view = proof['observed']
    require(proof['rc'] == 0 and view['slurm_mutations'] == view['physical_admissions'] == 0
            and len(view['targets']) == len(NODES) and {row['node'] for row in view['targets']} == set(NODES),
            'exact independently observed cohort differs')
    targets, predecessors, clear = {}, {}, {}
    budgets = {}
    for row in view['targets']:
        node = row['node']
        gpu, cpu, old_request, reply_sha, seconds = NODES[node]
        request = uuid5(UUID('ca38fd70-a538-4c6d-b306-1fc6560f4432'), f'{TERMINAL[1]}/{node}/{gpu}').hex
        previous, clean = f'{PARENT}.{gpu}', f'{PARENT}.{cpu}'
        family = 'training80-o01-3087-guarded-bf16-v3' if node == NODE_O01 else r27['STEM']
        reply = json.loads(pinned(LOCAL / f'{family}-{node}-{old_request}-reply.json', reply_sha))
        transport = reply['transport']
        actual = transport['response']
        actual_cpu = actual['fresh_zero_GRES_accounting']
        require(transport['rc'] == 0 and transport['remote_outcome'] == 'RESPONSE_OBSERVED'
                and actual['GPU_srun_attempted'] is actual['mutex_acquired'] is True
                and reply['request'] == transport['request'] == actual['request'] == row['original_request'] == old_request
                and actual['node'] == node and actual['parent'] == PARENT
                and actual['probe_step'] == clean == actual_cpu[0]
                and actual_cpu[:6] == [clean, 'training80-g4-fresh-' + old_request[:12], 'COMPLETED', '0:0', node, 'cpu=1,mem=256M,node=1']
                and actual['probe_stderr_SHA256'] == digest(b'') and actual['receipt']['worker_seconds'] == row['worker_seconds'],
                'original authorized send or zero-GRES terminal differs')
        [acct] = row['accounting']
        require(len(acct) == 9 and acct[:4] == [previous, 'training80-dcu-standby-' + old_request[:12] + '-' + node, 'COMPLETED', '0:0']
                and acct[5:7] == [node, 'cpu=8,gres/dcu=8,mem=16G,node=1'] and row['step'] == previous
                and int(acct[4]) >= row['worker_seconds'] and acct[7] < acct[8]
                and row['worker_returned'] is True and row['worker_return_UTC'] >= row['worker_start_UTC'] + row['worker_seconds']
                and row['scontrol_numeric'] == row['squeue_numeric'] == [] and row['worker_PID'] > 0
                and row['original_parent_parent_context'] == 'PARENT_REMAINS_ALLOCATED',
                'original GPU accounting or worker natural return differs')
        pins = [row[k] for k in ('worker_SHA256', 'worker_cgroup_SHA256', 'expected_SHA256', 'terminal_SHA256')]
        pci = sorted(set(row['original_8_PCI']))
        require(len(pci) == 8 and all(re.fullmatch(r'[a-f0-9]{4}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7]', x) for x in pci)
                and all(re.fullmatch(r'[a-f0-9]{64}', value) for value in pins), 'old worker fingerprints differ')
        targets[node] = dict(request=request, original_GPU=previous, original_zero_GRES=clean, original_request=old_request,
                             original_reply_SHA256=reply_sha, worker_seconds=seconds)
        predecessors[node] = dict(accounting=acct, file_pins=pins[:3], original_PCI=pci, original_return_SHA256=pins[3])
        # The original response supplied terminal End, with no Start or Elapsed.
        clear[node] = actual_cpu[:4] + [None] + actual_cpu[4:6] + [None, actual_cpu[6]]
        budgets[node] = seconds
    def budget(node, limit, request):
        require(limit == 10800 and request == targets[node]['request'] and 10500 <= budgets[node] <= limit, 'new bounded workload changed')
        return budgets[node]
    config = (*config[:7], budget)
    claim = dict(scope=STEM, parent=PARENT, targets=targets, GPU_per_node=8, max_invocations_per_node=1,
                 terminal_proof_SHA256=TERMINAL[1], original_ordinary_and_O01_fences_preserved=True,
                 requires_fresh_owned_zero_GRES=True, replay_allowed=False, launcher=WRAPPER.relative_to(ROOT).as_posix())
    return r27, base, config, claim, predecessors, clear, legacy_predecessors, legacy_ordinary, o01['PROTECTIONS'], protected


def rendered(material, node, publication):
    r27, base, conf, claim, prior, clear, _, _, protections, protected = material
    inherited = r27['bound']
    bound = FunctionType(inherited.__code__, dict(inherited.__globals__, STEM=STEM))
    render = FunctionType(r27['render'].__code__, dict(r27['render'].__globals__, bound=bound))
    guard = render(base, conf, claim, prior, clear, node, publication)
    if node == NODE_O01:
        source = guard['launch']
        anchor = '            check_previous(account(PREVIOUS))'
        require(source.count(anchor) == 1, 'O01 locked gate location differs')
        old_accounting = protected[:4] + protected[5:7] + protected[8:]
        gate = (
            f"            for relative, expected in {protections!r}.items():\n"
            "                original_o01 = RESULTS / relative\n"
            "                if expected is None:\n"
            "                    require(not os.path.lexists(original_o01), 'original O01 released marker changed')\n"
            "                    continue\n"
            "                require(original_o01.resolve(strict=True)==original_o01 and original_o01.stat().st_uid==os.getuid(), 'original O01 grant or release identity differs')\n"
            "                with original_o01.open('rb') as old_o01_handle:\n"
            "                    original_o01_bytes = old_o01_handle.read(8193)\n"
            "                require(0<len(original_o01_bytes)<=8192 and sha256(original_o01_bytes).hexdigest()==expected, 'original O01 grant or release changed')\n"
            f"            require(account('1103606.2424') == {old_accounting!r}, 'protected original O01 zero-GRES terminal changed')\n")
        guard['launch'] = source.replace(anchor, gate + anchor, 1)
        compile(guard['launch'], '<four terminal original O01 protection>', 'exec')
    return guard, bound


def check_o01_account_gate(source, protected):
    tree = ast.parse(source)
    candidates = [
        line for line in ast.walk(tree)
        if isinstance(line, ast.Expr)
        and isinstance(line.value, ast.Call)
        and "account('1103606.2424')" in ast.unparse(line)
        and 'protected original O01 zero-GRES terminal changed' in ast.unparse(line)
    ]
    [candidate] = candidates
    predicate = compile(ast.Module(body=[candidate], type_ignores=[]), '<actual O01 protected accounting gate>', 'exec')
    row = protected[:4] + protected[5:7] + protected[8:]
    exec(predicate, dict(account=lambda step: row, require=require))
    try:
        exec(predicate, dict(account=lambda step: row[:-1] + ['changed End'], require=require))
    except ValueError as error:
        require(str(error) == 'protected original O01 zero-GRES terminal changed', 'wrong O01 protected veto')
    else:
        raise ValueError('O01 protected terminal End mutation was accepted')


def local_fences(material, node):
    r27, base, config, claim, _, _, legacy_predecessors, legacy_ordinary, *_ = material
    require(node not in base['live_authority'](config) and node not in base['local_holds'](), 'formal workload or adopted ledger protects target')
    if node in legacy_ordinary:
        r27['require_no_slots'](base['JOURNAL'], node, legacy_ordinary[node], (1, 2))
        r27['require_no_slots'](base['JOURNAL'], node, legacy_predecessors[node]['accounting'][0], (0, 1, 2))
    r27['require_no_slots'](base['JOURNAL'], node, claim['targets'][node]['original_GPU'], (0, 1, 2))


def publication(material, revision):
    _, base, config, *_ = material
    require(Path(__file__).resolve() == WRAPPER and re.fullmatch(r'[a-f0-9]{40}', revision), 'exact installed publication required')
    base['verify_publication'](revision, config)
    git = ['git', '--no-optional-locks', '-c', 'safe.directory=' + ROOT.as_posix()]
    for path in (WRAPPER, CLAIM, ROOT / R27[0], ROOT / O01[0]):
        blob = subprocess.check_output([*git, 'cat-file', 'blob', revision + ':' + path.relative_to(ROOT).as_posix()],
                                       cwd=ROOT, stderr=subprocess.PIPE, timeout=8)
        require(blob == pinned(path), 'exact published successor bytes differ')


def run(material, node, revision, claim_sha, observe):
    _, base, config, claim, *_ = material
    spec = claim['targets'][node]
    provisional, bound = rendered(material, node, revision)
    fork = bound(base, 'observe', node, spec)
    if observe:
        original, send_token, _ = fork.__globals__['paths'](node, spec['request'])
        require(send_token.is_file(), 'no fixed original send token')
        saved = json.loads(pinned(original))
        guard, _ = rendered(material, node, saved['publication'])
        require(saved['program_SHA256'] == digest(guard['launch'].encode()), 'sent original host bytes differ')
        return fork(node, guard, config)
    publication(material, revision)
    local_fences(material, node)
    intent, send_token, reply = fork.__globals__['paths'](node, spec['request'])
    require(not any(os.path.lexists(path) for path in (intent, send_token, reply)), 'fixed successor already attempted; reconcile exact original')
    launcher = config[0]
    helper = runpy.run_path(str(launcher['HELPER']))
    sender = runpy.run_path(str(launcher['TRANSPORT']))
    record = dict(node=node, request=spec['request'], publication=revision, claim_SHA256=claim_sha,
                  program_SHA256=digest(provisional['launch'].encode()), GPU_requested=8, replay_allowed=False)
    config[1]['save'](intent, record)
    transport = sender['send'](helper['ssh'](ROOT.parent), request=spec['request'], program=provisional['launch'],
                              data=b'', send_token=send_token, authority_seconds=18, outcome_seconds=82)
    config[1]['save'](reply, dict(record, transport=transport))
    return dict(node=node, request=spec['request'], receipt=reply.name, SHA256=digest(pinned(reply)),
                state=(transport.get('response') or {}).get('state'), replay_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original', action='store_true')
    parser.add_argument('--node', choices=NODES)
    args = parser.parse_args()
    require(args.check or args.node, 'one fixed node required')
    material = load()
    raw = pinned(DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw) == material[3], 'exact four singleton claim differs')
    if args.check:
        r27, _, config, *_ = material
        nodes = {}
        for node in NODES:
            source, _ = rendered(material, node, None)
            r27['offline_staging_fixture'](source['launch'], config[6])
            if node == NODE_O01:
                check_o01_account_gate(source['launch'], material[-1])
            nodes[node] = dict(request=source['request'], worker_seconds=source['worker_seconds'],
                               program_SHA256=digest(source['launch'].encode()))
        out = dict(nodes=nodes, staging_pass=4, staging_vetoes=8, O01_accounting_pass=1, O01_accounting_vetoes=1,
                   GPU=0, Slurm=0, SCNet=0, Git=0)
    else:
        out = run(material, args.node, args.publication, digest(raw), args.observe_original)
    print(json.dumps(dict(out, claim_SHA256=digest(raw)), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
