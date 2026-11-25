"""Correct the protected O01 accounting projection for one guarded refill."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
from types import FunctionType


REPO = Path(__file__).resolve().parents[1]
V1 = REPO / 'tools/run_training80_o01_3087_guarded_refill_once.py'
V1_PIN = '97326a1adaee29a89966bf13676d7c6d7476e274fe5be43c41c0b854679468b2'
STEM = 'training80-o01-3087-guarded-bf16-v2'
CLAIM = REPO / 'coordination/tasks' / f'{STEM}.json'
NODE, PARENT, PROTECTED = 'f12r2n11', '1103606', '1103606.2424'
REQUEST = '4c893c07f6234e28b7d692a2eb3be06d'
REJECTED = '31c51e3924a143629ca435bd7e01d235'
OLD_PUBLICATION = '5acb238c75a521d085d7c2eb36336ccea399013e'
OLD_READ = ('98d642543dff42bfaca15f5d47d9c5b3', '2416ea4d8e78ff6652e3b2ddfa91faacdae6407110fe32d60186a06228ca90c8')
OLD_CLAIM = '6b16daee66a0ce2c52d6373875dc9e545c09c86f616d5cde547cf8475abe85d7'
OLD_PROGRAM = '0dd8749a0462463f46a9f50d8c305b46ce2a45e9eb2dbed145b9bf004c3eac73'
OLD_RECEIPTS = {'intent': 'f9bf8073b8f5b90f86691a5951b8f6aa7ef45dfeacf55d5d397ed42e2909b79a',
                'send': 'b29482c5d2b0f97614d8dee5e1548b6a313c3d0ae0806de6e1cb0f1dbeebefec',
                'reply': '82d6645b5d176f8c4dd05502ebaabc18aa6d67aedb85249c7bfc628a48cddcb0'}
ERROR = 'protected O01 zero-GRES terminal differs'
OLD_ERROR_PIN = '94df2a6ace48856a661e7226700afa037d70835d55c8a3adc3d0e4c13f0fa554'


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def account_columns(row):
    step, name, state, exit_code, _, node, tres, _, end = row
    require((step, state, exit_code, node, tres) == (PROTECTED, 'COMPLETED', '0:0', NODE, 'cpu=1,mem=256M,node=1'),
            'protected historical accounting identity differs')
    return [step, name, state, exit_code, node, tres, end]


def v2_bound(v1):
    function = v1['bound']
    return FunctionType(function.__code__, dict(function.__globals__, STEM=STEM))


def prove_v1_no_send(v1, material):
    base = material[0]
    paths = v1['bound'](base, 'paths')(NODE, REJECTED)
    intent, token, reply = (json.loads(base['pin'](path, pin)) for path, pin in zip(paths, OLD_RECEIPTS.values(), strict=True))
    expected_intent = dict(node=NODE, request=REJECTED, publication=OLD_PUBLICATION, claim_SHA256=OLD_CLAIM,
                           program_SHA256=OLD_PROGRAM, GPU_requested=8, replay_allowed=False)
    transport = reply['transport']
    observed = transport['response']
    require(intent == expected_intent and {key: reply[key] for key in intent} == intent
            and token['request'] == transport['request'] == REJECTED
            and token['payload_sha256'] == transport['payload_sha256']
            and transport['rc'] == 0 and transport['remote_outcome'] == 'RESPONSE_OBSERVED'
            and all(transport[name] is True for name in ('authority_verified', 'remote_received', 'payload_send_attempted', 'payload_flushed'))
            and observed == dict(GPU_requested=8, GPU_srun_attempted=False, detail_SHA256=OLD_ERROR_PIN,
                                 detail_class='ValueError', mutex_acquired=True, node=NODE, parent=PARENT,
                                 replay_allowed=False, request=REJECTED, state='GPU_NOT_SENT', zero_GRES_call_attempted=False),
            'v1 original authenticated no-Slurm-send receipt differs')
    rows, pins = v1['terminal_receipt'](material, *OLD_READ)
    program = v1['render'](material, OLD_PUBLICATION, rows, pins)['launch']
    old = f"            require(protected == {rows[2][:4] + rows[2][5:]!r}, '{ERROR}')"
    require(material[3] == OLD_CLAIM and digest(program.encode()) == OLD_PROGRAM
            and program.count(old) == 1 and digest(ERROR.encode()) == OLD_ERROR_PIN,
            'v1 actual erroneous projection is not bound to the original outcome')


def render(v1, material, publication, rows, pins):
    function = v1['render']
    inherited = FunctionType(function.__code__, dict(function.__globals__, bound=v2_bound(v1)))
    output = inherited(material, publication, rows, pins)
    protected = rows[2]
    original = f"            require(protected == {protected[:4] + protected[5:]!r}, '{ERROR}')"
    replacement = f"            require(protected == {account_columns(protected)!r}, '{ERROR}')"
    source = output['launch']
    require(source.count(original) == 1, 'frozen protected guard boundary moved')
    output['launch'] = source.replace(original, replacement, 1)
    compile(output['launch'], '<O01 3087 v2 guarded successor>', 'exec')
    require(output['request'] == REQUEST, 'rendered successor request differs')
    return output


def check_protected_fixture(program, protected):
    module = ast.parse(program)
    account = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'account')
    setup = next(node for node in module.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'FMT' for target in node.targets))
    gate = [node for node in ast.walk(module) if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
            and node.left.id == 'protected']
    require(len(gate) == 1 and len(gate[0].comparators[0].elts) == 7, 'protected guard is not using seven actual columns')
    predicate = next(node for node in ast.walk(module) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and node.value.args and node.value.args[0] is gate[0])
    binding = next(node for node in ast.walk(module) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'protected' for target in node.targets))
    region = ast.Module(body=[setup, account, binding, predicate], type_ignores=[])
    code = compile(region, '<real O01 v2 protected account and predicate>', 'exec')
    seven = account_columns(protected)

    def query(*argv):
        require(argv == ('sacct', '-nP', '-j', PROTECTED, '-o', 'JobIDRaw,JobName%110,State,ExitCode,NodeList,AllocTRES,End'),
                'actual accounting query does not have the pinned seven fields')
        return ['|'.join(seven)]

    exec(code, dict(query=query, require=require))
    seven[-1] = '2026-11-24T05:28:31'
    try:
        exec(code, dict(query=query, require=require))
    except ValueError as error:
        require(str(error) == ERROR, 'wrong rejection on mutated End')
    else:
        raise ValueError('actual rendered predicate accepted a mutated End')


def load():
    require(V1.resolve(strict=True) == V1 and digest(V1.read_bytes()) == V1_PIN, 'frozen O01 v1 runner changed')
    v1 = runpy.run_path(str(V1))
    material = v1['load']()
    base, config, old_claim, _, *rest = material
    prove_v1_no_send(v1, material)
    raw = base['pin'](CLAIM)
    claim = json.loads(raw)
    old = dict(old_claim, scope=STEM, request=REQUEST, launcher='tools/run_training80_o01_3087_guarded_refill_v2_once.py')
    require(claim == dict(old, predecessor_guard_failure=dict(request=REJECTED, publication=OLD_PUBLICATION,
                        source_SHA256=V1_PIN, claim_SHA256=OLD_CLAIM, rendered_SHA256=OLD_PROGRAM,
                        original_predecessor_read_id=OLD_READ[0], original_predecessor_read_SHA256=OLD_READ[1],
                        receipt_SHA256=OLD_RECEIPTS, detail_SHA256=OLD_ERROR_PIN, actual_GPU_send=False,
                        actual_zero_GRES_send=False, replay_allowed=False)), 'v2 exact successor scope differs')
    return v1, (base, config, claim, digest(raw), *rest)


def verify_publication(v1, revision, base, config):
    v1['publish'](revision, base, config)
    git = ['git', '--no-optional-locks', '-c', 'safe.directory=' + REPO.as_posix()]
    for path in (Path(__file__).resolve(), CLAIM):
        ref = revision + ':' + path.relative_to(REPO).as_posix()
        got = subprocess.check_output([*git, 'cat-file', 'blob', ref], cwd=REPO, stderr=subprocess.PIPE, timeout=10)
        require(got == base['pin'](path), 'exact published v2 successor bytes differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original', action='store_true')
    parser.add_argument('--terminal-receipt', nargs=2, metavar=('READ_ID', 'SHA256'))
    args = parser.parse_args()
    require(bool(args.publication) == bool(args.terminal_receipt), 'actual predecessor read ID and SHA required for dispatch')
    v1, material = load()
    base, config, _, claim_pin, *_ = material
    rows, pins = v1['terminal_receipt'](material, *OLD_READ)
    program = render(v1, material, None, rows, pins)
    if args.check:
        check_protected_fixture(program['launch'], rows[2])
        result = dict(state='V1_NO_SLURM_SEND_BOUND_V2_FRESH_ADMISSION_PENDING', request=REQUEST, claim_SHA256=claim_pin,
                      compiled_only_SHA256=digest(program['launch'].encode()), fixture_pass=1, mutated_End_rejected=1,
                      worker_seconds=program['worker_seconds'], GPU=0, Slurm=0, SCNet=0, Git=0)
    elif args.observe_original:
        result = v2_bound(v1)(base, 'observe')(NODE, program, config)
    else:
        verify_publication(v1, args.publication, base, config)
        ranks = base['live_authority'](config)
        require(args.terminal_receipt[0] != OLD_READ[0], 'the used v1 readback cannot admit v2')
        rows, pins = v1['terminal_receipt'](material, *args.terminal_receipt, fresh=True)
        program = render(v1, material, args.publication, rows, pins)
        check_protected_fixture(program['launch'], rows[2])
        result = v2_bound(v1)(base, 'dispatch')(NODE, program, args.publication, claim_pin, config, ranks)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
