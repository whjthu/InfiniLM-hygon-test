"""Resume the O01 standby after the independently sealed v2 pre-GPU failure."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory
import time
from types import FunctionType


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-o01-3087-guarded-bf16-v3'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
V2 = ROOT / 'tools/run_training80_o01_3087_guarded_refill_v2_once.py'
V2_SOURCE = '52d9d8bc677e0a41d0f60a9d316b9c0ed0cd796af71e2a4bd2fc50a838d8308c'
V2_CLAIM = '92c4fd39eac32c5f53186c224f06ed0448705a77a2f940f67cd154268f19e737'
V2_PROGRAM = '68e0933eff91460e2241ba83469f95c07ac91b8694407bb8b93d73795386edd1'
V2_PUBLICATION = '14ce14208c4a889768b9029a22a0863848d5f727'
V2_READ = ('026d9e3f626d43efa0b542f38200993c', '9d08bcd7e933015a3219cd54022ce2a7f1fbec97935bc3675bfb7bcec04ded7a')
V2_RECEIPTS = ('09f1cdcf5584aaa3e670b37225ce37e5ce9d9e7b8a0f85ff6f3cd46cd050f750',
               '1e607169b0e430a7b599616705c52da55c30cc6d442fd84c0571bfb981d33462',
               '809d02d7081c9e447b9a622227a460c1878bc1bcd5121ce2bbfcaac94a7ea15f')
V2_ARCHIVE = LOCAL / 'training80-o01-v2-r27-original-probe-and-namespace-independent.json'
V2_ARCHIVE_SHA = 'aa138770d819b62a6264393af41951d379ff9460e6045d60a231cce045d66aba'
REQUEST = 'cce84178ab4a44c2bda5464fcf089526'
OLD = '4c893c07f6234e28b7d692a2eb3be06d'
NODE, PARENT = 'f12r2n11', '1103606'
ZERO = '1103606.3295'
TYPE_ERROR = 'byte indices must be integers or slices, not str'
ERROR_SHA = '47a7271894feb6b5309ce1eee7f756854ef215331ebe716b8d377e6da7bdbd02'
NAMESPACE = f'training80-dcu-standby-{OLD}'
OLD_AUDIT = f'training80-o01-3087-guarded-bf16-v2-{NODE}-{OLD}'


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prove_v2(v2, v1, material):
    base = material[0]
    rows, pins = v1['terminal_receipt'](material, *V2_READ)
    old_program = v2['render'](v1, material, V2_PUBLICATION, rows, pins)['launch']
    require(material[3] == V2_CLAIM and digest(old_program.encode()) == V2_PROGRAM
            and digest(TYPE_ERROR.encode()) == ERROR_SHA, 'actual v2 renderer or fault identity differs')
    paths = v2['v2_bound'](v1)(base, 'paths')(NODE, OLD)
    intent, sent, reply = (json.loads(base['pin'](path, pin)) for path, pin in zip(paths, V2_RECEIPTS, strict=True))
    transport = reply['transport']
    response = transport['response']
    require(intent == dict(node=NODE, request=OLD, publication=V2_PUBLICATION, claim_SHA256=V2_CLAIM,
                           program_SHA256=V2_PROGRAM, GPU_requested=8, replay_allowed=False)
            and {key: reply[key] for key in intent} == intent and sent['request'] == transport['request'] == response['request'] == OLD
            and sent['payload_sha256'] == transport['payload_sha256'] and transport['rc'] == 0
            and transport['remote_outcome'] == 'RESPONSE_OBSERVED'
            and all(transport[name] is True for name in ('authority_verified', 'remote_received', 'payload_send_attempted', 'payload_flushed'))
            and (response['state'], response['detail_class'], response['detail_SHA256'], response['node'], response['parent'],
                 response['probe_step'], response['GPU_srun_attempted'], response['zero_GRES_call_attempted'], response['mutex_acquired'])
            == ('GPU_NOT_SENT', 'TypeError', ERROR_SHA, NODE, PARENT, ZERO, False, True, True),
            'actual authenticated v2 no-GPU-send receipt differs')
    evidence = json.loads(base['pin'](V2_ARCHIVE, V2_ARCHIVE_SHA))
    body = evidence['body']
    row, = body['accounting']
    old_artifacts = body['files']
    probe = old_artifacts['fresh-probe.json']
    require(evidence['rc'] == body['new_slurm'] == response['probe_rc'] == 0 and body['request'] == OLD
            and len(row) == 9 and row[0] == ZERO and row[2:4] == ['COMPLETED', '0:0']
            and row[:4] + row[5:7] + row[8:] == response['fresh_zero_GRES_accounting']
            and probe['reported_original_payload_SHA256'] == response['probe_stdout_SHA256']
            and old_artifacts['standby-namespace'] == {'directory': True}
            and old_artifacts['standby-worker'] == {'bytes': 0, 'SHA256': digest(b'')}
            and old_artifacts['gpu-send-intent.json'] is None,
            'independent v2 .3295 terminal or original incomplete namespace differs')
    preserved = {f'{OLD_AUDIT}/{name}': (old_artifacts[name]['bytes'], old_artifacts[name]['SHA256'])
                 for name in ('intent.json', 'fresh-probe-intent.json', 'fresh-probe.json', 'outcome.json')}
    preserved[f'{NAMESPACE}/intent.json'] = (old_artifacts['standby-intent']['bytes'], old_artifacts['standby-intent']['SHA256'])
    preserved[f'{NAMESPACE}/worker.py'] = (0, digest(b''))
    return row, preserved


def binder(v2, v1):
    original = v2['v2_bound']
    return FunctionType(original.__code__, dict(original.__globals__, STEM=STEM))(v1)


def render(v2, v1, material, publication, rows, pins, old_row, old_files):
    original = v2['render']
    inherited = FunctionType(original.__code__, dict(original.__globals__, REQUEST=REQUEST, v2_bound=lambda v1: binder(v2, v1)))
    program = inherited(v1, material, publication, rows, pins)
    source = program['launch']
    bad = ('                    raw = file.read(8193)\n'
           "                require(0 < len(raw) <= 8192 and sha256(raw).hexdigest() == expected, 'prior worker or release changed')")
    good = ('                    protected_bytes = file.read(8193)\n'
            "                require(0 < len(protected_bytes) <= 8192 and sha256(protected_bytes).hexdigest() == expected, 'prior worker or release changed')")
    boundary = '            check_previous(account(PREVIOUS))'
    stage = '            root.mkdir(mode=0o700)'
    require(source.count(bad) == source.count(boundary) == source.count(stage) == 1, 'source map, locked gate or worker staging moved')
    seven = old_row[:4] + old_row[5:7] + old_row[8:]
    tombstone = (
        f'            for rel_v2, (size_v2, sha_v2) in {old_files!r}.items():\n'
        '                old_v2_file = RESULTS / rel_v2\n'
        "                require(old_v2_file.resolve(strict=True) == old_v2_file, 'v2 old path differs')\n"
        "                with old_v2_file.open('rb') as old_v2_handle:\n"
        '                    old_v2_bytes = old_v2_handle.read(8193)\n'
        "                require(len(old_v2_bytes) == size_v2 <= 8192 and sha256(old_v2_bytes).hexdigest() == sha_v2, 'v2 tombstone differs')\n"
        f"            require(not os.path.lexists(RESULTS / {OLD_AUDIT!r} / 'gpu-send-intent.json'), 'v2 GPU intent now exists')\n"
        f"            require(account({ZERO!r}) == {seven!r}, 'v2 exact zero-GRES terminal differs')\n"
    )
    source_pins = ("            require(isinstance(raw, dict) and set(raw) == set(PINS)\n"
                   "                    and {name: sha256(value).hexdigest() for name, value in raw.items()} == PINS, 'standby staging source map differs')\n")
    program['launch'] = source.replace(bad, good, 1).replace(boundary, tombstone + boundary, 1).replace(stage, source_pins + stage, 1)
    compile(program['launch'], '<O01 v3 exact guarded continuation>', 'exec')
    return program


def staging_fixture(source, standby):
    module = ast.parse(source)
    guard = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'run_guarded')
    locked = next(node for node in ast.walk(guard) if isinstance(node, ast.With)
                  and any(isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == 'lock' for item in node.items))
    original = next(node for node in locked.body if isinstance(node, ast.For) and ast.unparse(node.target) == '(rel, expected)')
    offset = next(i for i, node in enumerate(locked.body) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'root.mkdir')
    stage = locked.body[offset - 1:offset + 3]
    require('standby staging source map differs' in ast.unparse(stage[0]) and 'stream.write(raw[\'worker.py\'])' in ast.unparse(stage[-1]),
            'actual source map guard or staging block differs')
    write = next(node for node in ast.parse(standby['controller.py']).body if isinstance(node, ast.FunctionDef) and node.name == 'write')
    writer = dict(os=os, json=json)
    exec(compile(ast.Module(body=[write], type_ignores=[]), '<frozen controller write>', 'exec'), writer)
    protected = b'fixture protected file'
    pins = {'fixture-present.json': digest(protected), 'fixture-absent.json': None}
    original.iter.func.value = ast.parse(repr(pins), mode='eval').body
    for variant in ('exact', 'changed', 'source_map', 'old_shadow'):
        loop = original
        body = stage
        if variant == 'old_shadow':
            loop = ast.parse(ast.unparse(original).replace('protected_bytes', 'raw')).body[0]
            body = stage[1:]
        with TemporaryDirectory() as temp:
            results = Path(temp).resolve()
            (results / 'fixture-present.json').write_bytes(b'drift' if variant == 'changed' else protected)
            raw = dict(standby)
            root = results / 'empty-v3-fixture'
            expected_pins = {name: digest(data) for name, data in raw.items()}
            if variant == 'source_map':
                raw = b'incorrect previously shadowed payload'
            scope = dict(RESULTS=results, NODE=NODE, PARENT=PARENT, REQUEST=REQUEST, sha256=sha256, require=require, root=root,
                         controller={'write': writer['write']}, PUBLICATION=None, outcome={'probe_stdout_SHA256': 'b' * 64},
                         time=time, CLEAR_STEP='1103606.3086', step='1103606.99999', os=os, raw=raw, PINS=expected_pins)
            try:
                exec(compile(ast.fix_missing_locations(ast.Module(body=[loop, *body], type_ignores=[])), '<actual O01 v3 staging>', 'exec'), scope)
            except (ValueError, TypeError) as error:
                if variant == 'changed':
                    require(type(error) is ValueError and not root.exists() and str(error) == 'prior worker or release changed', 'protected byte drift accepted')
                elif variant == 'source_map':
                    require(type(error) is ValueError and not root.exists() and str(error) == 'standby staging source map differs', 'invalid source map accepted')
                elif variant == 'old_shadow':
                    require(type(error) is TypeError and digest(str(error).encode()) == ERROR_SHA
                            and (root / 'worker.py').stat().st_size == 0, 'fixture did not reproduce actual v2 error')
                else:
                    raise
            else:
                require(variant == 'exact' and scope['raw'] is raw and (root / 'worker.py').read_bytes() == standby['worker.py'],
                        'v3 fixture did not write exact standby bytes')


def load():
    require(V2.resolve(strict=True) == V2 and digest(V2.read_bytes()) == V2_SOURCE, 'frozen v2 source differs')
    v2 = runpy.run_path(str(V2))
    v1, material = v2['load']()
    old_row, old_files = prove_v2(v2, v1, material)
    base, cfg, old_claim, _, *rest = material
    raw = base['pin'](CLAIM)
    claim = json.loads(raw)
    old_pins = dict(request=OLD, publication=V2_PUBLICATION, source_SHA256=V2_SOURCE, claim_SHA256=V2_CLAIM,
                    rendered_SHA256=V2_PROGRAM, failed_probe=ZERO, independent_SHA256=V2_ARCHIVE_SHA,
                    detail_SHA256=ERROR_SHA, actual_GPU_send=False, replay_allowed=False)
    require(claim == dict(old_claim, request=REQUEST, scope=STEM, launcher='tools/run_training80_o01_3087_guarded_refill_v3_once.py',
                         second_predecessor_guard_failure=old_pins), 'v3 scope differs')
    return v2, v1, (base, cfg, claim, digest(raw), *rest), old_row, old_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original', action='store_true')
    parser.add_argument('--terminal-receipt', nargs=2, metavar=('READ_ID', 'SHA256'))
    args = parser.parse_args()
    require(bool(args.publication) == bool(args.terminal_receipt), 'a newly read live .3087 terminal receipt is required')
    v2, v1, material, old_row, old_files = load()
    base, cfg, _, claim_pin, *_ = material
    rows, pins = v1['terminal_receipt'](material, *V2_READ)
    if args.publication:
        v2['verify_publication'](v1, args.publication, base, cfg)
        git = ['git', '--no-optional-locks', '-c', 'safe.directory=' + ROOT.as_posix()]
        for path in (Path(__file__).resolve(), CLAIM):
            ref = args.publication + ':' + path.relative_to(ROOT).as_posix()
            observed = subprocess.check_output([*git, 'cat-file', 'blob', ref], cwd=ROOT, stderr=subprocess.PIPE, timeout=10)
            require(observed == base['pin'](path), 'published v3 bytes differ')
        ranks = base['live_authority'](cfg)
        require(args.terminal_receipt[0] not in {V2_READ[0], v2['OLD_READ'][0]}, 'used v1 or v2 observation cannot admit v3')
        rows, pins = v1['terminal_receipt'](material, *args.terminal_receipt, fresh=True)
    program = render(v2, v1, material, args.publication, rows, pins, old_row, old_files)
    bound = binder(v2, v1)
    if args.observe_original:
        result = bound(base, 'observe')(NODE, program, cfg)
    else:
        v2['check_protected_fixture'](program['launch'], rows[2])
        staging_fixture(program['launch'], cfg[6])
        result = (bound(base, 'dispatch')(NODE, program, args.publication, claim_pin, cfg, ranks) if args.publication else
                  dict(state='V2_NO_GPU_SEND_BOUND_V3_FRESH_ADMISSION_PENDING', request=REQUEST, claim_SHA256=claim_pin,
                       launch_SHA256=digest(program['launch'].encode()), worker_seconds=program['worker_seconds'],
                       accounting_fixtures=2, staging_fixtures=4, actual_Slurm=0, actual_GPU=0, SCNet=0, Git=0))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
