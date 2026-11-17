"""Small new contract boundaries only; no remote, source weight, Torch or SDK read."""
import ast
import copy
import json
from pathlib import Path
import tempfile

D = Path(__file__).resolve().parent


def functions(name, names):
    module = ast.parse((D/name).read_bytes())
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), name, 'exec'), scope)
    return scope


def main():
    p = json.loads((D/'PLAN.json').read_bytes())
    funcs = functions('endpoint.py', {'require', 'validate_phase', 'cross_receipt'})
    validate, cross = funcs['validate_phase'], funcs['cross_receipt']
    valid, rejected = [], []

    def bad(label, operation):
        try:
            operation()
        except (ValueError, KeyError):
            rejected.append(label)
        else:
            raise AssertionError('bad input accepted: '+label)

    for i in (0, 1, 18):
        validate(p, i); valid.append('valid-phase-'+str(i))
    for index in (-1, 19, True):
        bad('unknown-phase-'+str(index), lambda i=index: validate(p, i))
    for label, mutate in (
        ('oversize', lambda x: x['phases'][0].update(payload_bytes=(1<<30)+1)),
        ('partial-denominator', lambda x: x['phases'][0].update(payload_bytes=1)),
        ('path-traversal', lambda x: x['phases'][0]['segments'][0].update(name='../config.json')),
        ('negative-offset', lambda x: x['phases'][0]['segments'][0].update(offset=-1)),
        ('range-outside-file', lambda x: x['phases'][0]['segments'][0].update(offset=10**12)),
    ):
        value = copy.deepcopy(p); mutate(value)
        bad(label, lambda x=value: validate(x, 0))
    context = dict(request_id='a'*32, publication='b'*40, source_manifest_SHA256='c'*64,
                   asset_plan_SHA256='d'*64, endpoint_SHA256='e'*64, phase=0, epoch=100., deadline=7300.)
    segments = [dict(name='config.json', offset=0, bytes=2505, SHA256='f'*64)]
    receipt = dict(context, passed=True, mode='source', segments=segments, host='source-host', pid=42, startticks=7, uid=1001)
    cross(receipt, context, segments); valid.append('complete-source-receipt')
    for label, changes in (('partial-failure', {'passed': False}), ('nonce-mismatch', {'request_id': '0'*32}),
                           ('range-hash-different', {'segments': []}), ('wrong-UID', {'uid': 0}),
                           ('epoch-reset', {'deadline': 7301.})):
        value = dict(receipt, **changes)
        bad(label, lambda x=value: cross(x, context, segments))
    observation = functions('reconcile.py', {'process_state'})['process_state']
    identity = dict(host='original', boot_id='boot', startticks=7)
    assert observation(identity, 'another-login', 'other-boot', None) is None
    assert observation(identity, 'original', 'boot', dict(startticks=7, state='S')) is True
    assert observation(identity, 'original', 'boot', None) is False
    valid.append('cross-host-is-UNKNOWN')
    transport = functions('relay_once.py', {'transport_state'})['transport_state']
    assert transport(True, True, 0) == transport(True, False, 255) == 'UNKNOWN'
    valid.append('local-SSH-kill-is-UNKNOWN')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); phase = root/'00'; phase.mkdir()
        (phase/'00.partial').write_bytes(b'partial')
        try:
            phase.mkdir()
        except FileExistsError:
            rejected.append('fixed-phase-partial-blocks-new-nonce')
        else:
            raise AssertionError('partial phase replayed')
    print(json.dumps(dict(passed=True, valid=valid, rejected=rejected, valid_count=len(valid),
                          rejected_count=len(rejected), pure_CPU=True, payload_reads=0, remote_calls=0)))


if __name__ == '__main__':
    main()
