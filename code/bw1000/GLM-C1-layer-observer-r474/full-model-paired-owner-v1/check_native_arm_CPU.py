
import ast, copy, hashlib, json, pathlib, sys, tempfile
from unittest.mock import patch
import native_arm as arm
source = pathlib.Path(__file__).resolve().parent
repo = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else source.parents[3]
g = repo
p=g/'code/bw1000/GLM-C1-layer-observer-r474/full-model-screen32-v1'
sys.path.insert(0,str(p))
import observer_trial, typed_binding
terminal=json.loads((g/'results/bw1000/2026-11-13/GLM474-projection-single-drain-pair-1103606-59-CPU-terminal.json').read_bytes())
assert terminal['verified'] and terminal['outcome_SHA256']==arm.OUTCOME_SHA256
fixture=dict(claim=terminal['claim_commit'],passed=terminal['passed'],owned_srun_reaped=True,
    terminal={'terminal_visible':True},GPU_cards=terminal['GPU_cards'],accounting=terminal['accounting'],build=terminal['build'])
arm.verify_outcome(fixture)
failures=[
  lambda x:x.update(claim='wrong'),lambda x:x.update(passed=False),
  lambda x:x.update(owned_srun_reaped=False),lambda x:x['terminal'].update(terminal_visible=False),
  lambda x:x.update(GPU_cards=1),lambda x:x['accounting'][0].__setitem__(0,'1103606.58'),
  lambda x:x['build']['MD'].update(equal_fresh_M=False),lambda x:x['build'].update(runtime_created=True),
  lambda x:x['build']['binaries'].pop('baseline'),lambda x:x['build']['binaries']['baseline'].update(providers=70),
  lambda x:x['build']['binaries']['single-drain'].update(library_SHA256='0'*64),
]
def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('bad evidence accepted')
for mutate in failures:
    value=copy.deepcopy(fixture);mutate(value);reject(lambda:arm.verify_outcome(value))
reject(lambda:arm.verified_library('invalid'))
with tempfile.TemporaryDirectory() as d:
    root=pathlib.Path(d)
    data=json.dumps(fixture).encode()
    (root/'outcome.json').write_bytes(data)
    libraries={}
    for name in arm.ARMS:
        content=name.encode()
        path=root/name
        path.write_bytes(content)
        libraries[name]=(path,hashlib.sha256(content).hexdigest())
    with patch.object(arm,'BUILD',root),patch.object(arm,'OUTCOME_SHA256',hashlib.sha256(data).hexdigest()),patch.object(arm,'verify_outcome'):
        with patch.object(arm,'ARMS',libraries):
            assert all(arm.verified_library(name)==libraries[name] for name in libraries)
            libraries['single-drain'][0].write_bytes(b'wrong')
            reject(lambda:arm.verified_library('single-drain'))
            (root/'outcome.json').write_bytes(data+b' ')
            reject(lambda:arm.verified_library('baseline'))
    with patch.object(arm,'verified_library',return_value=libraries['baseline']),patch.object(arm.ctypes,'CDLL') as cdll:
        class WrongTyped:
            __file__=str(root/'outcome.json')
        reject(lambda:arm.load_owner('baseline',WrongTyped,observer_trial,None))
        cdll.assert_not_called()
        assert arm.sha256(pathlib.Path(typed_binding.__file__))==arm.TYPED_BINDING_SHA256
        assert arm.sha256(pathlib.Path(observer_trial.__file__))==arm.OBSERVER_BINDING_SHA256
        mock_maps='12-34 r-xp 00 00:00 1 '+str(next(iter(arm.ARMS.values()))[0])+' (deleted)\n'
        with patch.object(pathlib.Path,'read_text',return_value=mock_maps):
            reject(lambda:arm.load_owner('baseline',typed_binding,observer_trial,None))
        cdll.assert_not_called()
original=ast.parse((p/'owner.py').read_bytes())
candidate=ast.parse((source / 'owner.py').read_bytes())
body=lambda m:next(n for n in m.body if isinstance(n,ast.FunctionDef) and n.name=='main_owned')
old_try=next(n for n in body(original).body if isinstance(n,ast.Try))
new_try=next(n for n in body(candidate).body if isinstance(n,ast.Try))
def request_and_cleanup(block):
    start = next(i for i, n in enumerate(block.body)
                 if isinstance(n, ast.ImportFrom) and n.module == 'scoring')
    return ast.dump(ast.Module(body=block.body[start:], type_ignores=[]))
assert request_and_cleanup(old_try)==request_and_cleanup(new_try)
assert all(ast.dump(a)==ast.dump(b) for a,b in zip(old_try.handlers+old_try.finalbody,new_try.handlers+new_try.finalbody))
new_src=(source / 'owner.py').read_text()
assert new_src.index('load_owner(arm_name')<new_src.index('ctx=GlobalTP16Context')<new_src.index('load_checkpoint(ctx')
print('PASS: actual paired link summary, both arm identities, 16 rejection cases, preserved request/cleanup AST; no Native/GPU')
