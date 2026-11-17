"""New parser-only fixture; AST extracts pure functions, no Linux/resource import."""
import ast
import hashlib
import json
import math
from pathlib import Path

D=Path(__file__).resolve().parent
tree=ast.parse((D/'read_tail.py').read_text(encoding='utf8'))
parts=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('need','parse_tail')]
scope=dict(json=json,math=math)
exec(compile(ast.Module(body=parts,type_ignores=[]),'pure-tail-parser','exec'),scope)
parse=scope['parse_tail']
progress=dict(ordinal=20,generated=10,forward_attempts=12019,forward_completed=12019)
# Actual owner.mark keys from owner.py54-65,120,123; extra event fields must not escape.
event=dict(phase='forward.return',epoch=1794958494.0,monotonic=100.0,
           tokens=1,past=1322,elapsed_s=.2,progress=progress)
line=lambda x:json.dumps(x).encode()+b'\n'
cases=[]
for name,raw,offset in [('complete',line(event),0),
                        ('leading_fragment',b'partial}\n'+line(event),123),
                        ('unfinished_suffix',line(event)+b'{"phase":',0)]:
    result=parse(raw,offset,101.)
    assert result['progress']==progress and result['same_host_boot_event_age_s']==1.
    assert not set(result)&{'tokens','past','elapsed_s','epoch'}
    cases.append(dict(name=name,expected='PASS',passed=True))
for name,raw in [('unknown_phase',line(dict(event,phase='unregistered.phase'))),
                 ('partial_only',b'{"phase":'),
                 ('bad_counter',line(dict(event,progress=dict(progress,generated=True)))),
                 ('missing_progress',line({k:v for k,v in event.items() if k!='progress'}))]:
    try:parse(raw,0,101.)
    except ValueError:cases.append(dict(name=name,expected='REJECT',passed=True))
    else:raise AssertionError(name)
print(json.dumps(dict(passed=True,normal=3,rejected=4,cases=cases,
                     source_SHA256=hashlib.sha256((D/'read_tail.py').read_bytes()).hexdigest(),
                     source_scope='new pure tail parser only; no target/proc/Torch/GPU/old fixtures')))
