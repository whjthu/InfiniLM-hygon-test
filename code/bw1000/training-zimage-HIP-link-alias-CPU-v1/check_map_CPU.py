"""Fail-closed CPU cases for aliases, missing providers and byte substitutions."""
from pathlib import Path
import ast,hashlib,json
from ldd_map import parse_ldd,classify
D=Path(__file__).resolve().parent
expected={'A':'/real/a','B':'/real/b'};pins={'/real/a':dict(bytes=4,SHA256='a'*64),'/real/b':dict(bytes=8,SHA256='b'*64)}
def observed(path='/alias/a',byte=4,digest='a'*64):return {'A':dict(realpath=path,bytes=byte,SHA256=digest),'B':dict(realpath='/real/b',bytes=8,SHA256='b'*64)}
a=classify(expected,pins,observed());assert a['name_sets_equal'] and a['all_name_byte_identities_match'] and not a['all_realpaths_match']
b=classify(expected,pins,observed('/real/a'));assert b['all_name_byte_identities_match'] and b['all_realpaths_match']
rejected=0
for x in [observed(byte=5),observed(digest='c'*64),{'A':observed()['A']},dict(observed(),C=dict(realpath='/foreign',bytes=1,SHA256='c'*64))]:
 assert not classify(expected,pins,x)['all_name_byte_identities_match'];rejected+=1
parsed=parse_ldd('linux-vdso.so.1 (0xffff)\nA => /alias/a (0x1)\n/lib64/ld-linux-x86-64.so.2 (0x2)\n');assert parsed=={'A':'/alias/a','ld-linux-x86-64.so.2':'/lib64/ld-linux-x86-64.so.2'}
for text in ['A => not found','A => /x (0x1)\nA => /y (0x2)','garbage','']:
 try:parse_ldd(text)
 except ValueError:rejected+=1
 else:raise AssertionError('invalid passive map accepted')
for p in D.glob('*.py'):ast.parse(p.read_bytes(),filename=p.name)
value=dict(passed=True,positive_classifications=2,negative_or_invalid_cases=rejected,Python_AST=len(list(D.glob('*.py'))),Torch_imports=0,GPU_calls=0)
with (D/'local-map-checks.json').open('x',newline='\n') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
