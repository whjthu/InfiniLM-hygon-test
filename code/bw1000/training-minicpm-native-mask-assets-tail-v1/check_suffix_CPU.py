"""Use the actual append code to reject wrong prefixes, ranges and suffix bytes."""
from pathlib import Path
import copy,hashlib,io,tempfile
from append_suffix import append_verified_suffix
whole=b'original prefix plus exact remaining bytes';offset=15
base=dict(prefix_bytes=offset,prefix_SHA256=hashlib.sha256(whole[:offset]).hexdigest(),
          remaining_bytes=len(whole)-offset,archive_bytes=len(whole),archive_SHA256=hashlib.sha256(whole).hexdigest())
def run(cfg,tail,prefix=None,expect=None):
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'part';p.write_bytes(whole[:offset] if prefix is None else prefix)
        try:append_verified_suffix(p,io.BytesIO(tail),cfg,lambda:None)
        except AssertionError:
            assert p.read_bytes().startswith(whole[:offset] if prefix is None else prefix)
            if expect=='unchanged':assert p.read_bytes()==(whole[:offset] if prefix is None else prefix)
            raise
        assert p.read_bytes()==whole
run(base,whole[offset:]);rejected=0
for cfg,tail,prefix,expect in [
    (dict(base,prefix_SHA256='0'*64),whole[offset:],None,'unchanged'),
    (dict(base,prefix_bytes=offset+1),whole[offset:],None,'unchanged'),
    (dict(base,remaining_bytes=1),whole[offset:],None,'unchanged'),
    (dict(base,archive_bytes=len(whole)+1),whole[offset:],None,'unchanged'),
    (base,whole[offset:],b'x'*offset,'unchanged'),
    (base,whole[offset:-1],None,None),
    (base,whole[offset:]+b'extra',None,None),
    (base,b'x'*(len(whole)-offset),None,None),
    (dict(base,archive_SHA256='f'*64),whole[offset:],None,None),
]:
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'part';original=whole[:offset] if prefix is None else prefix;path.write_bytes(original)
        error=None
        try:append_verified_suffix(path,io.BytesIO(tail),cfg,lambda:None)
        except AssertionError as exc:error=exc
        assert error is not None,'invalid suffix accepted'
        assert path.read_bytes().startswith(original)
        if expect=='unchanged':assert path.read_bytes()==original
        rejected+=1
assert rejected==9
print('PASS: exact suffix preserved original prefix;9 wrong-prefix/range/missing/extra/hash rejections. Tiny CPU bytes only.')
