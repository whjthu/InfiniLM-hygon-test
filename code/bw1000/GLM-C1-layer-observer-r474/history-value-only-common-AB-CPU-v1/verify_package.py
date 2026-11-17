"""Package/schema admission only. No runtime, compiler or device access."""
import hashlib,json
from pathlib import Path

def verify(directory):
    D=Path(directory);H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())
    actual={p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file()}
    assert actual==set(mf)|{'SOURCE_FILES.json'}
    for name,pin in mf.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=D/name;assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and H(p)==pin['SHA256']
    p=json.loads((D/'PLAN.json').read_bytes());c=json.loads((D/'CLAIM.json').read_bytes())
    assert len(p['units'])==32 and len({u['source'] for u in p['units']})==32
    assert [u['index'] for u in p['units']]==list(range(32))
    assert sum(u['kind']=='HOST' for u in p['units'])==25 and sum(u['kind']=='GPU' for u in p['units'])==7
    assert len(p['prepare_aliases'])==11 and len(set(p['prepare_aliases']))==11
    assert p['expected_totals']==dict(freshM=64,HOST_compile=50,GPU_compile=14,link=2)
    assert c['whole_epoch_seconds']==1800 and c['inside_from_origin_seconds']==1740 and c['work_from_origin_seconds']==1680
    assert c['max_invocations']==1 and not c['GPU_GO'] and not c['Model_GO'] and not c['Native_dlopen']
    assert len(p['original169']['object_order'])==32 and len(p['original169']['providers'])==71
    for u in p['units']:assert p['project_pins'][u['source']]==u['source_SHA256']
    for h,blob in p['source_blobs'].items():assert H(D/blob)==h
    assert set(p['B_overrides'])==set(p['prepare_aliases'])|{p['units'][15]['source'],p['units'][17]['source']}
    assert all(p['project_pins'][s]=='79b5d89e96cc8b2d236bc21bbecd307c03f9ab14ce22d85cbb8c16e223fa32dc' for s in p['prepare_aliases'])
    return dict(source_schema_passed=True,units=32,planned_M=64,prepare_aliases=11,GO=c['ROOT_GO'])
