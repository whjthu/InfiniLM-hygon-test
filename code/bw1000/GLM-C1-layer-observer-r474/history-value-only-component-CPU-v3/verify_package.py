"""This package's schema/bytes only; never compiler, Native or target execution."""
import hashlib,json,re
from pathlib import Path
def verify(directory):
    d=Path(directory);h=lambda b:hashlib.sha256(b).hexdigest();mf=json.loads((d/'SOURCE_FILES.json').read_bytes())
    actual={p.relative_to(d).as_posix() for p in d.rglob('*') if p.is_file()}
    assert actual==set(mf)|{'SOURCE_FILES.json'}
    for name,pin in mf.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=d/name;assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and h(p.read_bytes())==pin['SHA256']
    c=json.loads((d/'CLAIM.json').read_bytes())
    assert c['whole_epoch_seconds']==300 and c['inside_from_origin_seconds']==270 and c['work_from_origin_seconds']==265
    assert c['parent']=='1103606' and c['node']=='f13r1n05' and c['CPU']==4 and c['memory_GiB']==8
    assert c['max_invocations']==1 and not c['GPU_GO'] and not c['Model_GO'] and not c['Native_dlopen']
    assert c['planned']==dict(freshM=1,HOST_compile=1,GPU_compile=0,objects=1,links=1)
    b=(d/'component.cpp').read_bytes();assert h(b)==c['component_source_SHA256']
    proof=json.loads((d/'SOURCE_PROOF.json').read_bytes());patches=proof['cast_removals']
    assert proof['original_CPP_SHA256']=='8fdef1b3442ee82db6de9e60ffcaf98c176f3e6cdc5ac57180670bc6f44ac295' and len(patches)==5
    restored=b
    for patch in reversed(patches):
        a,z=patch['new_begin'],patch['new_end'];before=patch['before'].encode();after=patch['after'].encode()
        assert re.fullmatch(rb'(view|backing)->data\(\)',after) and before==b'static_cast<std::uint8_t*>('+after+b')'
        assert restored[a:z]==after
        restored=restored[:a]+before+restored[z:]
    assert len(restored)==proof['original_CPP_bytes']==18372 and h(restored)==proof['original_CPP_SHA256']
    assert h(restored.replace(b'\r\n',b'\n'))==proof['normalized_LF_original_SHA256']
    assert h(b.replace(b'\r\n',b'\n'))==proof['normalized_LF_corrected_SHA256']
    names=set(re.findall(r'extern "C"\s+(?:const char\*|void\*|int)\s*(glm_history_component_[a-z]+)\(',b.decode()))
    assert names==set(c['expected_CABI']) and len(names)==9
    assert h((d/'ENV.sh').read_bytes())=='147ea951bfe67a82ba39ef62e2e695e4ae7d21f8cc03f95b60054527ed960db9'
    assert h((d/'TOOLCHAIN.json').read_bytes())=='6d126b777d7abe4e30e311615303dade6a304d968c01b765c9dc9ee93e4d0b46'
    return dict(source_schema_passed=True,HOST=1,GPU=0,CABI=9,ROOT_GO=c['ROOT_GO'])
