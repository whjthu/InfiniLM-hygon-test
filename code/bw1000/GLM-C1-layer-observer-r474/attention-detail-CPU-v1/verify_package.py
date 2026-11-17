"""Local bytes/structure checks only; never imports the builder or launcher."""
import hashlib
import json
from pathlib import Path


def verify(directory):
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((directory/'SOURCE_FILES.json').read_bytes())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    assert actual == set(manifest['files']) | {'SOURCE_FILES.json'}
    for name, pin in manifest['files'].items():
        p = directory/name
        assert not p.is_symlink() and p.stat().st_size == pin['bytes'] and digest(p) == pin['SHA256'], name
    assert digest(directory/'source/SOURCE_MANIFEST.json') == 'be1177120696b89d419b40439256b5fa20248c962a6a8cc5332fbe43e5b8fb0c'
    original = json.loads((directory/'source/SOURCE_MANIFEST.json').read_bytes())
    for name, pin in original['files_SHA256'].items():
        assert digest(directory/'source'/name) == pin
    claim = json.loads((directory/'CLAIM.json').read_bytes())
    assert (claim['parent'],claim['node'],claim['CPU_cores'],claim['memory_MiB'],claim['GPU_cards']) == ('1103606','f13r1n05',4,16384,0)
    assert claim['GPU_GO'] is False and claim['target_compile_performed'] is False
    plan = json.loads((directory/'BUILD_PLAN.json').read_bytes())
    assert len(plan['units']) == 3 and len(plan['reused_objects']) == 29 and len(plan['parent_objects']) == 32
    assert len(plan['providers']) == 71 and plan['strong_exports'] == 280
    assert [u['name'] for u in plan['units']] == ['nov','prepare','output']
    replacement = {u['parent_object']:u['object'] for u in plan['units']}
    assert [x for x in plan['link_argv'] if x.endswith('.o')] == [replacement.get(x,x) for x in plan['parent_object_order']]
    assert set(plan['parent_objects']) == set(plan['reused_objects']) | set(replacement)
    assert plan['parent_library_SHA256'] == 'a98a093aab52f2e4d85a2d11e3407a033909daf9e247c643bffce6fc8b64b1bb'
    for unit in plan['units']:
        assert unit['compiler_prefix'][0] == '/usr/bin/g++'
        assert unit['required_dependencies'][unit['source']] == digest(directory/'source'/Path(unit['source']).name)
        assert all(not p.startswith('/refactor/') for p in unit['optional_pinned_system_dependencies'])
    return dict(state='LOCAL_BYTES_AND_STRUCTURE_PASS', units=3, reused=29, GPU_cards=0, target_compile=False)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent)))
