"""Local-only readiness report; deliberately cannot launch or compile."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()

def main():
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf.items():
        b=(D/name).read_bytes()
        if len(b)!=pin['bytes'] or H(b)!=pin['SHA256']:raise ValueError('PACKAGE_MEMBER_PIN')
    blocked=json.loads((D/'BLOCKERS.json').read_bytes())
    claim=json.loads((D/'CLAIM.json').read_bytes())
    binding=json.loads((D/'NATIVE169_BINDING.json').read_bytes())
    assert len(binding['object_order'])==len(set(binding['object_order']))==32
    assert set(binding['object_order'])==set(binding['objects']) and len(binding['providers'])==71
    assert all(claim[k] is False for k in ('ROOT_GO','CPU_GO','Native_compile_GO','GPU_GO','Model_GO'))
    assert claim['has_target_launcher'] is False
    result=dict(state='BLOCKED_EXACT_RECIPE_METADATA',READY=False,
        historical_compile_argv_missing=blocked['historical_compile_argv_missing_count'],
        dependency_pin_missing=blocked['missing_dependency_pin_count'],
        current_recipe_not_yet_closed=blocked['current_recipe_not_yet_closed'],
        parent_objects=32,providers=71,source_pins_verified=True,
        target_calls=0,compiler_calls=0,GPU_calls=0)
    print(json.dumps(result));return 2

if __name__=='__main__':raise SystemExit(main())
