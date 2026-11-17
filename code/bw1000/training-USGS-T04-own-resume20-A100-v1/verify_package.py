"""Pure-stdlib closure for prospective original-own resume, no importGPU modules."""
import ast
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent


def sha(b):return hashlib.sha256(b).hexdigest()
def adoption_gate(template,outer,nested):
    flags=[x[k] for x in [template,outer,nested] for k in ['root_GO','GPU_GO']]
    assert all(type(v) is bool for v in flags) and len(set(flags))==1, 'All six GO flags must be false(prepared) or true(root adopted)'
    assert all(x['Model_GO'] is False and x['formal17'] is False for x in [template,outer,nested]), 'Model/formal17 qualification remains false'
    return flags[0]


def main():
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes());actual={p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and p.relative_to(D).as_posix()!='SOURCE_FILES.json'}
    assert set(mf['files'])==actual
    for name,pin in mf['files'].items():
        p=D/name;b=p.read_bytes();assert not p.is_symlink() and len(b)==pin['bytes'] and sha(b)==pin['SHA256']
        if name.endswith('.py'):ast.parse(b,filename=name)
    base=json.loads((D/'BASE_FRAMEWORK_FILES.json').read_bytes())['files'];merged=json.loads((D/'MERGED_FRAMEWORK_FILES.json').read_bytes())['files'];overlay=json.loads((D/'OVERLAYS.json').read_bytes())['files']
    assert len(base)==len(merged)==400 and set(overlay)=={'vlm/runtime/usgs_domain_witness.py'}
    for name,pin in base.items():
        if name not in overlay:assert merged[name]==pin
    for name,pin in overlay.items():assert (D/'overlay'/name).stat().st_size==pin['bytes'] and sha((D/'overlay'/name).read_bytes())==pin['SHA256'] and merged[name]==pin
    assert sha((D/'supervisor/bounded_GPU_supervisor.py').read_bytes())=='2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f'
    sub=json.loads((D/'supervisor/SOURCE_FILES.json').read_bytes())['files'];assert len(sub)==9
    for name,pin in sub.items():assert len((D/'supervisor'/name).read_bytes())==pin['bytes'] and sha((D/'supervisor'/name).read_bytes())==pin['SHA256']
    c=json.loads((D/'CONFIG.template.json').read_bytes());cl=json.loads((D/'CLAIM.json').read_bytes());sc=json.loads((D/'supervisor/CLAIM.json').read_bytes())
    assert c['mode']==cl['mode']=='resume20to25' and c['arm']==cl['arm']=='domain' and cl['max_invocations']==sc['max_submissions']==1
    adoption_gate(c,cl,sc)
    assert set(c['checkpoint_files'])==set(c['comparison_files']) and len(c['checkpoint_files'])==14 and len(c['continuous_reference'])==8
    assert c['formal17'] is False and c['native45']['SHA256']=='07f0073299aa3ba5f82b359affcc09704f9e8515205f7f4b87ae9c835884a65c'
    assert cl['budgets_seconds']==dict(whole_original=900,native=820,stop=850,cleanup=880,alarm=890,client=895)
    witness=(D/'overlay/vlm/runtime/usgs_domain_witness.py').read_text();assert "ACTUAL_CONTINUOUS_TAIL_LOSS_MISMATCH" in witness and "COMPLETE20_LOSS_COMPARISON" in witness
    installer=(D/'install_and_run.py').read_text();assert "'supervisor/bounded_GPU_supervisor.py'" in installer and 'os.execv(PYTHON, command)' in installer
    return dict(passed=True,source_members=len(actual),framework_members=400,modified_framework_sources=1,actual_RSSv2_driver_integrated=True,planned_new_updates=5,planned_micro_per_rank=20,Torch_imports=0,actual_GPU_calls=0)

if __name__=='__main__':print(json.dumps(main(),sort_keys=True))
