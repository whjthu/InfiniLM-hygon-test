"""Stdlib byte/AST verification only; no original data, fixtures, imports or devices."""
import ast
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent

def need(ok,why):
    if not ok:raise ValueError(why)

def verify(budget=lambda:None):
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
    actual={p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    need(actual==set(mf)|{'SOURCE_FILES.json'},'EXACT_OUTER_INVENTORY')
    for name,pin in mf.items():
        budget();p=D/name;raw=p.read_bytes()
        need(not p.is_symlink() and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'EXACT_SOURCE_PIN')
        if p.suffix=='.py':ast.parse(raw)
    plan=json.loads((D/'INSTALL_PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    need(hashlib.sha256((D/'core/SOURCE_FILES.json').read_bytes()).hexdigest()==plan['core_manifest_SHA256'],'CORE_MF')
    flags=[]
    for name in ['CLAIM.json','INSTALL_PLAN.json','core/PLAN.json','core/CLAIM.json']:
        x=json.loads((D/name).read_bytes());flags.extend([x['root_GO'],x['GPU_GO']])
    need(all(type(x)is bool and x==flags[0] for x in flags),'EIGHT_MATCHING_GO_FLAGS')
    need(claim['max_invocations']==1 and not claim['Model_GO'] and not claim['training_GO'] and not claim['formal17'],'DEV_DIAGNOSTIC_ONLY')
    need(hashlib.sha256((D/'qualified/bounded_GPU_supervisor.py').read_bytes()).hexdigest()==plan['qualified_driver_SHA256']=='2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f','QUALIFIED_RSS_HELPER')
    c=json.loads((D/'core/PLAN.json').read_bytes());f=json.loads((D/'core/FRAMEWORK_BINDING.json').read_bytes())
    need(len(f['files'])==404 and c['initial_named_SHA256']=='086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137','SAME_INITIAL_ORIGINAL404')
    need(c['native_identities']['bytes']==73407 and c['native_identities']['SHA256']=='34ce51a75a3ffa98ca9362ad1a0f339011e66927b9583c27df5f310a84e16b0e','ORIGINAL_DEV_INPUT_REFERENCE')
    need(c['paired25_training_metadata_qualified'] is True and c['original_T03_recovery_drift_retained'] is True,'NO_OLD_RECOVERY_RECLASSIFICATION')
    return dict(passed=True,files=len(mf),root_GO=flags[0],GPU_GO=flags[0],
        source_manifest_SHA256=hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest(),
        actual_model=0,actual_remote=0,actual_GPU=0,fixture_executions=0)

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
