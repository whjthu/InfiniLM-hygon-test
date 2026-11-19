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
    need(c['native_identities']['bytes']==209332 and c['native_identities']['SHA256']=='8a00cba2a880325ca194245c57dfd2452b3e8e7a7103b697331492d126c6a118','ACTUAL_NATIVE150_REFERENCE')
    need(c['data_binding']['bytes']==98073 and c['data_binding']['SHA256']=='6553a078e06232bf6edaf9082d6fa66b62c5cc8affa587e3d92cc9b92646fe8c','ACTUAL_NEW_DATA_BINDING')
    need(c['native150_public']['bytes']==5202 and c['native150_public']['SHA256']=='24db6e22097401f1f9cab77cdb83b37b62bcef54482a28cb8600123412a11648','ACTUAL_PUBLISHED_NATIVE150')
    need(c['planned_generations']==claim['scope']['real_generations']==300 and c['per_arm']==100
         and c['split_denominators']=={'dev':50,'blind':50} and len(c['media'])==20,'FIXED300_SCOPE')
    need(c['inputs']['dev']=={'bytes':32020,'SHA256':'bd435df1258d1d35aec6f1b4998aaede7ce6aa4e6d5a5b7755815d79e85f292e','rows':50}
         and c['inputs']['blind']=={'bytes':32120,'SHA256':'baa6938bb171f3fd883a131941b63d752802fe6319320f62061e1130ad645c06','rows':50},'EXACT_NEW_TWO_INPUT_FILES')
    for name,pin in c['public_prerequisites'].items():
        raw=(D/'core/prerequisites'/name).read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'PUBLISHED_PREREQUISITE')
    for name,pin in [('response_protocol.py',c['content_parser']),('EVALUATION_PROTOCOL.json',c['protocol_pin'])]:
        raw=(D/'core'/name).read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'FROZEN_NEW_PROTOCOL')
    need(c['content_parser']['SHA256']=='5dce77a93bc557bb3c6aee479ca4775d5862190896ecfca5c68f171cb65f98c6','PREREGISTERED_PARSER')
    need(claim['clock']['whole_seconds']==plan['whole_seconds']==c['whole_epoch_cap_seconds']==5400
         and claim['clock']['native_deadline_offset']==5250 and claim['clock']['supervisor_stop_offset']==5300
         and claim['clock']['held_and_registry_cleanup_offset']==5360 and claim['clock']['ordinary_SSH_client_offset']==5395,'ONE_CLOCK_OFFSETS')
    need(plan['CPU_count']==4 and plan['RSS_soft_bytes']==64*1024**3
         and plan['host_MemAvailable_admission_bytes']==32*1024**3 and plan['host_MemAvailable_runtime_stop_bytes']==8*1024**3,'RESOURCE_SCOPE')
    origin=json.loads((D/'ORIGIN.json').read_bytes())
    for pin in origin['unchanged_members']:
        raw=(D/pin['path']).read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'UNCHANGED_QUALIFIED_HELPER_OR_MATH')
    reader=ast.parse((D/'read_original.py').read_bytes())
    cp=[n.value.value for n in reader.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REVIEWED_CALLER_SHA256' for t in n.targets)]
    need(cp==[hashlib.sha256((D/'core/infer_wide100.py').read_bytes()).hexdigest()],'PRE_REGISTERED_CALLER_PIN')
    need(c['paired25_training_metadata_qualified'] is True and c['original_T03_recovery_drift_retained'] is True,'NO_OLD_RECOVERY_RECLASSIFICATION')
    return dict(passed=True,files=len(mf),root_GO=flags[0],GPU_GO=flags[0],
        source_manifest_SHA256=hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest(),
        actual_model=0,actual_remote=0,actual_GPU=0,fixture_executions=0)

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
