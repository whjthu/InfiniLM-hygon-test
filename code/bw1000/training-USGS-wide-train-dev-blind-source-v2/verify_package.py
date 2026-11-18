"""Stdlib AST/byte/package invariants only; no fixtures, original data or network."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent

def verify():
    mf=json.loads((D/'MANIFEST.json').read_bytes())['files']
    actual={p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==set(mf)|{'MANIFEST.json'}
    for name,pin in mf.items():
        p=D/name;raw=p.read_bytes();assert not p.is_symlink() and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
        if p.suffix=='.py':ast.parse(raw)
    plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    assert all(plan[k] is claim[k] and type(plan[k])is bool for k in ['root_GO','CPU_GO','source_GET_GO'])
    assert not claim['GPU_GO'] and not claim['training_GO'] and claim['max_invocations']==1
    assert plan['maximum_features']==claim['network']['max_event_slots']==500 and plan['body_cap_bytes']==1048576 and plan['GETs']==1
    assert plan['grouping']['geographic_overlap_allowed'] is True and plan['grouping']['geographic_OOD_required'] is False
    assert plan['minimum_new_groups']==9 and plan['hit_cap_policy'].startswith('ACCEPT_BOUNDED_FIRST500')
    assert plan['whole_epoch_seconds']==120 and plan['old_source_rows']==250 and plan['charts']==30 and plan['QA']==150
    assert all(plan['split_requirements'][s]==dict(minimum_groups=3,charts=10,QA=50) for s in ['train','dev','blind'])
    protocol=(D/'EVALUATION_PROTOCOL.json').read_bytes();assert len(protocol)==plan['evaluation_protocol']['bytes'] and hashlib.sha256(protocol).hexdigest()==plan['evaluation_protocol']['SHA256']==claim['evaluation_protocol']['SHA256']
    p=json.loads(protocol);assert p['generation']['max_new_tokens']==256 and p['arms']==['initial','domain25','generic25']
    original=json.loads((D/'ORIGIN.json').read_bytes())
    for n in ['saved_parser.py','chart_contracts.py','job_guard.py','clock_ipc.py','with_qualified_overlay.py']:
        b=(D/n).read_bytes();assert len(b)==original['original_sources'][n]['bytes'] and hashlib.sha256(b).hexdigest()==original['original_sources'][n]['SHA256']
    for n,pin in original['legacy_scorer'].items():
        b=(D/'frozen_scorer'/n).read_bytes();assert len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['SHA256']
    return dict(passed=True,source_members=len(mf),AST_files=len(list(D.rglob('*.py'))),root_GO=claim['root_GO'],CPU_GO=claim['CPU_GO'],source_GET_GO=claim['source_GET_GO'],actual_GETs=0,actual_fixtures=0,actual_model=0,MF_SHA256=hashlib.sha256((D/'MANIFEST.json').read_bytes()).hexdigest())

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
