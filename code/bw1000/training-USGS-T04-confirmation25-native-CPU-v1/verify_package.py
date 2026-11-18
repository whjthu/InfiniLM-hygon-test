"""Static package identity and CPU-only adoption consistency; no data/model read."""
import ast,json,time
from common import D,need,pinned,sha


def verify(budget=lambda:None):
    raw=(D/'SOURCE_FILES.json').read_bytes();mf=json.loads(raw)
    need(set(p.name for p in D.iterdir() if p.is_file())==set(mf['files'])|{'SOURCE_FILES.json'},'EXACT_SOURCE_INVENTORY')
    for name,pin in mf['files'].items():
        budget();need(name==__import__('pathlib').Path(name).name,'FLAT_SOURCE_PATH');body=pinned(D/name,pin)
        if name.endswith('.py'):ast.parse(body)
    plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    need(type(claim['root_GO']) is bool and claim['root_GO']==claim['CPU_GO']==plan['root_GO']==plan['CPU_GO'],'CONSISTENT_ROOT_CPU_GO')
    for key in ['GPU_GO','training_GO','Model_GO']:
        need(claim[key] is False and plan[key] is False,'CPU_ONLY_FALSE_DOWNSTREAM_GO')
    need(plan['whole_seconds']==300 and plan['planned_native_rows']==25 and len(plan['media'])==5,'FROZEN300_25_5')
    need(set(plan['cases'])=={'T04.sft'} and len(plan['cases']['T04.sft']['model_metadata']['files'])==14
         and len(plan['cases']['T04.sft']['Swift_sources'])==7,'EXACT14_7_T04')
    need(len(json.loads((D/'WRAPPER_SOURCES.json').read_bytes())['chain'])==3,'EXACT3WRAPPER')
    return dict(passed=True,root_GO=claim['root_GO'],CPU_GO=claim['CPU_GO'],source_members=len(mf['files']),source_manifest_SHA256=sha(raw))


if __name__=='__main__':print(json.dumps(verify()))
