"""Reuse only the three completed metadata bodies; original countries failure stays failed."""
import hashlib
import json
from pathlib import Path
from contracts import canonical,need,parse_indicator,strict_json

G=Path(__file__).resolve().parents[3]
ORIGINAL=G/'local-only/training-worldbank-macro-chart-v1-original'
REQUEST='22ae4128d8294182834782278956d36b'
PINS={
    'intent.json':('698568bf9ad159889201f3bc954dd9d9ca9550db75e8e304d1ad6ab1916c1cdc',479),
    'outcome.json':('5ed8ed62e9a468603918219c5698aa7f9502f35099a245d75c302a4185723c3e',1171),
    'worker-result.json':('de490c93b0696d4b681740b5fdb6f3c4d066f86da234c6f00ad9c22e9e0de84f',415),
    'progress.jsonl':('41802e238cc2b5427b210b970e4d0b262b394fad27255774a90f24a5ff956cba',1816),
    'raw/indicator_0.json':('ca424dd8dac40839e4c48353ba5b9bb3d0344a44ba22edac002762b49f1cc82b',852),
    'raw/indicator_1.json':('2498a3b5be51eb1bb22e2e4d058d6c9fec726e3fc09b6b34035b6e8cb0edf1b7',673),
    'raw/indicator_2.json':('ed7703d841d4dde2d1cb858f0dd53229fe3e841f72abb17dbc8c950ad20c88ce',909)
}


def validate_records(intent,outcome,worker,events):
    need(intent['request']==outcome['request']==REQUEST and intent['publication']=='009e0d693e8cf8c52634e01cebeb5d734f74a5fa'
        and intent['source_manifest_SHA256']=='93ef8719b59659e460b40bda8549bd2fd681e57eaa2cea6c9104d83a5e6034e6','Original request/publication differs')
    need(outcome['known_terminal'] is True and outcome['job']['returncode']==1 and outcome['job']['timed_out'] is False
        and worker['passed'] is False and worker['error_type']=='TimeoutError' and worker['attempted_GETs']==4 and worker['completed_GETs']==3
        and worker['actual_charts']==worker['actual_inputs']==0,'Original known failure differs')
    need(intent['epoch']==outcome['epoch']==worker['epoch']==1794928469.7191937 and intent['deadline']==outcome['deadline']==worker['deadline']==1794928769.7191937,'Original epoch differs')
    entered=[e['endpoint'] for e in events if e['phase']=='GET.enter']
    need(entered==['indicator_0','indicator_1','indicator_2','countries'] and events[-1]['phase']=='failure','Original facts may have been attempted; prohibit any new request')
    need([e['endpoint'] for e in events if e['phase']=='GET.headers']==['indicator_0','indicator_1','indicator_2'],'Original completed metadata headers differ')
    return dict(original_request=REQUEST,original_known_terminal=True,original_failed=True,original_attempted_GETs=4,original_completed_GETs=3,
        original_entered_endpoints=entered,facts_previously_requested=False,original_FAILURE_retained=True)


def load_cached_metadata(directory,plan,progress):
    originals={}
    for name,(expected,size) in PINS.items():
        path=ORIGINAL/name;need(path.is_file() and not path.is_symlink() and path.stat().st_size==size,'Original cached source missing/changed')
        raw=path.read_bytes();need(len(raw)==size and hashlib.sha256(raw).hexdigest()==expected,'Original cached source SHA differs');originals[name]=raw
    intent=json.loads(originals['intent.json']);outcome=json.loads(originals['outcome.json']);worker=json.loads(originals['worker-result.json'])
    events=[json.loads(x) for x in originals['progress.jsonl'].splitlines()];chain=validate_records(intent,outcome,worker,events)
    documents={};cached={}
    for name,endpoint in plan['cached_metadata'].items():
        raw=originals['raw/'+name+'.json'];parse_indicator(strict_json(raw),endpoint['indicator'],plan)
        (directory/(name+'.json')).write_bytes(raw);documents[name]=raw
        cached[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),original_file='raw/'+name+'.json',new_HTTP_requests=0)
        progress('metadata.cached',dict(endpoint=name,**cached[name]))
    chain.update(original_pins={n:dict(SHA256=p[0],bytes=p[1]) for n,p in PINS.items()},cached_metadata=cached)
    return documents,chain
