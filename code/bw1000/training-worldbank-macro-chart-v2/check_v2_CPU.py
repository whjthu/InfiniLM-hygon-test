"""New country-name and original-attempt/timeout boundaries only. No originals/HTTP/render."""
import copy
import json
from pathlib import Path
from contracts import country_names_from_facts,need,plan_check
from fetch import socket_timeout
from original_chain import REQUEST,validate_records


def run():
    p=json.loads((Path(__file__).parent/'PLAN.json').read_bytes());plan_check(p);payloads={}
    for split,g in p['splits'].items():
        rows=[dict(countryiso3code=c,country=dict(value='Synthetic country '+c)) for c in g['countries'] for _ in range(21)]
        payloads[split]=[dict(page=1,pages=1,total=168),rows]
    names=country_names_from_facts(payloads,p);need(len(names)==24,'Actual full24 fixture')
    rejected=[]
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):rejected.append(name);return
        raise AssertionError('Accepted bad fixture '+name)
    q=copy.deepcopy(payloads);q['train'][1][0]['country']['value']=''
    bad('empty_country_name',lambda:country_names_from_facts(q,p))
    q2=copy.deepcopy(payloads);q2['train'][1][0]['country']['value']='different'
    bad('same_iso_name_drift',lambda:country_names_from_facts(q2,p))
    q3=copy.deepcopy(payloads);q3['train'][1][0]['countryiso3code']=p['splits']['dev']['countries'][0]
    bad('cross_split_iso',lambda:country_names_from_facts(q3,p))
    q4=copy.deepcopy(payloads);q4['train'][1][0]['country']['value']=123
    bad('untyped_country_name',lambda:country_names_from_facts(q4,p))
    bad('missing_split_names',lambda:country_names_from_facts({'train':payloads['train']},p))
    q5=copy.deepcopy(payloads);q5['train'][1][0]['countryiso3code']=q5['train'][1][21]['countryiso3code'];q5['train'][1][0]['country']['value']=q5['train'][1][21]['country']['value']
    bad('country_row_denominator_20_22',lambda:country_names_from_facts(q5,p))
    intent=dict(request=REQUEST,publication='009e0d693e8cf8c52634e01cebeb5d734f74a5fa',source_manifest_SHA256='93ef8719b59659e460b40bda8549bd2fd681e57eaa2cea6c9104d83a5e6034e6',epoch=1794928469.7191937,deadline=1794928769.7191937)
    outcome=dict(intent,known_terminal=True,job=dict(returncode=1,timed_out=False));worker=dict(epoch=intent['epoch'],deadline=intent['deadline'],passed=False,error_type='TimeoutError',attempted_GETs=4,completed_GETs=3,actual_charts=0,actual_inputs=0)
    events=[dict(phase='GET.enter',endpoint=n) for n in ['indicator_0','indicator_1','indicator_2','countries']]+[dict(phase='GET.headers',endpoint=n) for n in ['indicator_0','indicator_1','indicator_2']]+[dict(phase='failure')]
    validate_records(intent,outcome,worker,events)
    prior=copy.deepcopy(events);prior.insert(-1,dict(phase='GET.enter',endpoint='facts_train'))
    bad('prior_fact_attempt_forbids_scope',lambda:validate_records(intent,outcome,worker,prior))
    unknown=copy.deepcopy(outcome);unknown['known_terminal']=False
    bad('UNKNOWN_prior_forbids_scope',lambda:validate_records(intent,unknown,worker,events))
    need([socket_timeout(x) for x in (100,75,3)]==[75,75,3],'Actual75/remaining cap changed')
    bad('expired_socket',lambda:socket_timeout(0));bad('nonfinite_socket',lambda:socket_timeout(float('nan')))
    need(set(p['endpoints'])=={'facts_train','facts_dev','facts_blind'} and sum(e['cap_bytes'] for e in p['endpoints'].values())==6291456,'Only new3 facts GET scope')
    return dict(passed=True,normal_contracts=4,rejected_count=len(rejected),rejected=rejected,synthetic_names=24,synthetic_rows=504,
        new_GETs_planned=3,original_attempts_retained=4,total_network_denominator=7,actual_HTTP=0,original_files_read=0,render_calls=0,old_tests_repeated=False,
        Dataset_GO=False,training_GO=False,GPU_GO=False)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
