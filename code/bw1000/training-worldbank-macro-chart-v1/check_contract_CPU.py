"""Synthetic stdlib boundary checks only; no official facts/media or renderer import."""
import copy
from decimal import Decimal
from email.message import Message
import hashlib
import json
from pathlib import Path

from contracts import array_response,canonical,display,need,parse_facts,parse_indicator,plan_check,questions,score,strict_json
from fetch import headers_guard


def run():
    plan=json.loads((Path(__file__).parent/'PLAN.json').read_bytes());valid=[];rejected=[]
    def good(name,fn):fn();valid.append(name)
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):rejected.append(name);return
        raise AssertionError('Bad fixture accepted: '+name)
    good('fixed_seed_country_year_query',lambda:plan_check(plan))
    badplan=copy.deepcopy(plan);badplan['splits']['dev']['countries'][0]=plan['splits']['train']['countries'][0]
    bad('cross_split_country',lambda:plan_check(badplan))
    badplan2=copy.deepcopy(plan);badplan2['endpoints']['facts_dev']['URL']+='&date=2000:2006'
    bad('query_drift',lambda:plan_check(badplan2))
    good('decimal_rounding_negativezero',lambda:need(display(Decimal('1.005'))=='1.01' and display(Decimal('-.004'))=='0.00','Round mismatch'))
    bad('duplicate_JSON',lambda:strict_json(b'{"x":1,"x":2}'))
    bad('nonfinite_JSON',lambda:strict_json(b'{"x":NaN}'))
    group=plan['splits']['train'];rows=[]
    for country in group['countries']:
        for indicator in plan['indicators']:
            for year in group['years']:
                rows.append(dict(countryiso3code=country,indicator=dict(id=indicator,value=plan['indicators'][indicator]['name']),date=str(year),unit='',value=Decimal('-1.005')))
    facts=[dict(page=1,pages=1,total=168,sourceid='2',lastupdated='2026-11-01'),rows]
    good('synthetic_full168',lambda:need(parse_facts(facts,'train',plan)[1]['actual_nonmissing']==168,'Count'))
    missing=copy.deepcopy(facts);missing[1][0]['value']=None
    good('missing_keeps168',lambda:need(parse_facts(missing,'train',plan)[1]['missing']==1,'Missing lost'))
    duplicate=copy.deepcopy(facts);duplicate[1][1]=duplicate[1][0]
    bad('duplicate_fact',lambda:parse_facts(duplicate,'train',plan))
    outside=copy.deepcopy(facts);outside[1][0]['date']='2020'
    bad('other_split_year',lambda:parse_facts(outside,'train',plan))
    unit=copy.deepcopy(facts);unit[1][0]['unit']='USD'
    bad('unexpected_unit',lambda:parse_facts(unit,'train',plan))
    nan=copy.deepcopy(facts);nan[1][0]['value']=Decimal('NaN')
    bad('nonfinite_fact',lambda:parse_facts(nan,'train',plan))
    code=next(iter(plan['indicators']))
    meta=[dict(page=1,pages=1,total=1),[dict(id=code,name=plan['indicators'][code]['name'],source=dict(id='2',value='World Development Indicators'),unit='',sourceNote='Synthetic not real source.',sourceOrganization='Fixture')]]
    good('metadata_schema_fixture',lambda:parse_indicator(meta,code,plan))
    rights=copy.deepcopy(meta);rights[1][0]['license']='new restricted terms'
    bad('new_rights_override',lambda:parse_indicator(rights,code,plan))
    qs,pins=questions('FIX',code,list(range(2000,2007)),[Decimal(v) for v in ('1.005','2','3','4','4','-1','4')],'fixture',plan)
    good('nine_questions_display_math_earliesttie',lambda:need(len(qs)==9 and qs[3]['answer']=='2.99' and qs[-1]['answer']=='2003','QA math/tie'))
    bad('incomplete_chart',lambda:questions('FIX',code,list(range(7)),[None]*7,'fixture',plan))
    labels=[dict(q,data_status='QUALIFIED') for q in qs]
    outputs={q['id']:'FINAL: '+q['answer'] for q in qs}
    good('score_all9',lambda:need(score(labels,outputs,9)['correct']==9,'Score'))
    outputs.pop(qs[0]['id']);outputs[qs[1]['id']]='FINAL: 0.00';outputs[qs[2]['id']]='truncated/cap'
    good('score_full_denominator',lambda:need((lambda s:(s['planned'],s['correct'],s['missing_outputs'],s['wrong_answers'],s['parse_failures'])==(9,6,1,1,1))(score(labels,outputs,9)),'Failed denominator'))
    bad('unknown_output_id',lambda:score(labels,{'foreign':'FINAL: 0.00'},9))
    h=Message();h['Content-Type']='application/json';h['Content-Length']='100'
    good('HTTP_valid_headers',lambda:need(headers_guard(200,h,100)==100,'Header'))
    bad('HTTP200_other_status',lambda:headers_guard(206,h,100))
    bad('HTTP_length_cap_beforebody',lambda:headers_guard(200,h,99))
    duplicateh=Message();duplicateh['Content-Type']='application/json';duplicateh['Content-Length']='10';duplicateh['Content-Length']='10'
    bad('HTTP_duplicate_length',lambda:headers_guard(200,duplicateh,100))
    encoded=Message();encoded['Content-Type']='application/json';encoded['Content-Encoding']='gzip'
    bad('HTTP_encodedbody',lambda:headers_guard(200,encoded,100))
    return dict(passed=True,valid_count=len(valid),rejected_count=len(rejected),valid=valid,rejected=rejected,
        real_HTTP_requests=0,official_facts_read=0,charts_rendered=0,synthetic_only=True,GPU_GO=False,training_GO=False)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
