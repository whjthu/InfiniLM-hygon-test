"""Synthetic new scientific source contracts only, no actual USGS query/data."""
import copy
from datetime import datetime
from decimal import Decimal
from email.message import Message
import json
from pathlib import Path
from sample import audit,contract_gate,headers_gate,js,need


def run():
    c=json.loads((Path(__file__).parent/'CONTRACT.json').read_bytes());contract_gate(c)
    t=int(datetime.fromisoformat(c['fixed_start_UTC']).timestamp()*1000)+1000
    def feature(identifier,time,depth):
        return dict(type='Feature',id=identifier,properties=dict(type='earthquake',time=time,updated=time+1000,mag=Decimal('4.50'),magType='mb',ids=','+identifier+',',net='xx',sources=',xx,',status='reviewed'),geometry=dict(type='Point',coordinates=[Decimal('-10.1'),Decimal('2.3'),Decimal(depth)]))
    source=dict(type='FeatureCollection',metadata=dict(status=200,count=2,generated=t+10000,api='SYNTHETIC-ONLY'),features=[feature('xx1',t,'-1.20'),feature('xx2',t+100,'12.30')])
    result=audit(source,c);need(result['all_source_numeric_slots_qualified'] and result['actual_event_slots']==2 and result['maximum_requested_slots']==50 and result['rows'][0]['values']['depth_km']=='-1.20','Negative depth/Decimal/source denominator changed')
    missing=copy.deepcopy(source);missing['features'][0]['properties']['mag']=None
    r=audit(missing,c);need(r['actual_event_slots']==2 and r['unqualified_retained_slots']==1 and not r['all_source_numeric_slots_qualified'],'Null removed/imputed')
    aliases=copy.deepcopy(source);aliases['features'][1]['properties']['ids']=',xx2,xx1,'
    r=audit(aliases,c);need(r['event_alias_groups']==1 and r['unqualified_retained_slots']==2,'Alias duplicates inflated independent samples')
    empty=dict(type='FeatureCollection',metadata=dict(status=200,count=0,generated=t,api='SYNTHETIC'),features=[])
    need(not audit(empty,c)['all_source_numeric_slots_qualified'],'Empty sample qualified')
    rejected=[]
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):rejected.append(name);return
        raise AssertionError('Accepted '+name)
    x=copy.deepcopy(source);x['features'][1]['id']='xx1';bad('duplicate_primary_event',lambda:audit(x,c))
    x2=copy.deepcopy(source);x2['metadata']['count']=3;bad('header_feature_count_drift',lambda:audit(x2,c))
    x3=copy.deepcopy(c);x3['URL']=c['URL'].replace('2026-10-01','2026-10-02');bad('different_date_not_fallback',lambda:contract_gate(x3))
    x4=copy.deepcopy(c);x4['query_URL_previously_requested']=True;bad('old_requested_exactscope',lambda:contract_gate(x4))
    bad('nonfinite_JSON',lambda:js(b'{"mag":NaN}'))
    outside=copy.deepcopy(source);outside['features'][0]['properties']['time']=t-100000
    need(audit(outside,c)['unqualified_retained_slots']==1,'Out-of-window source slot lost')
    headers=Message();headers['Content-Type']='application/json';headers['Content-Length']='500'
    need(headers_gate(200,headers,c)==500,'Valid JSON header')
    bad('redirect_rejected',lambda:headers_gate(302,headers,c))
    over=Message();over['Content-Type']='application/json';over['Content-Length']='1048577';bad('cap_rejected_before_body',lambda:headers_gate(200,over,c))
    encoded=Message();encoded['Content-Type']='application/json';encoded['Content-Encoding']='gzip';bad('encoded_response',lambda:headers_gate(200,encoded,c))
    return dict(passed=True,normal_contracts=6,rejected_count=len(rejected),rejected=rejected,synthetic_features=2,
        actual_USGS_query_GETs=0,actual_original_data_read=0,charts_generated=0,QA_generated=0,models_loaded=0,
        old_tests_repeated=False,Dataset_GO=False,training_GO=False,GPU_GO=False)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
