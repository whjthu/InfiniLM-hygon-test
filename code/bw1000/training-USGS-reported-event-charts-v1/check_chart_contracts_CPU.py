"""Only new grouping/display/QA/EOS contracts. Synthetic; no actual facts/render/network."""
from copy import deepcopy
from contracts import display,displayed_rows,make_plan,questions
from score_once import score
import json


def run():
    rows=[]
    # Three isolated coarse regions, each has two events. Group near closure includes alias cases.
    for i,(lat,lon) in enumerate([(-60,-150),(-59,-149),(0,0),(1,1),(60,150),(61,151)]):
        rows.append(dict(id='x'+str(i),slot=i,status='QUALIFIED_SOURCE_NUMERIC_RECORD',errors=[],null_fields=[],aliases=['x'+str(i)],event_alias_group='x'+str(i),source_geo_region_key='g'+str(i//2),time_UTC_epoch_ms=1790812800000+i*1000,magType='mb',values=dict(latitude=str(lat),longitude=str(lon),magnitude='4.995' if i%2==0 else '4.996',depth_km='1.004' if i%2==0 else '-1.004')))
    plan=dict(source_event_denominator=6,seed='synthetic-contract',charts_max=36,QA_max=180)
    p=make_plan(rows,plan);assert p['closure_groups']==3 and p['planned_charts']==3 and p['planned_QA']==15 and set(p['split_group_counts'].values())=={1}
    q=questions(p['charts'][0]);assert q[4]['answer']=='E01 E02' and q[2]['answer']=='-2.00' and display('-0.004')=='0.00'
    assert all(qi['display_rows_SHA256']==q[0]['display_rows_SHA256'] for qi in q)
    labels=[dict(x,data_status='QUALIFIED') for x in q];pred=[dict(id=x['id'],text=x['answer'],status='COMPLETED',natural_EOS=True,cap_hit=False) for x in q]
    assert score(labels,pred)['correct']==5
    pred[0]['cap_hit']=True;pred[1].pop('natural_EOS');pred.pop();assert score(labels,pred)['denominator']==5 and score(labels,pred)['correct']==2
    rejected=[]
    for name,modify in [('duplicate_ID',lambda x:x[1].update(id=x[0]['id'])),('unqualified_slot',lambda x:x[1].update(status='UNQUALIFIED_RETAINED_SOURCE_SLOT')),('missing_region',lambda x:x[1].update(source_geo_region_key=None)),('one_group',lambda x:[r.update(source_geo_region_key='all') for r in x])]:
        sample=deepcopy(rows);modify(sample)
        try:make_plan(sample,plan)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    missing=deepcopy(plan);missing['source_event_denominator']=7
    try:make_plan(rows,missing)
    except ValueError:rejected.append('wrong_source_denominator')
    else:raise AssertionError('wrong_source_denominator')
    # Close geographic boundary neighbours merge even when original coarse keys differ.
    extra=deepcopy(rows);extra[1]['source_geo_region_key']='boundary';assert make_plan(extra,plan)['closure_groups']==3
    # Aliases force closure even for far records; may eliminate enough split groups => fail closed.
    extra=deepcopy(rows);extra[4]['aliases'].append('x0')
    try:make_plan(extra,plan)
    except ValueError:rejected.append('alias_reduces_to_two_groups')
    else:raise AssertionError('alias closure')
    return dict(passed=True,normal_contracts=7,rejected=len(rejected),rejected_cases=rejected,source_actual_records_used=0,charts_rendered=0,HTTP_GETs=0,old_tests_rerun=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
