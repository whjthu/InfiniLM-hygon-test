"""Synthetic new closure contract only; no event files, HTTP, renderer or model."""
import copy,json
from select_groups import select


def row(i,region,lon,time):
    return dict(id=i,aliases=[i],source_geo_region_key=region,time_UTC_epoch_ms=time,
        values=dict(latitude='0',longitude=str(lon),magnitude='5',depth_km='10'),status='QUALIFIED_SOURCE_NUMERIC_RECORD',errors=[],null_fields=[])


def check():
    old=[row('old'+str(i),'old',-60,0) for i in range(50)]
    new=[row('new'+str(i),'g'+str(0 if i<15 else 1 if i<20 else 2),30 if i<15 else 90 if i<20 else 150,100*86400000+i*1000) for i in range(25)]
    plan={'seed':'synthetic-only'};good=select(old,new,plan)
    assert good['passed'] and len(good['charts'])==5 and good['selected_group_count']==3 and good['excluded_count']==0
    # An old alias bridge contaminates every new component connected through its region, not only one row.
    contaminated=copy.deepcopy(new);contaminated[0]['aliases'].append('old0');q=select(old,contaminated,plan)
    assert q['excluded_count']==15 and q['passed'] is False and q['charts']==[]
    insufficient=select(old,new[:10],plan);assert insufficient['passed'] is False and insufficient['charts']==[]
    bad=copy.deepcopy(new);bad[0]['null_fields']=['depth_km']
    try:select(old,bad,plan)
    except ValueError:pass
    else:raise AssertionError('NULL_NOT_REJECTED')
    assert [c['id'] for c in good['charts']]==[c['id'] for c in select(old,list(reversed(new)),plan)['charts']]
    return dict(passed=True,synthetic_qualified=1,component_alias_contamination_rejected=1,insufficient_no_backfill=1,null_source_rejected=1,permutation_stable=1,
        actual_source_or_gold_reads=0,HTTP=0,renderer=0,native_model=0)


if __name__=='__main__':print(json.dumps(check()))
