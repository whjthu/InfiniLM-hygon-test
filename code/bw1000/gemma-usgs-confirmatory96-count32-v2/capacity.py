"""Freeze and match the fixed 32-query Gemma USGS capacity design without network I/O."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from urllib.parse import urlencode


D=Path(__file__).resolve().parent
REPO=D.parents[1]
QUARTER=REPO/'local-only/gemma-usgs-confirmatory96-v1-four-month-source'
IDENTITY=REPO/'local-only/gemma-usgs-confirmatory96-v1-historic-identity/metadata-only-identity.json'
IDENTITY_PIN={'bytes':225451,'SHA256':'e22d89b8c58aea49e2291395b47a04b3c27aacf8b18e489cf945fe3a4c2e2885'}
PREVIOUS='f305c9c471a6f5142c5d4ad728257845f2f83aa67a043764d94aaaaef3a58de1'
MONTHS={
    '2023-01-01':(456786,'4b03b0eb1daca44059b99f1252a3ff3e1e397e2960368141fe5c8edb43db99e7',645,639),
    '2023-04-01':(383177,'234071438fd6adba920ff3fefc2422df2af109c6f5c357e55375d70c916b58b8',538,535),
    '2023-07-01':(344041,'564f98d0eb3e6d52616a7599d3829abb552f826e652d9ee7c6eb6659efdd7140',484,478),
    '2023-10-01':(482674,'4730c61357a39f47066a164fb4cd9e594a63d8898db9b9fd78c9f3d4a5cade11',677,674),
}
COUNT_ENDPOINT='https://earthquake.usgs.gov/fdsnws/event/1/count'
BODY_ENDPOINT='https://earthquake.usgs.gov/fdsnws/event/1/query'
FILTERS=dict(contributor='us',eventtype='earthquake',minmagnitude='4.5',reviewstatus='reviewed')
REQUIRED=12
LIMITS=(30,999)


def need(flag,code):
    if not flag:
        raise ValueError(code)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def read_pinned(path,expected):
    with path.open('rb') as stream:
        raw=stream.read(expected['bytes']+1)
    need(pin(raw)==expected,'ORIGINAL_IDENTITY_OR_QUARTER_SOURCE_BYTES_CHANGED')
    return raw


def write(path,body):
    raw=canonical(body)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(raw)


def coordinates(latitude,longitude):
    lat,lon=map(float,(latitude,longitude))
    need(-90<=lat<=90 and -180<=lon<=180,'FINITE_VALID_COORDINATE_IDENTITY')
    return lat,lon


def event(row):
    values=row['values']
    lat,lon=coordinates(values['latitude'],values['longitude'])
    code,stamp=row['id'],row['time_UTC_epoch_ms']
    names=row['aliases']
    need(isinstance(code,str) and code and type(stamp) is int and isinstance(names,list)
         and all(isinstance(name,str) and name for name in names),'EXACT_ID_ALIAS_UTC_METADATA_ONLY')
    return dict(id=code,aliases=sorted({code,*names}),time_UTC_epoch_ms=stamp,
                values=dict(latitude=lat,longitude=lon))


def body_record(feature):
    properties=feature['properties']
    need(feature['geometry']['type']=='Point','ORIGINAL_POINT_GEOMETRY')
    lon,lat,_=feature['geometry']['coordinates']
    raw=properties['ids']
    need(isinstance(raw,str),'ALL_AVAILABLE_SOURCE_ALIAS_IDS_REQUIRED')
    ident=event(dict(id=feature['id'],aliases=[name for name in raw.split(',') if name],
                     time_UTC_epoch_ms=properties['time'],values=dict(latitude=lat,longitude=lon)))
    eligible=properties['net']=='us' and 'us' in properties['sources'].strip(',').split(',')
    if eligible:
        mag=Decimal(str(properties['mag']))
        need(properties['type']=='earthquake' and properties['status']=='reviewed'
             and mag.is_finite() and mag>=Decimal('4.5') and bool(properties['magType']),
             'MATCH_PREVIOUS_PROVENANCE_FILTER')
    return ident,eligible


def grid_id(lat,lon):
    lat,lon=coordinates(lat,lon)
    if lat==90 or lon==180:
        return None
    # Fixed latitude-major index in the original 30 by 60 degree grid.
    lat_index=sum(lat>=boundary for boundary in (-60,-30,0,30,60))
    lon_index=sum(lon>=boundary for boundary in (-120,-60,0,60,120))
    return 6*lat_index+lon_index


def grid_bounds(number):
    lat,lon=divmod(number,6)
    return dict(minlatitude=-90+30*lat,maxlatitude=-60+30*lat,
                minlongitude=-180+60*lon,maxlongitude=-120+60*lon)


def slots():
    return [dict(slot_id=2*(year-2010)+half,
               starttime=f'{year}-{start}T00:00:00Z',endtime_exclusive=f'{year}-{end}T00:00:00Z')
            for year in range(2010,2018) for half,(start,end) in enumerate((('01-05','06-26'),('07-05','12-26')))]


def params(grid,slot):
    geometry=grid_bounds(grid)
    for key in ('maxlatitude','maxlongitude'):
        geometry[key]=format(Decimal(geometry[key])-Decimal('0.000000001'),'f')
    end=datetime.fromisoformat(slot['endtime_exclusive'].replace('Z','+00:00'))-timedelta(milliseconds=1)
    return dict(FILTERS,**geometry,starttime=slot['starttime'],
                endtime=end.isoformat(timespec='milliseconds').replace('+00:00','Z'))


def contains(query,latitude,longitude,epoch_ms):
    lat,lon=coordinates(latitude,longitude)
    bounds=query['half_open_bounds']
    lo,hi=[round(datetime.fromisoformat(query[k].replace('Z','+00:00')).timestamp()*1000)
           for k in ('starttime','endtime_exclusive')]
    return (bounds['minlatitude']<=lat<bounds['maxlatitude'] and
            bounds['minlongitude']<=lon<bounds['maxlongitude'] and
            type(epoch_ms) is int and lo<=epoch_ms<hi)


def plan_queries(ranking):
    selected=sorted(sorted(range(36),key=lambda n:(-ranking[n],n))[:16])
    spans=slots()
    result=[]
    for i,grid in enumerate(selected):
        for slot_id in (i,(i+7)%16):
            span=spans[slot_id]
            query=params(grid,span)
            result.append(dict(query_id=f'G{grid:02d}-S{slot_id:02d}',grid_id=grid,slot_id=slot_id,
                half_open_bounds=grid_bounds(grid),**{k:span[k] for k in ('starttime','endtime_exclusive')},
                query_params=query,url=COUNT_ENDPOINT+'?'+urlencode(dict(query,format='text'))))
    return selected,result


def freeze(output):
    old=json.loads(read_pinned(IDENTITY,IDENTITY_PIN))
    need(isinstance(old['rows'],list) and len(old['rows'])==1488,'EXACT_EXISTING_HISTORICAL_METADATA_ONLY_INVENTORY')
    historic=list(map(event,old['rows']))
    rank=Counter({index:0 for index in range(36)})
    months=[]
    boundary=0
    for key,(size,sha,total,wanted) in MONTHS.items():
        expected=dict(bytes=size,SHA256=sha)
        doc=json.loads(read_pinned(QUARTER/(key+'.geojson'),expected),parse_float=Decimal)
        sequence=doc['features']
        need(doc['type']=='FeatureCollection' and len(sequence)==total
             and len({item['id'] for item in sequence})==total,'ORIGINAL_SNAPSHOT_SHAPE')
        eligible_count=0
        for row in sequence:
            ident,eligible=body_record(row)
            historic.append(ident)
            if eligible:
                eligible_count+=1
                gid=grid_id(ident['values']['latitude'],ident['values']['longitude'])
                if gid is None:boundary+=1
                else:rank[gid]+=1
        need(eligible_count==wanted,'ORIGINAL_FOUR_MONTH_QUALIFIED_COUNT')
        months.append(dict(month=key,body=expected,all_metadata_records=total,eligible_for_design=eligible_count,
                           retained_nonpreferred_for_exclusion=total-eligible_count))
    need(sum(rank.values())+boundary==2326,'FOUR_SAVED_MONTHS_2326_DESIGN_ROWS')
    normalized={canonical(row):row for row in historic}
    records=[normalized[raw] for raw in sorted(normalized)]
    aliases={alias for row in records for alias in row['aliases']}
    output.mkdir(mode=0o700)
    exclusion_pin=write(output/'historical-identities-private.json',records)
    summary=dict(legacy_records=1488,failed_four_month_catalog_records=sum(r['all_metadata_records'] for r in months),
        four_month_eligible_design_records=2326,nonpreferred_four_month_retained=sum(
        r['retained_nonpreferred_for_exclusion'] for r in months),canonical_identity_rows=len(records),
        unique_alias_strings=len(aliases),excluded_half_open_positive_global_edge_design_rows=boundary,
        source_identity=IDENTITY_PIN,failed_four_month_parent_receipt_SHA256=PREVIOUS,months=months,
        historical_other_identity_kinds={key:dict(count=len(old[key]),pin=pin(canonical(old[key])))
                                         for key in ('chart_ids','group_ids','media_SHA256','first_blind_groups')},
        private_identity_index=exclusion_pin,alias_only_index_digest=pin(canonical(sorted(aliases))),
        scope='event ID, aliases, UTC milliseconds and coordinates only',gold_reads=0,HTTP_gets=0)
    write(output/'HISTORIC_IDENTITY_SUMMARY.json',summary)
    selected,queries=plan_queries(rank)
    plan=dict(schema='GEMMA_USGS_CONFIRMATORY96_OFFLINE_COUNT32_CAPACITY_V2',
        endpoint=COUNT_ENDPOINT,official_credit='U.S. Geological Survey',planned_queries=32,qualified_count_min=30,
        qualified_count_max=999,required_distinct_grid_and_slot_count=12,selection_order=[
            'stop unless maximum-cardinality is at least 12','within size 12 maximize minimum count',
            'then maximize sum count','then lexicographically smallest sorted (grid_id, slot_id)'],
        spatial_rule='latitude-major 6 by 6 grid, inclusive minima and strictly exclusive maxima; service COUNT uses a 1e-9 degree inward upper bound',
        temporal_rule='UTC inclusive start; inclusive service end is the exclusive UTC day minus 1 millisecond',
        spatial_rule_body='recheck exact half-open grid from actual feature metadata before body admission',
        raw_history_and_candidate_closure='aliases and 72h/500km transitive closure, then grid joins; duplicate and exposure checks separate',
        no_body_or_gold_in_this_scope=True,count_not_actual_eligible_or_independent_group_evidence=True,
        all_count_rows_terminal_before_matching=True,no_count_retry_or_replacement=True,selected_grids=selected,
        grid_order_histogram=[dict(grid_id=i,design_count=rank[i]) for i in range(36)],
        historical_exclusion_summary=pin(canonical(summary)),queries=queries)
    return dict(plan=write(output/'COUNT32_PLAN.json',plan),historical_identity=pin(canonical(summary)),
                private_exclusion=exclusion_pin,planned=len(queries),selected_grids=selected,
                design_records=2326,all_historical_records=len(historic),HTTP_gets=0,gold_reads=0)


def load_edges(plan,rows):
    wanted=plan['queries']
    need(len(wanted)==len(rows)==32,'ALL_32_ORIGINAL_COUNT_RECEIPTS_REQUIRED')
    mapped={}
    for query,result in zip(wanted,rows,strict=True):
        need(result['query_id']==query['query_id'] and result['url']==query['url']
             and result['status'] in ('COMPLETE','FAILED'),'COUNT_RECEIPT_ORDER_OR_IDENTITY_DIFFERS')
        count=result.get('count')
        if result['status']=='FAILED':
            need(count is None,'FAILED_COUNT_MUST_NOT_SUPPLY_CAPACITY')
            continue
        need(type(count) is int and count>=0,'ACTUAL_COUNT_LITERAL_REQUIRED')
        if LIMITS[0]<=count<=LIMITS[1]:
            mapped[(query['grid_id'],query['slot_id'])]=count
    return mapped


def optim(alternatives,edges,mincount=0,k=None):
    by_grid={grid:[(slot,edges[grid,slot]) for slot in range(16) if (
             grid,slot) in edges and edges[grid,slot]>=mincount] for grid in alternatives}
    states={0:(0,())}
    for grid,choices in by_grid.items():
        following=states.copy()
        for mask,(total,matching) in states.items():
            if k is not None and mask.bit_count()>=k:
                continue
            for slot,count in choices:
                member=1<<slot
                if mask&member:
                    continue
                target=mask|member
                candidate=total+count,matching+((grid,slot),)
                old=following.get(target)
                if old is None or candidate[0]>old[0] or candidate[0]==old[0] and candidate[1]<old[1]:
                    following[target]=candidate
        states=following
    return states


def match(plan,rows):
    edges=load_edges(plan,rows)
    grid=plan['selected_grids']
    maximum=max(mask.bit_count() for mask in optim(grid,edges))
    result=dict(maximum_cardinality=maximum,required=REQUIRED,qualified_queries=len(edges),
                planned=32,terminal_counts=len(rows),count_not_scientific_admission=True,selected=[])
    if maximum<REQUIRED:
        result.update(state='CAPACITY_SHORTFALL_NO_BODY_OR_REFILL')
        return result
    thresholds=sorted(set(edges.values()))
    left,right=0,len(thresholds)-1
    while left<right:
        index=(left+right+1)//2
        enough=any(mask.bit_count()==REQUIRED for mask in optim(grid,edges,thresholds[index],REQUIRED))
        if enough:left=index
        else:right=index-1
    threshold=thresholds[left]
    exact=[choice for mask,choice in optim(grid,edges,threshold,REQUIRED).items() if mask.bit_count()==REQUIRED]
    total,chosen=min(exact,key=lambda choice:(-choice[0],choice[1]))
    source={(x['grid_id'],x['slot_id']):x for x in plan['queries']}
    chosen_rows=[]
    for index,key in enumerate(chosen,1):
        q=source[key]
        chosen_rows.append(dict(order=index,query_id=q['query_id'],grid_id=key[0],slot_id=key[1],count=edges[key],
            half_open_bounds=q['half_open_bounds'],starttime=q['starttime'],endtime_exclusive=q['endtime_exclusive'],
            body_url=BODY_ENDPOINT+'?'+urlencode(dict(q['query_params'],format='geojson',limit=1000,orderby='time-asc',nodata=404))))
    result.update(state='EXACT_12_BODY_CANDIDATES_FROZEN_NO_BODY_GET',minimum_selected=threshold,
                  total_count=total,selected=chosen_rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('freeze')
    a.add_argument('output',type=Path)
    b=sub.add_parser('match')
    b.add_argument('--plan',type=Path,required=True)
    b.add_argument('--plan-sha',required=True)
    b.add_argument('--counts',type=Path,required=True)
    b.add_argument('--counts-sha',required=True)
    b.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='freeze':
        print(json.dumps(freeze(args.output),sort_keys=True))
    else:
        with args.plan.open('rb') as stream:
            raw=stream.read(65537)
        need(pin(raw)['SHA256']==args.plan_sha,'EXACT_PREREGISTERED_CAPACITY_PLAN_SHA256')
        with args.counts.open('rb') as stream:
            counts=stream.read(262145)
        need(0<len(counts)<=262144 and pin(counts)['SHA256']==args.counts_sha,'EXACT_COMPLETED_COUNT_RECEIPT_SHA256')
        result=match(json.loads(raw),json.loads(counts))
        print(json.dumps(dict(match=write(args.output,result),state=result['state'],
                         selected=len(result['selected']),maximum_cardinality=result['maximum_cardinality']),sort_keys=True))


if __name__=='__main__':
    main()
