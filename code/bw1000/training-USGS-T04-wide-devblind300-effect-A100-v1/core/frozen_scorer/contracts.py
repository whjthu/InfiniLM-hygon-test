"""Deterministic grouping before questions, display-derived targets; no I/O or network."""
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
import math


def need(ok,message):
    if not ok:raise ValueError(message)

def canonical(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def display(x):
    v=Decimal(x);need(v.is_finite(),'Nonfinite display')
    r=v.quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    return format(Decimal('0.00') if r==0 else r,'.2f')

def distance(a,b):
    lat1,lat2=map(math.radians,[float(a['values']['latitude']),float(b['values']['latitude'])])
    dlat=lat2-lat1;dlon=math.radians(float(b['values']['longitude'])-float(a['values']['longitude']))
    h=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088*2*math.asin(min(1,math.sqrt(max(0,h))))

def make_plan(rows,plan):
    need(len(rows)==plan['source_event_denominator'] and len({r['id'] for r in rows})==len(rows),'Source denominator/IDs differ')
    need(all(r['status']=='QUALIFIED_SOURCE_NUMERIC_RECORD' and not r['errors'] and not r['null_fields'] for r in rows),'Unqualified source slot retained; refuse build, no substitution')
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[max(a,b)]=min(a,b)
    edges=[]
    for i,a in enumerate(rows):
        need(a['source_geo_region_key'] is not None,'Missing source region')
        for j in range(i):
            b=rows[j]
            reasons=[]
            if set(a['aliases'])&set(b['aliases']):reasons.append('same_alias')
            if a['source_geo_region_key']==b['source_geo_region_key']:reasons.append('same_coarse_region')
            if abs(a['time_UTC_epoch_ms']-b['time_UTC_epoch_ms'])<=72*3600000 and distance(a,b)<=500:reasons.append('near_72h_500km')
            if reasons:union(i,j);edges.append(dict(a=a['id'],b=b['id'],reasons=reasons))
    groups={}
    for i,r in enumerate(rows):groups.setdefault(find(i),[]).append(r)
    ordered=[]
    for members in groups.values():
        members=sorted(members,key=lambda r:(r['time_UTC_epoch_ms'],r['id']));gid=sha(canonical(sorted(r['id'] for r in members)))
        ordered.append(dict(group=gid,members=members,order=sha((plan['seed']+gid).encode())))
    ordered.sort(key=lambda g:(g['order'],g['group']));n=len(ordered);need(n>=3,'Insufficient isolated groups: no split or model qualification')
    train=max(1,int(n*.6));dev=max(1,(n-train)//2);need(train+dev<n,'Blind group missing')
    charts=[];event_split={}
    for number,g in enumerate(ordered):
        split='train' if number<train else ('dev' if number<train+dev else 'blind')
        g['split']=split
        for r in g['members']:event_split[r['id']]=split
        for offset in range(0,len(g['members']),5):
            window=g['members'][offset:offset+5];cid='usgs:'+g['group']+':'+str(offset//5)
            charts.append(dict(id=cid,group=g['group'],split=split,rows=window,source_slots=[r['slot'] for r in window]))
    need(len(charts)<=plan['charts_max'] and len(charts)*5<=plan['QA_max'],'Planned output cap exceeded; no sample dropping')
    need(len({i for c in charts for i in c['source_slots']})==len(rows),'Source slot coverage differs')
    need(all(event_split[e['a']]==event_split[e['b']] for e in edges),'Closure crosses split')
    return dict(source_slots=len(rows),source_alias_groups=len({r['event_alias_group'] for r in rows}),source_geographic_groups=len({r['source_geo_region_key'] for r in rows}),
        closure_groups=n,groups=[dict(group=g['group'],split=g['split'],event_ids=[r['id'] for r in g['members']]) for g in ordered],
        edges=edges,event_split=event_split,charts=charts,planned_charts=len(charts),planned_QA=len(charts)*5,
        split_group_counts={s:sum(g['split']==s for g in ordered) for s in ['train','dev','blind']})

def displayed_rows(chart):
    return [dict(label='E'+str(i+1).zfill(2),date_UTC=datetime.fromtimestamp(r['time_UTC_epoch_ms']/1000,timezone.utc).strftime('%m-%d %H:%M:%S'),
        reported_magnitude=display(r['values']['magnitude']),magType=r['magType'],depth_km=display(r['values']['depth_km'])) for i,r in enumerate(chart['rows'])]

def questions(chart):
    rows=displayed_rows(chart);first,last=rows[0],rows[-1]
    depths=[Decimal(r['depth_km']) for r in rows];mags=[Decimal(r['reported_magnitude']) for r in rows]
    definitions=[('lookup_reported_magnitude','What is the reported magnitude number for E01? Answer with exactly two decimal places.',first['reported_magnitude'],'decimal2'),
        ('lookup_depth','What is the depth in km for E01? Answer with exactly two decimal places.',first['depth_km'],'decimal2'),
        ('depth_change','Compute the last listed row depth minus E01 depth, in km. Use the displayed numbers and answer with exactly two decimal places, with a minus sign if negative.',display(depths[-1]-depths[0]),'decimal2'),
        ('maximum_depth','Which event label(s) have the largest displayed depth in km? Include every tie, separated by one space in table order.',' '.join(r['label'] for r,d in zip(rows,depths) if d==max(depths)),'labels'),
        ('maximum_reported_number','Which event label(s) have the largest displayed reported magnitude NUMBER? Compare displayed numbers only, not physical energy or risk. Include every tie, separated by one space in table order.',' '.join(r['label'] for r,m in zip(rows,mags) if m==max(mags)),'labels')]
    return [dict(id=chart['id']+':'+str(i),chart_id=chart['id'],group=chart['group'],kind=kind,question=q,answer=a,answer_type=t,
        display_rows_SHA256=sha(canonical(rows)),single_event_window=len(rows)==1) for i,(kind,q,a,t) in enumerate(definitions)]
