"""Combined old/new alias+geographic/time closure before QA; no I/O/model/answers."""
from chart_contracts import canonical,sha,distance,need


def select(old,new,plan):
    need(len(old)==50 and 0<len(new)<=200,'FULL_OLD50_AND_NEW_SOURCE_DENOMINATOR')
    rows=old+new
    need(all(r['status']=='QUALIFIED_SOURCE_NUMERIC_RECORD' and not r['errors'] and not r['null_fields'] for r in rows),'UNQUALIFIED_SOURCE_REFUSES_SELECTION_NO_SUBSTITUTION')
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[max(a,b)]=min(a,b)
    edges=[]
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j];why=[]
            if set(a['aliases'])&set(b['aliases']):why.append('alias')
            if a['source_geo_region_key']==b['source_geo_region_key']:why.append('coarse_region')
            if abs(a['time_UTC_epoch_ms']-b['time_UTC_epoch_ms'])<=72*3600000 and distance(a,b)<=500:why.append('near72h500km')
            if why:union(i,j);edges.append(dict(a=i,b=j,reasons=why))
    groups={}
    for i,row in enumerate(rows):groups.setdefault(find(i),[]).append(i)
    eligible=[];excluded=[]
    for indexes in groups.values():
        fresh=[i-50 for i in indexes if i>=50]
        if not fresh:continue
        if any(i<50 for i in indexes):excluded.extend(fresh);continue
        members=sorted((new[i] for i in fresh),key=lambda r:(r['time_UTC_epoch_ms'],r['id']))
        gid=sha(canonical(sorted(r['id'] for r in members)))
        windows=[dict(id='usgs-confirm:'+gid+':'+str(k//5),group=gid,split='confirmation',rows=members[k:k+5]) for k in range(0,len(members),5)]
        eligible.append(dict(group=gid,order=sha((plan['seed']+gid).encode()),windows=windows))
    eligible.sort(key=lambda g:(g['order'],g['group']));charts=[];round_index=0
    while len(charts)<5 and any(round_index<len(g['windows']) for g in eligible):
        for group in eligible:
            if round_index<len(group['windows']):charts.append(group['windows'][round_index])
            if len(charts)==5:break
        round_index+=1
    enough=len(charts)==5 and len({c['group'] for c in charts})>=3
    if not enough:charts=[]
    selected=[r['id'] for c in charts for r in c['rows']]
    need(len(selected)==len(set(selected)),'NO_SELECTED_EVENT_OR_WINDOW_DUPLICATION')
    return dict(passed=enough,old_slots=50,new_slots=len(new),closure_edges=edges,excluded_new_slots=sorted(excluded),
        excluded_count=len(excluded),eligible_groups=len(eligible),selected_group_count=len({c['group'] for c in charts}),
        planned_chart_slots=5,planned_QA_slots=25,charts=charts,selected_event_count=len(selected),
        state='FIVE_CHARTS_AT_LEAST_THREE_NEW_COMPONENTS' if enough else 'INSUFFICIENT_NEW_DISJOINT_COMPONENTS_STOP_NO_BACKFILL')
