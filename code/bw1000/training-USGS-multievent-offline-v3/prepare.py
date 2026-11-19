"""Offline source validation and pre-QA multievent selection; never reads old gold."""
import argparse
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path

def need(ok,why):
    if not ok:raise ValueError(why)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def display(x):return format(Decimal(x).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)+Decimal('0'),'.2f')
def distance(a,b):
    la,lb=map(math.radians,[float(a['values']['latitude']),float(b['values']['latitude'])])
    dl=math.radians(float(a['values']['longitude'])-float(b['values']['longitude']))
    h=math.sin((la-lb)/2)**2+math.cos(la)*math.cos(lb)*math.sin(dl/2)**2
    return 12742.0176*math.asin(min(1,math.sqrt(max(0,h))))
def cell(row):return (math.floor((float(row['values']['latitude'])+90)/30),math.floor((float(row['values']['longitude'])+180)/60))

def parse(raw,p):
    need(len(raw)<=p['body_cap_bytes'],'body cap')
    obj=json.loads(raw,parse_float=Decimal);features=obj.get('features')
    need(len(features)==p['source_event_denominator'],'all711 frozen source slots required')
    need(obj.get('type')=='FeatureCollection' and isinstance(features,list) and 0<len(features)<p['limit_stop_at'],'empty/limit-truncated source: STOP')
    if 'count' in obj.get('metadata',{}):need(obj['metadata']['count']==len(features),'metadata count differs')
    rows=[];rejected=[];ids=set()
    lo=int(datetime(2026,6,1,tzinfo=timezone.utc).timestamp()*1000);hi=int(datetime(2026,7,1,tzinfo=timezone.utc).timestamp()*1000)
    for slot,f in enumerate(features):
        try:
            q=f['properties'];xyz=f['geometry']['coordinates'];eid=f['id']
            need(type(eid)is str and eid and eid not in ids,'source ID duplicate/missing');ids.add(eid)
            need(f['geometry']['type']=='Point' and len(xyz)==3 and q['type']=='earthquake' and type(q.get('net')) is str and bool(q['net']) and type(q.get('sources')) is str and 'us' in q['sources'].strip(',').split(',') and q['status']=='reviewed','source contributor/type/status differs')
            need(type(q['time'])is int and lo<=q['time']<hi and type(q['updated'])is int,'fixed UTC timestamp/window differs')
            vals=dict(magnitude=Decimal(str(q['mag'])),longitude=Decimal(str(xyz[0])),latitude=Decimal(str(xyz[1])),depth_km=Decimal(str(xyz[2])))
            need(all(v.is_finite() for v in vals.values()) and vals['magnitude']>=Decimal('4.5') and -90<=vals['latitude']<=90 and -180<=vals['longitude']<=180 and -100<=vals['depth_km']<=1000,'numeric bounds/null/nonfinite')
            aliases=sorted(set([eid]+[x for x in q['ids'].split(',') if x]));need(bool(q['magType']),'missing magnitude type')
            rows.append(dict(id=eid,aliases=aliases,slot=slot,time_UTC_epoch_ms=q['time'],updated_epoch_ms=q['updated'],magType=q['magType'],preferred_net=q['net'],sources=sorted(set(x for x in q['sources'].split(',') if x)),values={k:str(v) for k,v in vals.items()}))
        except (ValueError,TypeError,KeyError,ArithmeticError) as e:rejected.append(dict(slot=slot,error_type=type(e).__name__,reason=str(e)[:128]))
    need([r['time_UTC_epoch_ms'] for r in rows]==sorted(r['time_UTC_epoch_ms'] for r in rows),'response not time ascending')
    return rows,dict(source_features=len(features),numeric_qualified=len(rows),rejected=rejected,metadata=obj.get('metadata',{}),source_SHA256=sha(raw))

def select(old,new,p,check=lambda:None):
    rows=old+new;nold=len(old);parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[max(a,b)]=min(a,b)
    for i,a in enumerate(rows):
        if i%32==0:check()
        for j,b in enumerate(rows[:i]):
            if set(a['aliases'])&set(b['aliases']) or (abs(a['time_UTC_epoch_ms']-b['time_UTC_epoch_ms'])<=72*3600000 and distance(a,b)<=500):union(i,j)
    oldroots={find(i) for i in range(nold)};eligible=[i for i in range(nold,len(rows)) if find(i) not in oldroots]
    excluded=len(new)-len(eligible)
    # Extra geographical closure is within NEW data only. It never redefines the old benchmark.
    for x,i in enumerate(eligible):
        for j in eligible[:x]:
            if cell(rows[i])==cell(rows[j]):union(i,j)
    groups={}
    for i in eligible:groups.setdefault(find(i),[]).append(rows[i])
    ordered=[]
    for members in groups.values():
        gid=sha(canonical(sorted(r['id'] for r in members)))
        ordered.append(dict(group=gid,order=sha((p['seed']+gid).encode()),rows=sorted(members,key=lambda r:(r['time_UTC_epoch_ms'],r['id']))))
    ordered.sort(key=lambda g:(g['order'],g['group']));n=len(ordered)
    rejects=[]
    for g in ordered:
        g['windows']=[]
        for start in range(0,len(g['rows']),6):
            window=g['rows'][start:start+6];reason=None
            if len(window)<4:reason='SHORT_INPUT_WINDOW'
            elif len({display(r['values']['magnitude']) for r in window})<2 or len({display(r['values']['depth_km']) for r in window})<2:reason='CONSTANT_DISPLAYED_INPUT_COLUMN'
            if reason:rejects.append(dict(group=g['group'],offset=start,event_count=len(window),reason=reason));continue
            wid=sha(canonical([r['id'] for r in window]))
            g['windows'].append(dict(id=wid,group=g['group'],rows=window,order=sha((p['seed']+wid).encode())))
        g['windows'].sort(key=lambda c:(c['order'],c['id']))
    active=[g for g in ordered if g['windows']]
    capacities=[len(g['windows']) for g in active]
    assignment,dp=allocate(capacities,[12,6,6],p['DP_max_states'],check)
    names=['train','dev','blind'];selected=[];coverage={s:0 for s in names};missing=dict(p['chart_slots'])
    if assignment is not None:
        for g,i in zip(active,assignment):g['split']=names[i]
        for split in names:
            bygroup={g['group']:list(g['windows']) for g in active if g['split']==split}
            take=[];quota=p['chart_slots'][split]
            while len(take)<quota and any(bygroup.values()):
                for gid in sorted(bygroup):
                    if bygroup[gid] and len(take)<quota:
                        c=dict(bygroup[gid].pop(0),split=split);take.append(c)
            selected+=take;missing[split]=quota-len(take);coverage[split]=len({c['group'] for c in take})
    ok=assignment is not None and not any(missing.values()) and all(v>=2 for v in coverage.values())
    return dict(passed=ok,state='INPUT_SELECTION_ONLY_FROZEN_NO_QA_OR_IMAGES' if ok else 'NO_FEASIBLE_FIXED_SPLIT_STOP',
        old_metadata_events=nold,new_numeric_events=len(new),excluded_old_closure_events=excluded,new_internal_closure_groups=n,
        eligible_window_groups=len(active),capacities=[dict(group=g['group'],windows=len(g['windows']),split=g.get('split')) for g in ordered],
        DP=dp,planned_charts=p['chart_slots'],selected_charts={s:sum(c['split']==s for c in selected) for s in names},
        missing_charts=missing,selected_group_counts=coverage,input_window_rejections=rejects,charts=selected,
        singleton_charts=0,constant_display_columns_charts=0,actual_QA=0,actual_images=0,actual_native_calls=0,actual_model_calls=0,
        new_recipe_not_legacy_permanent_coarse_isolation=True,absolute_leakage_absence_proven=False)


def allocate(capacities,quotas,max_states,check=lambda:None):
    """Boolean memo DP: at most (N+1)*13*7*7*3^3 capped states, no 3^N search."""
    from functools import lru_cache
    states=0;n=len(capacities)
    @lru_cache(None)
    def feasible(i,a,b,c,ga,gb,gc):
        nonlocal states
        states+=1
        if states%256==0:check()
        need(states<=max_states,'DP_STATE_BOUND_STOP_NO_ALTERNATE_SEARCH')
        caps=(a,b,c);groups=(ga,gb,gc)
        if all(caps[k]>=quotas[k] and groups[k]>=2 for k in range(3)):return True
        if i==n or sum(capacities[i:])<sum(max(0,quotas[k]-caps[k]) for k in range(3)):return False
        for target in range(3):
            cc=list(caps);gg=list(groups);cc[target]=min(quotas[target],cc[target]+capacities[i]);gg[target]=min(2,gg[target]+1)
            if feasible(i+1,*cc,*gg):return True
        return False
    ok=feasible(0,0,0,0,0,0,0);assignment=[];caps=[0,0,0];groups=[0,0,0]
    if ok:
        for i,size in enumerate(capacities):
            check()
            for target in range(3):
                cc=list(caps);gg=list(groups);cc[target]=min(quotas[target],cc[target]+size);gg[target]=min(2,gg[target]+1)
                if feasible(i+1,*cc,*gg):assignment.append(target);caps,groups=cc,gg;break
            else:raise ValueError('DP_RECONSTRUCTION_INCONSISTENT')
    return assignment if ok else None,dict(states=states,state_bound=max_states,feasible=ok,quota_caps=quotas,group_count_caps=[2,2,2],source_group_order_fixed=True)
