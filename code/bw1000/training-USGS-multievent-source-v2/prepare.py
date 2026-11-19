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
    need(obj.get('type')=='FeatureCollection' and isinstance(features,list) and 0<len(features)<p['limit_stop_at'],'empty/limit-truncated source: STOP')
    if 'count' in obj.get('metadata',{}):need(obj['metadata']['count']==len(features),'metadata count differs')
    rows=[];rejected=[];ids=set()
    lo=int(datetime(2026,6,1,tzinfo=timezone.utc).timestamp()*1000);hi=int(datetime(2026,7,1,tzinfo=timezone.utc).timestamp()*1000)
    for slot,f in enumerate(features):
        try:
            q=f['properties'];xyz=f['geometry']['coordinates'];eid=f['id']
            need(type(eid)is str and eid and eid not in ids,'source ID duplicate/missing');ids.add(eid)
            need(f['geometry']['type']=='Point' and len(xyz)==3 and q['type']=='earthquake' and q['net']=='us' and q['status']=='reviewed','source contributor/type/status differs')
            need(type(q['time'])is int and lo<=q['time']<hi and type(q['updated'])is int,'fixed UTC timestamp/window differs')
            vals=dict(magnitude=Decimal(str(q['mag'])),longitude=Decimal(str(xyz[0])),latitude=Decimal(str(xyz[1])),depth_km=Decimal(str(xyz[2])))
            need(all(v.is_finite() for v in vals.values()) and vals['magnitude']>=Decimal('4.5') and -90<=vals['latitude']<=90 and -180<=vals['longitude']<=180 and -100<=vals['depth_km']<=1000,'numeric bounds/null/nonfinite')
            aliases=sorted(set([eid]+[x for x in q['ids'].split(',') if x]));need(bool(q['magType']),'missing magnitude type')
            rows.append(dict(id=eid,aliases=aliases,slot=slot,time_UTC_epoch_ms=q['time'],updated_epoch_ms=q['updated'],magType=q['magType'],values={k:str(v) for k,v in vals.items()}))
        except (ValueError,TypeError,KeyError,ArithmeticError) as e:rejected.append(dict(slot=slot,error_type=type(e).__name__))
    need([r['time_UTC_epoch_ms'] for r in rows]==sorted(r['time_UTC_epoch_ms'] for r in rows),'response not time ascending')
    return rows,dict(source_features=len(features),numeric_qualified=len(rows),rejected=rejected,metadata=obj.get('metadata',{}),source_SHA256=sha(raw))

def select(old,new,p):
    rows=old+new;nold=len(old);parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[max(a,b)]=min(a,b)
    for i,a in enumerate(rows):
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
    split_sizes=[n//2,(n-n//2)//2];candidates={s:[] for s in ('train','dev','blind')};rejects=[]
    for i,g in enumerate(ordered):
        split='train' if i<split_sizes[0] else ('dev' if i<sum(split_sizes) else 'blind');g['split']=split
        for start in range(0,len(g['rows']),6):
            window=g['rows'][start:start+6];reason=None
            if len(window)<4:reason='SHORT_INPUT_WINDOW'
            elif len({display(r['values']['magnitude']) for r in window})<2 or len({display(r['values']['depth_km']) for r in window})<2:reason='CONSTANT_DISPLAYED_INPUT_COLUMN'
            if reason:rejects.append(dict(group=g['group'],offset=start,event_count=len(window),reason=reason));continue
            wid=sha(canonical([r['id'] for r in window]));candidates[split].append(dict(id=wid,group=g['group'],split=split,rows=window,order=sha((p['seed']+wid).encode())))
    selected=[];missing={};coverage={}
    for split,quota in p['chart_slots'].items():
        pool=sorted(candidates[split],key=lambda c:(c['order'],c['id']));bygroup={}
        for c in pool:bygroup.setdefault(c['group'],[]).append(c)
        take=[]
        while len(take)<quota and any(bygroup.values()):
            for group in sorted(bygroup):
                if bygroup[group] and len(take)<quota:take.append(bygroup[group].pop(0))
        selected+=take;missing[split]=quota-len(take);coverage[split]=len({c['group'] for c in take})
    ok=n>=8 and not any(missing.values()) and coverage['dev']>=2 and coverage['blind']>=2
    return dict(passed=ok,state='INPUT_WINDOWS_FROZEN_PENDING_RENDER_NATIVE_QA' if ok else 'INSUFFICIENT_FIXED_WINDOWS_STOP_NO_REFILL',
        old_metadata_events=nold,new_numeric_events=len(new),excluded_old_closure_events=excluded,new_internal_closure_groups=n,
        planned_charts=p['chart_slots'],selected_charts={s:sum(c['split']==s for c in selected) for s in candidates},missing_charts=missing,selected_group_counts=coverage,
        singleton_charts=0,constant_display_columns_charts=0,input_window_rejections=rejects,charts=selected,
        new_recipe_not_legacy_permanent_coarse_isolation=True,absolute_leakage_absence_proven=False)

def main(a):
    p=json.loads((Path(__file__).parent/'PLAN.json').read_bytes());need(p['root_GO'] is p['CPU_GO'] is True,'source-only not adopted')
    # Root freezes this metadata-only inventory before acquisition; no gold/image/prediction paths.
    inv=json.loads(a.old_inventory.read_bytes());expected={s['path']:s for s in p['old_sources']}
    need(len(inv)==3 and {s['path'] for s in inv}==set(expected),'exact distinct original three source metadata files required')
    need(all(type(s['bytes']) is int and type(s['SHA256']) is str and len(s['SHA256'])==64 for s in expected.values()),'old inventory unresolved: freeze pins before execution')
    old=[]
    for spec in inv:
        need(spec==expected[spec['path']],'old inventory differs from published identity')
        source=a.repo/spec['path'];raw=source.read_bytes();need(len(raw)==spec['bytes']<=4*1024**2 and sha(raw)==spec['SHA256'],'old source metadata pin')
        old+=json.loads(raw)['rows']
    old=list({canonical({k:r[k] for k in ('id','aliases','time_UTC_epoch_ms','values')}):r for r in old}.values())
    raw=a.source.read_bytes();new,audit=parse(raw,p);selection=select(old,new,p)
    if audit['rejected']:selection['passed']=False;selection['state']='SOURCE_INVALID_SLOTS_RETAINED_STOP'
    a.output.mkdir(mode=0o700)
    for name,v in [('source-audit-private.json',audit),('selection-private.json',selection)]:
        with (a.output/name).open('xb') as stream:stream.write(canonical(v))
    public={k:v for k,v in selection.items() if k not in ('charts','input_window_rejections')};public['rejected_source_slots']=len(audit['rejected']);public['source_SHA256']=sha(raw)
    print(json.dumps(public,sort_keys=True));return 0 if selection['passed'] else 1

if __name__=='__main__':
    q=argparse.ArgumentParser()
    for name in ('source','old-inventory','repo','output'):q.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(main(q.parse_args()))
