"""V2 preregistration: event-related strong closure, coarse geography only stratifies whole groups."""
from chart_contracts import canonical,sha,distance,need
SPLITS=('train','dev','blind')

def select(old,new,plan,prior_edges):
    need(len(old)==250 and 0<len(new)<=500 and len(prior_edges)==2724,'OLD250_NEW500_ORIGINAL2724')
    rows=old+new;need(all(r['status']=='QUALIFIED_SOURCE_NUMERIC_RECORD' and not r['errors'] and not r['null_fields'] for r in rows),'INVALID_SOURCE_SLOT_NO_SUBSTITUTION')
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[max(a,b)]=min(a,b)
    # The full2724-edge history remains unchanged at its pinned original path.
    # Coarse-only history edges do not enter this new V2 strong graph.
    edges=[]
    for e in prior_edges:
        reasons=[r for r in e['reasons'] if r in ('alias','near72h500km')]
        if reasons:edges.append(dict(a=e['a'],b=e['b'],reasons=reasons));union(e['a'],e['b'])
    prior_strong_edges=len(edges)
    for i in range(250,len(rows)):
        a=rows[i]
        for k in range(i):
            b=rows[k];why=[]
            if set(a['aliases'])&set(b['aliases']):why.append('alias')
            if abs(a['time_UTC_epoch_ms']-b['time_UTC_epoch_ms'])<=72*3600000 and distance(a,b)<=500:why.append('near72h500km')
            if why:
                need(len(edges)<plan['maximum_saved_closure_edges'],'BOUNDED_STRONG_EDGE_CAP_NO_RELAXATION')
                union(i,k);edges.append(dict(a=i,b=k,reasons=why))
    groups={}
    for i in range(len(rows)):groups.setdefault(find(i),[]).append(i)
    eligible=[];excluded=[]
    for indexes in groups.values():
        fresh=[i-250 for i in indexes if i>=250]
        if not fresh:continue
        if any(i<250 for i in indexes):excluded.extend(fresh);continue
        members=sorted((new[i] for i in fresh),key=lambda r:(r['time_UTC_epoch_ms'],r['id']))
        gid=sha(canonical(sorted(r['id'] for r in members)))
        eligible.append(dict(group=gid,order=sha((plan['seed']+gid).encode()),stratum=min(r['source_geo_region_key'] for r in members),members=members))
    eligible.sort(key=lambda g:(g['stratum'],g['order'],g['group']))
    assignments={s:[] for s in SPLITS};counts={};totals={s:0 for s in SPLITS};allocation=[]
    for g in eligible:
        stratum=g['stratum'];counts.setdefault(stratum,{s:0 for s in SPLITS})
        split=min(SPLITS,key=lambda s:(counts[stratum][s],totals[s],sha((plan['seed']+'|'+stratum+'|'+g['group']+'|'+s).encode()),s))
        counts[stratum][split]+=1;totals[split]+=1
        windows=[dict(id='usgs-wide:'+g['group']+':'+str(k//5),group=g['group'],split=split,rows=g['members'][k:k+5]) for k in range(0,len(g['members']),5)]
        assignments[split].append(dict(group=g['group'],order=g['order'],windows=windows))
        allocation.append(dict(group=g['group'],stratum=stratum,split=split,events=len(g['members']),windows=len(windows)))
    charts=[];split_counts={};extra_train=[]
    for split in SPLITS:
        groups_for_split=sorted(assignments[split],key=lambda g:(g['order'],g['group']))
        selected=[];round_index=0
        while len(selected)<10 and any(round_index<len(g['windows']) for g in groups_for_split):
            for g in groups_for_split:
                if round_index<len(g['windows']):selected.append(g['windows'][round_index])
                if len(selected)==10:break
            round_index+=1
        split_counts[split]=dict(charts=len(selected),QA=5*len(selected),groups=len({c['group'] for c in selected}))
        charts.extend(selected)
        if split=='train':
            chosen={c['id'] for c in selected}
            extra_train=[dict(group=g['group'],unselected_windows=sum(w['id'] not in chosen for w in g['windows'])) for g in groups_for_split if any(w['id'] not in chosen for w in g['windows'])]
    passed=len(eligible)>=9 and all(v['charts']==10 and v['QA']==50 and v['groups']>=3 for v in split_counts.values())
    selected_ids=[r['id'] for c in charts for r in c['rows']]
    need(len(selected_ids)==len(set(selected_ids)),'NO_SELECTED_EVENT_OR_WINDOW_DUPLICATION')
    group_sets={s:{c['group'] for c in charts if c['split']==s} for s in SPLITS}
    need(all(not (group_sets[a]&group_sets[b]) for i,a in enumerate(SPLITS) for b in SPLITS[:i]),'WHOLE_STRONG_COMPONENT_SPLITS_REQUIRED')
    old_cells={r['source_geo_region_key'] for r in old};new_cells={r['source_geo_region_key'] for r in new}
    return dict(passed=passed,old_slots=250,new_slots=len(new),historical_closure_pin=plan['prior_sources']['closure'],prior_history_closure_edges=2724,prior_strong_edges=prior_strong_edges,closure_definition=plan['strong_closure_policy'],closure_edges=edges,
        excluded_new_slots=sorted(excluded),excluded_count=len(excluded),eligible_groups=len(eligible),whole_group_allocation=allocation,stratum_group_counts=counts,
        geographic_overlap_allowed=True,geographic_OOD=False,old_coarse_cells=len(old_cells),new_coarse_cells=len(new_cells),overlap_coarse_cells=len(old_cells&new_cells),
        selected_group_count=len({c['group'] for c in charts}),selected_event_count=len(selected_ids),planned_chart_slots=30,planned_QA_slots=150,split_counts=split_counts,charts=charts,
        additional_train_candidates=extra_train,additional_train_data_generated=False,
        state='TRAIN10_DEV10_BLIND10_AT_LEAST3_STRONG_GROUPS_EACH' if passed else 'INSUFFICIENT_FROZEN_THREE_SPLIT_WIDTH_STOP_NO_SWAP_OR_BACKFILL')
