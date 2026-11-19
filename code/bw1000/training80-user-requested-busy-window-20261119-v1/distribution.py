"""Card-window distributions retain missing denominators and task assignments."""
import math
import statistics
from qualified_sampler import POINTS, INTERVAL_S, MAX_LAG_S, require


def stats(rows,planned):
    means=[r['window_mean_busy_percent'] for r in rows]
    return dict(planned_cards=planned,observed_cards=len(rows),missing_cards=planned-len(rows),
                window_mean_min=min(means) if means else None,
                window_mean_median=statistics.median(means) if means else None,
                window_mean_max=max(means) if means else None,
                observed_card_window_mean=sum(means)/len(means) if means else None,
                over80_cards=sum(x>80 for x in means),zero_cards=sum(x==0 for x in means),
                over80_threshold_strict=True,driver_counter_not_SM_occupancy=True)


def describe(owners,pins,observed,start):
    require(set(observed)<=set(owners),'UNKNOWN_NODE')
    cards=[];planned_groups={};groups={};categories={};planned_categories={}
    for node,(_,group) in owners.items():
        planned_groups[group]=planned_groups.get(group,0)+8
        category={'T':'training','I':'inference','O':'operator'}[group[0]]
        planned_categories[category]=planned_categories.get(category,0)+8
        groups.setdefault(group,[]);categories.setdefault(category,[])
        if node not in observed:
            continue
        row=observed[node]
        require(len(row['rows'])==POINTS and len(pins[node])==len(set(pins[node]))==8,'CARD_WINDOW_COVERAGE')
        for i,point in enumerate(row['rows']):
            planned=start+i*INTERVAL_S
            require(point['point']==i and planned<=point['opened_epoch']<=point['closed_epoch']<=planned+MAX_LAG_S,
                    'CARD_WINDOW_CLOCK')
            require(set(point['busy'])==set(pins[node]) and all(type(v)is int and 0<=v<=100 for v in point['busy'].values()),'CARD_WINDOW_VALUES')
        for pci in pins[node]:
            values=[p['busy'][pci] for p in row['rows']]
            card=dict(node=node,PCI=pci,task=group,points=POINTS,window_mean_busy_percent=sum(values)/POINTS,
                      point_min=min(values),point_median=statistics.median(values),point_max=max(values))
            require(math.isfinite(card['window_mean_busy_percent']),'CARD_WINDOW_FINITE')
            cards.append(card);groups[group].append(card);categories[category].append(card)
    return dict(per_card_private=cards,overall=stats(cards,len(owners)*8),
                tasks={g:stats(groups[g],n) for g,n in sorted(planned_groups.items())},
                partitions={g:stats(categories[g],n) for g,n in sorted(planned_categories.items())},
                scope='Three sysfs driver samples per card in one4s window; no training/busy causality or SMoccupancy')
