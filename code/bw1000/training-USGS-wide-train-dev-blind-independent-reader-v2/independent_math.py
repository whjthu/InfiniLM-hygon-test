"""Independent contractual reconstruction; no producer imports, files or rendering."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP


def require(value, code):
    if not value:
        raise ValueError(code)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def decode(raw, decimals=False):
    def pairs(items):
        require(len({k for k, _ in items}) == len(items), 'DUPLICATE_JSON_KEY')
        return dict(items)
    def invalid(_):
        raise ValueError('NONFINITE_JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid,
                      parse_float=Decimal if decimals else float)


def source_rows(raw, plan):
    data = decode(raw, True)
    require(data['type'] == 'FeatureCollection', 'GEOJSON_COLLECTION')
    meta, features = data['metadata'], data['features']
    require(type(meta['status']) is int and meta['status'] == 200
            and len(features) == 500 and ('count' not in meta or (type(meta['count']) is int and meta['count']==500)), 'SOURCE500_STATUS')
    require(type(meta['generated']) is int and meta['generated'] > 0
            and isinstance(meta['api'], str) and bool(meta['api']), 'SOURCE_VINTAGE')
    start = int(datetime.fromisoformat(plan['fixed_start_UTC']).timestamp() * 1000)
    end = int(datetime.fromisoformat(plan['fixed_end_UTC']).timestamp() * 1000)
    seen, aliases_seen, records = set(), set(), []
    previous = None
    for ordinal, feature in enumerate(features):
        identifier, props, geometry = feature['id'], feature['properties'], feature['geometry']
        require(isinstance(identifier, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identifier)
                and identifier not in seen, 'EVENT_ID_UNIQUE')
        seen.add(identifier)
        require(feature['type'] == 'Feature' and geometry['type'] == 'Point'
                and props['type'] == 'earthquake' and len(geometry['coordinates']) == 3,
                'SOURCE_FEATURE_KIND')
        lon, lat, depth = geometry['coordinates']
        values = {}
        for key, value, low, high in (('magnitude', props['mag'], '4.5', '20'),
                ('longitude', lon, '-180', '180'), ('latitude', lat, '-90', '90'),
                ('depth_km', depth, '-100', '1000')):
            require(type(value) in (int, Decimal) and Decimal(value).is_finite()
                    and Decimal(low) <= Decimal(value) <= Decimal(high), 'SOURCE_FINITE_RANGE')
            values[key] = str(value)
        timestamp, updated = props['time'], props['updated']
        require(type(timestamp) is int and start <= timestamp <= end
                and (previous is None or previous <= timestamp)
                and type(updated) is int and updated >= timestamp, 'SOURCE_TIME_ORDER_WINDOW')
        previous = timestamp
        require(isinstance(props['magType'], str) and bool(props['magType']), 'SOURCE_MAGTYPE')
        require(isinstance(props['ids'], str), 'SOURCE_ALIASES')
        aliases = [s for s in props['ids'].split(',') if s]
        require(identifier in aliases and all(re.fullmatch(r'[A-Za-z0-9_-]{1,128}', s)
                for s in aliases) and not aliases_seen.intersection(aliases), 'SOURCE_ALIAS_UNIQUE_FEATURE')
        aliases_seen.update(aliases)
        cell = 'lat30_' + str(min(5, int((Decimal(values['latitude']) + 90) // 30)))
        cell += '_lon60_' + str(min(5, int((Decimal(values['longitude']) + 180) // 60)))
        records.append(dict(slot=ordinal, id=identifier, aliases=aliases,
            source_net=props.get('net'), source_networks=props.get('sources'),
            time_UTC_epoch_ms=timestamp, updated_UTC_epoch_ms=updated, magType=props['magType'],
            review_status=props.get('status'), values=values, source_geo_region_key=cell,
            null_fields=[], errors=[], event_alias_group=min(aliases), status='QUALIFIED_SOURCE_NUMERIC_RECORD'))
    return records, meta


def distance(a, b):
    x1, x2 = (math.radians(float(r['values']['latitude'])) for r in (a, b))
    y = math.radians(float(a['values']['longitude']) - float(b['values']['longitude']))
    t = math.sin((x1-x2)/2)**2 + math.cos(x1)*math.cos(x2)*math.sin(y/2)**2
    t = min(1.0, max(0.0, t))
    return 12742.0176 * math.atan2(math.sqrt(t), math.sqrt(1-t))


def reconstruct(old, new, plan, history, check):
    require(len(old) == 250 and len(new) == 500 and len(history) == 2724, 'SOURCE_HISTORY_DENOMINATORS')
    records = old + new
    parent = list(range(750))
    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    edges, historical = [], []
    for i, a in enumerate(records):
        check()
        require(a['status'] == 'QUALIFIED_SOURCE_NUMERIC_RECORD' and not a['errors']
                and not a['null_fields'], 'OLD_AND_NEW_QUALIFIED_SLOTS')
        for j in range(i):
            b = records[j]
            alias = bool(set(a['aliases']).intersection(b['aliases']))
            near = False
            if abs(a['time_UTC_epoch_ms']-b['time_UTC_epoch_ms']) <= 259200000:
                km = distance(a, b)
                require(abs(km-500) > 1e-7, 'DISTANCE_THRESHOLD_NUMERIC_AMBIGUITY')
                near = km <= 500
            reasons = (['alias'] if alias else []) + (['near72h500km'] if near else [])
            if i < 250:
                old_reasons = (['alias'] if alias else [])
                if a['source_geo_region_key'] == b['source_geo_region_key']:
                    old_reasons.append('coarse_region')
                if near:
                    old_reasons.append('near72h500km')
                if old_reasons:
                    historical.append(dict(a=i, b=j, reasons=old_reasons))
            if reasons:
                edges.append(dict(a=i, b=j, reasons=reasons))
                join(i, j)
    require(historical == history, 'INDEPENDENT_HISTORICAL2724_EDGES')
    require(len(edges) <= plan['maximum_saved_closure_edges'], 'STRONG_EDGE_CAP')
    components = {}
    for i in range(750):
        components.setdefault(root(i), []).append(i)
    excluded, eligible = [], []
    for indexes in components.values():
        fresh = [i-250 for i in indexes if i >= 250]
        if not fresh:
            continue
        if any(i < 250 for i in indexes):
            excluded += fresh
            continue
        members = sorted((new[i] for i in fresh), key=lambda r:(r['time_UTC_epoch_ms'], r['id']))
        group = digest(canonical(sorted(r['id'] for r in members)))
        eligible.append(dict(group=group, order=digest((plan['seed']+group).encode()),
                             stratum=min(r['source_geo_region_key'] for r in members), members=members))
    eligible.sort(key=lambda g:(g['stratum'], g['order'], g['group']))
    splits = ('train', 'dev', 'blind')
    totals = dict.fromkeys(splits, 0)
    counts, allocation, assigned = {}, [], {s:[] for s in splits}
    for group in eligible:
        cell = group['stratum']; counts.setdefault(cell, dict.fromkeys(splits, 0))
        ranking = [(counts[cell][s], totals[s], digest((plan['seed']+'|'+cell+'|'+group['group']+'|'+s).encode()), s) for s in splits]
        split = sorted(ranking)[0][-1]
        counts[cell][split] += 1; totals[split] += 1
        windows = [dict(id='usgs-wide:'+group['group']+':'+str(k//5), group=group['group'], split=split,
                        rows=group['members'][k:k+5]) for k in range(0,len(group['members']),5)]
        assigned[split].append(dict(group=group['group'], order=group['order'], windows=windows))
        allocation.append(dict(group=group['group'], stratum=cell, split=split,
                               events=len(group['members']), windows=len(windows)))
    charts, split_counts, extra_train = [], {}, []
    for split in splits:
        groups = sorted(assigned[split], key=lambda g:(g['order'],g['group']))
        flattened = [g['windows'][round_index] for round_index in range(max((len(g['windows']) for g in groups), default=0))
                     for g in groups if round_index < len(g['windows'])]
        selected = flattened[:10]
        require(len(selected)==10 and len({c['group'] for c in selected})>=3, 'NO_INSUFFICIENT_SPLIT_BACKFILL')
        charts += selected
        split_counts[split] = dict(charts=10, QA=50, groups=len({c['group'] for c in selected}))
        if split == 'train':
            chosen = {c['id'] for c in selected}
            extra_train = [dict(group=g['group'], unselected_windows=sum(w['id'] not in chosen for w in g['windows']))
                           for g in groups if any(w['id'] not in chosen for w in g['windows'])]
    event_ids = [r['id'] for c in charts for r in c['rows']]
    require(len(event_ids)==len(set(event_ids)) and len(eligible)>=9, 'SELECTED_EVENT_DUPLICATE_OR_GROUP_SHORTFALL')
    old_cells={r['source_geo_region_key'] for r in old}; new_cells={r['source_geo_region_key'] for r in new}
    return dict(passed=True, old_slots=250, new_slots=500, historical_closure_pin=plan['prior_sources']['closure'],
        prior_history_closure_edges=2724, prior_strong_edges=sum(e['a']<250 for e in edges),
        closure_definition=plan['strong_closure_policy'], closure_edges=edges, excluded_new_slots=sorted(excluded),
        excluded_count=len(excluded), eligible_groups=len(eligible), whole_group_allocation=allocation,
        stratum_group_counts=counts, geographic_overlap_allowed=True, geographic_OOD=False,
        old_coarse_cells=len(old_cells), new_coarse_cells=len(new_cells), overlap_coarse_cells=len(old_cells&new_cells),
        selected_group_count=len({c['group'] for c in charts}), selected_event_count=len(event_ids),
        planned_chart_slots=30, planned_QA_slots=150, split_counts=split_counts, charts=charts,
        additional_train_candidates=extra_train, additional_train_data_generated=False,
        state='TRAIN10_DEV10_BLIND10_AT_LEAST3_STRONG_GROUPS_EACH')


def decimal2(value):
    value = Decimal(value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return '0.00' if value == 0 else format(value, '.2f')


def chart_targets(chart):
    shown=[]
    for i, row in enumerate(chart['rows']):
        shown.append(dict(label=f'E{i+1:02d}',date_UTC=datetime.fromtimestamp(row['time_UTC_epoch_ms']/1000,timezone.utc).strftime('%m-%d %H:%M:%S'),
                          reported_magnitude=decimal2(row['values']['magnitude']), magType=row['magType'],depth_km=decimal2(row['values']['depth_km'])))
    d=[Decimal(r['depth_km']) for r in shown]; m=[Decimal(r['reported_magnitude']) for r in shown]
    answers=[shown[0]['reported_magnitude'],shown[0]['depth_km'],decimal2(d[-1]-d[0]),
             ' '.join(shown[i]['label'] for i in range(len(d)) if d[i]==max(d)),
             ' '.join(shown[i]['label'] for i in range(len(m)) if m[i]==max(m))]
    kinds=['lookup_reported_magnitude','lookup_depth','depth_change','maximum_depth','maximum_reported_number']
    questions=[
        'What is the reported magnitude number for E01? Answer with exactly two decimal places.',
        'What is the depth in km for E01? Answer with exactly two decimal places.',
        'Compute the last listed row depth minus E01 depth, in km. Use the displayed numbers and answer with exactly two decimal places, with a minus sign if negative.',
        'Which event label(s) have the largest displayed depth in km? Include every tie, separated by one space in table order.',
        'Which event label(s) have the largest displayed reported magnitude NUMBER? Compare displayed numbers only, not physical energy or risk. Include every tie, separated by one space in table order.']
    targets=[dict(id=chart['id']+':'+str(k),chart_id=chart['id'],group=chart['group'],kind=kinds[k],question=questions[k],
                  answer=answers[k],answer_type='decimal2' if k<3 else 'labels',display_rows_SHA256=digest(canonical(shown)),single_event_window=len(shown)==1)
             for k in range(5)]
    return shown, targets
