"""Verify the fixed body12 handoff and qualify groups from saved official catalogs."""

import argparse
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
from datetime import datetime, timezone
from http.client import HTTPException
from types import ModuleType
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent if HERE.parent.name=='local-only' else HERE.parents[2]
COUNT = ROOT/'code/bw1000/gemma-usgs-confirmatory96-count32-v2'
COUNT_RESULT = ROOT/'local-only/gemma-usgs-confirmatory96-count32-6911bef4-original'
HISTORY = ROOT/'local-only/gemma-usgs-confirmatory96-count32-v2/FROZEN/historical-identities-private.json'
LEGACY = ROOT/'local-only/gemma-usgs-confirmatory96-v1-historic-identity/metadata-only-identity.json'
PREREG = ROOT/'coordination/tasks/training-t02-gemma-usgs-confirmatory96-preregistration-v1.json'
METHODS = ROOT/'code/bw1000/training-USGS-multievent-source-v2/prepare.py'
PLAN_SHA = 'ba270eaf01a7ae0d1bca6ae85568d986602e536077c54c0d896dcf354e9ff1db'
COUNT_SHA = 'c998cce43ad8f7c1bf68f8ac5ad33b8859bda9effccd74380d9fe2065feb25a0'
MATCH_SHA = 'dc22d01c93e758de78507a7121cbdc7da07f95cf63dfbc104d7c9f4d2552c48f'
OUTPUT = ROOT/'local-only/gemma-usgs-confirmatory96-body12-dc22d01c93e7-original'
HISTORY_SHA = 'c06a48b3bfbd387062616541a414cfe09d70427a4e3913781651ec5f47eba6e3'
PREREG_SHA = 'b5a9a308dd50c2a940da9f9eb2a6b2bfc23872cecd37ab0b5a07816d6790f1a6'
CAP = 4194304


def need(test, code):
    if not test:
        raise ValueError(code)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, limit, expected=None):
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(0 < len(raw) <= limit and (expected is None or expected == digest(raw)), 'EXACT_PINNED_INPUT')
    return raw


def frozen_module(path, expected):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(read(path, 65536, expected), str(path), 'exec'), m.__dict__)
    return m


C = frozen_module(COUNT/'capacity.py','189f5219247c83d194b33b797eeb447a54511da2c57b16a5366e0db1516ddfe0')
M = frozen_module(METHODS,'c0b063f0b48dbfe24f2546e390e8ff705aa05cd71605f029552346c19f8e0858')


def verify_match(plan, counts, match):
    expected = C.match(plan,counts)
    need(all(r['status']=='COMPLETE' for r in counts) and all(match.get(k)==v for k,v in expected.items())
         and expected['state']=='EXACT_12_BODY_CANDIDATES_FROZEN_NO_BODY_GET', 'EXACT_RECOMPUTED_MATCH')
    rows = expected['selected']
    need(len(rows)==len({r['query_id'] for r in rows})==len({r['grid_id'] for r in rows})
         ==len({r['slot_id'] for r in rows})==12, 'EXACT_TWELVE_DISTINCT_FROZEN_ROUTES')
    need(all(official_route(r) for r in rows),'ALL_TWELVE_ROUTES_MUST_USE_OFFICIAL_ENDPOINT')
    return rows


def history_inputs(plan):
    hist = json.loads(read(HISTORY,510415,HISTORY_SHA))
    legacy = json.loads(read(LEGACY,225451,C.IDENTITY_PIN['SHA256']))
    summary_raw = read(COUNT/'FROZEN/HISTORIC_IDENTITY_SUMMARY.json',65536)
    need(C.pin(summary_raw)==plan['historical_exclusion_summary'],'EXACT_PUBLISHED_HISTORICAL_SUMMARY')
    summary = json.loads(summary_raw)
    need(len(hist)==3832 and len(legacy['rows'])==1488
         and summary['private_identity_index']==dict(bytes=510415,SHA256=HISTORY_SHA)
         and C.pin(C.canonical(sorted({a for r in hist for a in r['aliases']})))==summary['alias_only_index_digest'],
         'COMPLETE_FOUR_MONTH_AND_PRIOR_METADATA_IDENTITY_INDEX')
    prereg = json.loads(read(PREREG,65536,PREREG_SHA))
    need(prereg['acquisition']['body_cap_bytes']==CAP and prereg['selection']['required_groups']==8,
         'FROZEN_BODY_CAP_AND_MINIMUM_GROUPS')
    return hist,legacy,prereg['selection']


def preflight(count_dir=COUNT_RESULT):
    plan = json.loads(read(COUNT/'FROZEN/COUNT32_PLAN.json',65536,PLAN_SHA))
    counts = json.loads(read(count_dir/'COUNTS.json',20514,COUNT_SHA))
    match = json.loads(read(count_dir/'MATCH.json',7318,MATCH_SHA))
    return verify_match(plan,counts,match),*history_inputs(plan)


def strict_object(pairs):
    result = dict(pairs)
    need(len(result)==len(pairs),'DUPLICATE_JSON_PROPERTY')
    return result


def rejects_nonfinite(value):
    raise ValueError('NONFINITE_JSON_CONSTANT')


def body(raw, route):
    data = json.loads(raw,parse_float=Decimal,parse_constant=rejects_nonfinite,object_pairs_hook=strict_object)
    source = data['features']
    need(data['type']=='FeatureCollection' and isinstance(source,list) and 0<len(source)<1000,
         'EMPTY_INVALID_OR_POSSIBLY_TRUNCATED_BODY')
    meta = data['metadata']
    need(meta.get('status')==200 and ('count' not in meta or (type(meta['count']) is int and meta['count']==len(source))),
         'EXACT_BODY_METADATA_STATUS_AND_OPTIONAL_COUNT')
    rows = []
    last = None
    for f in source:
        props = f['properties']
        need(f['type']=='Feature' and isinstance(props['ids'],str) and props['ids'].strip(','), 'IDENTITY_REQUIRED')
        ident, preferred = C.body_record(f)
        lon,lat,depth = f['geometry']['coordinates']
        stamp = ident['time_UTC_epoch_ms']
        need(all(type(v) in (int,Decimal) for v in (lat,lon,depth)) and C.contains(route,lat,lon,stamp)
             and (last is None or last<=stamp), 'SOURCE_HALF_OPEN_BOUNDS_AND_CHRONOLOGY')
        last = stamp
        ident.update(preferred_US=preferred,route=route['query_id'])
        if preferred:
            mag = props['mag']
            need(type(mag) in (int,Decimal) and -100<=depth<=1000 and type(props['updated']) is int
                 and isinstance(props['magType'],str), 'FROZEN_PREFERRED_NUMERIC_PROVENANCE')
            ident.update(preferred_net='us',magType=props['magType'],updated_epoch_ms=props['updated'],
                         values=dict(latitude=str(lat),longitude=str(lon),depth_km=str(depth),magnitude=str(mag)))
        rows.append(ident)
    return rows


def qualify(historic,new,legacy,selection):
    rows = historic + new
    old_end = len(historic)
    parents = list(range(len(rows)))

    def root(i):
        while parents[i]!=i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(i,j):
        a,b = root(i),root(j)
        if a!=b:
            parents[max(a,b)] = min(a,b)

    # Nonpreferred raw records still participate in alias and proximity closure.
    aliases = {}
    body_aliases = defaultdict(set)
    ordered = sorted(range(len(rows)),key=lambda i:rows[i]['time_UTC_epoch_ms'])
    left = 0
    for pos,i in enumerate(ordered):
        r = rows[i]
        for name in r['aliases']:
            join(aliases.setdefault(name,i),i)
            if i>=old_end:
                body_aliases[name].add(i)
        while r['time_UTC_epoch_ms']-rows[ordered[left]]['time_UTC_epoch_ms']>72*3600000:
            left += 1
        for j in ordered[left:pos]:
            if root(i)!=root(j) and M.distance(r,rows[j])<=500:
                join(i,j)
    old_roots = {root(i) for i in range(old_end)}
    repeated = {root(i) for tied in body_aliases.values() if len(tied)>1 for i in tied}
    before_roots = [root(i) for i in range(old_end,len(rows))]
    clean = [i for i in range(old_end,len(rows)) if root(i) not in old_roots|repeated]
    cells = {}
    for i in clean:
        r=rows[i]
        grid=C.grid_id(r['values']['latitude'],r['values']['longitude'])
        need(grid is not None,'EXACT_HALF_OPEN_GROUPING')
        join(cells.setdefault(grid,i),i)
    groups = defaultdict(list)
    for i in clean:
        if rows[i]['preferred_US']:
            groups[root(i)].append(rows[i])
    found = []
    for group in groups.values():
        ordered = sorted(group,key=lambda r:(r['time_UTC_epoch_ms'],r['id']))
        gid = digest(M.canonical(sorted(r['id'] for r in group)))
        windows = []
        if gid in legacy['group_ids']:
            continue
        for start in range(0,len(group)-5,6):
            window = ordered[start:start+6]
            wid = digest(M.canonical([r['id'] for r in window]))
            if wid in legacy['chart_ids'] or any(len({M.display(r['values'][key]) for r in window})<2
                                                  for key in ('magnitude','depth_km')):
                continue
            windows.append(dict(id=wid,group=gid,rows=window))
        if len(windows)>=2:
            found.append((digest((selection['seed']+gid).encode()),gid,windows))
    chosen = [w for _,_,windows in sorted(found)[:8] for w in windows[:2]] if len(found)>=8 else []
    if chosen:
        need(len(chosen)==16 and len({r['id'] for w in chosen for r in w['rows']})==96,
             'EXACT_PRELABEL_SELECTION_DENOMINATORS')
    return chosen,dict(raw_events=len(new),preferred_US=sum(r['preferred_US'] for r in new),
        prior_closure_excluded=sum(root_i in old_roots for root_i in before_roots),
        duplicate_alias_closure_excluded=sum(root_i in repeated for root_i in before_roots),
        distinct_body_primary_ids=len({r['id'] for r in new}),new_eligible_groups=len(groups),
        qualified_groups=len(found),required_groups=8,selected_charts=len(chosen))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_raw(path, raw):
    import os
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return C.pin(raw)


def approval(path, expected, rows):
    value = json.loads(read(path,16384,expected))
    need(value.get('body12_GO') is True and value.get('no_gold') is True
         and value.get('COUNT32_PLAN_SHA256')==PLAN_SHA and value.get('COUNTS_SHA256')==COUNT_SHA
         and value.get('MATCH_SHA256')==MATCH_SHA and value.get('historical_index_SHA256')==HISTORY_SHA
         and value.get('ordered_routes_SHA256')==digest(C.canonical(rows))
         and value.get('qualification_source_SHA256')==digest(Path(__file__).read_bytes())
         and re.fullmatch('[0-9a-f]{40}', value.get('published_commit','')) is not None,
         'ROOT_EXACT_PUBLISHED_BODY12_AUTHORITY_REQUIRED')
    return value


def official_route(route):
    parsed = urlsplit(route['body_url'])
    return (re.fullmatch('G[0-9]{2}-S[0-9]{2}',route['query_id']) is not None
            and not parsed.fragment and parsed._replace(query='').geturl()==C.BODY_ENDPOINT)


def one_route(route, output, opener):
    route_id,url = route['query_id'],route['body_url']
    need(official_route(route),'OFFICIAL_EXACT_BODY_ENDPOINT')
    report = dict(query_id=route_id,url=url,planned_count=route['count'],start_UTC=utc(),request_attempts=1)
    C.write(output/(route_id+'-intent.json'),report)
    rows = []
    try:
        request = Request(url,headers={'Accept':'application/geo+json, application/json'},method='GET')
        with opener.open(request,timeout=14) as response:
            raw = response.read(CAP+1)
            report.update(http_status=response.status,content_type=response.headers.get_content_type(),
                          server_date=response.headers.get('Date'),received_bytes=len(raw))
            need(response.geturl()==url and response.status==200 and 0<len(raw)<=CAP,
                 'EXACT_BODY_URL_STATUS_AND_EXTENT')
        report['body'] = write_raw(output/(route_id+'.geojson'),raw)
        rows = body(raw,route)
        report.update(status='COMPLETE',body_rows=len(rows),body_minus_prior_count=len(rows)-route['count'],
                      preferred_US=sum(r['preferred_US'] for r in rows))
    except (HTTPException,OSError,ValueError,KeyError,TypeError,ArithmeticError) as exc:
        report.update(status='FAILED',error_type=type(exc).__name__)
        if isinstance(exc,HTTPError):
            report['http_status']=exc.code
            exc.close()
        if type(exc) is ValueError and re.fullmatch('[A-Z0-9_]{2,80}',str(exc)):
            report['gate']=str(exc)
    report['finish_UTC']=utc()
    C.write(output/(route_id+'-receipt.json'),report)
    return rows,report


def collect(rows, historic, legacy, selection, authority, output, opener):
    output.mkdir(mode=0o700)
    C.write(output/'RUN_INTENT.json',dict(routes=rows,publication=authority['published_commit'],
        historical_index_SHA256=HISTORY_SHA,source_SHA256=authority['qualification_source_SHA256'],
        policy='one attempt per frozen route; no replacement, retry or resume; incomplete body set stops selection'))
    new,reports = [],[]
    for route in rows:
        additions,observation = one_route(route,output,opener)
        new.extend(additions)
        reports.append(observation)
    report = dict(planned=12,terminal=len(reports),completed=sum(r['status']=='COMPLETE' for r in reports),
        observed_features=sum(r.get('body_rows',0) for r in reports),gold_reads=0,gold_generated=0,
        predictions=0,formal_delivery_increment=0,qualified_groups=None,selected_charts=0,
        body_receipts=C.write(output/'BODY_RECEIPTS.json',reports))
    if report['completed']!=12:
        report['state'] = 'INCOMPLETE_BODY12_STOP_NO_REPLACEMENT'
    else:
        windows,detail = qualify(historic,new,legacy,selection)
        report.update(detail)
        if windows:
            report.update(state='EIGHT_GROUPS_PRELABEL_SELECTION_PENDING_MEDIA_CROSSCHECK',
                          selection=C.write(output/'selection-private.json',windows))
        else:
            report['state'] = 'FEWER_THAN_EIGHT_GROUPS_STOP_NO_REPLACEMENT'
    return {'receipt':C.write(output/'QUALIFICATION.json',report),**report}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('preflight','collect'))
    p.add_argument('--count-dir',type=Path,default=COUNT_RESULT)
    p.add_argument('--authority',type=Path)
    p.add_argument('--authority-sha')
    args=p.parse_args()
    rows,historic,legacy,selection=preflight(args.count_dir)
    if args.mode=='preflight':
        result=dict(planned=12,ordered_routes_SHA256=digest(C.canonical(rows)),historical_records=len(historic),
                    history_SHA256=HISTORY_SHA,actual_body_gets=0,gold_reads=0,qualified_groups=None,
                    output_path=OUTPUT.relative_to(ROOT).as_posix())
    else:
        need(args.authority and args.authority_sha,'ROOT_AUTHORITY_REQUIRED')
        authority=approval(args.authority,args.authority_sha,rows)
        result=collect(rows,historic,legacy,selection,authority,OUTPUT,build_opener(NoRedirect))
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
