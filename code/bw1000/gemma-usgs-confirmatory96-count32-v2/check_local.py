"""Validate the actual frozen corpus projection and matching without HTTP."""
from datetime import datetime
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import capacity as c
import collect_counts as collector


def rejected(code,fn,*args):
    try:fn(*args)
    except ValueError as error:assert str(error)==code,(str(error),code)
    else:raise AssertionError(code)


def plan_boundary(plan):
    q=plan['queries']
    assert len(q)==len({x['query_id'] for x in q})==len({x['url'] for x in q})==32
    assert plan['selected_grids']==sorted(plan['selected_grids']) and len(set(plan['selected_grids']))==16
    for i,grid in enumerate(plan['selected_grids']):
        assert [x['slot_id'] for x in q if x['grid_id']==grid]==[i,(i+7)%16]
    assert sum(x['design_count'] for x in plan['grid_order_histogram'])==2326
    assert c.plan_queries(dict.fromkeys(range(36),0))[0]==list(range(16))
    assert c.slots()[0]['starttime']=='2010-01-05T00:00:00Z'
    assert c.slots()[0]['endtime_exclusive']=='2010-06-26T00:00:00Z'
    assert c.slots()[15]['endtime_exclusive']=='2017-12-26T00:00:00Z'
    assert c.grid_id(0,0)==21 and c.grid_id(0,-0.00001)==20
    assert c.grid_id(-90,-180)==0 and c.grid_id(90,0) is None and c.grid_id(0,180) is None
    for x in q:
        pv=urlparse(x['url'])
        params=parse_qs(pv.query)
        assert pv.netloc=='earthquake.usgs.gov' and pv.path=='/fdsnws/event/1/count' and params['format']==['text']
        assert params['endtime'][0].endswith('23:59:59.999Z')
        bounds=x['half_open_bounds']
        lo,hi=[round(datetime.fromisoformat(x[key].replace('Z','+00:00')).timestamp()*1000)
               for key in ('starttime','endtime_exclusive')]
        lat,lon=(bounds['minlatitude'],bounds['minlongitude'])
        assert c.contains(x,lat,lon,lo) and c.contains(x,lat,lon,hi-1)
        assert not c.contains(x,lat,lon,hi) and not c.contains(x,lat,lon,lo-1)
        assert not c.contains(x,bounds['maxlatitude'],lon,lo)
        assert not c.contains(x,lat,bounds['maxlongitude'],lo)


def test_match(plan):
    qs=plan['queries']
    def rows(counts):
        return [dict(query_id=q['query_id'],url=q['url'],status='FAILED' if count is None else 'COMPLETE',count=count)
                for q,count in zip(qs,counts,strict=True)]
    rejected('ALL_32_ORIGINAL_COUNT_RECEIPTS_REQUIRED',c.match,plan,rows([35]*32)[:-1])
    malformed=rows([35]*32)
    malformed[0]['query_id']=malformed[1]['query_id']
    rejected('COUNT_RECEIPT_ORDER_OR_IDENTITY_DIFFERS',c.match,plan,malformed)
    unavailable=c.match(plan,rows([29,1000]*16))
    assert unavailable['maximum_cardinality']==0 and unavailable['selected']==[]
    all_equal=c.match(plan,rows([30]*32))
    assert all_equal['maximum_cardinality']==16 and all_equal['minimum_selected']==30 and all_equal['total_count']==360
    assert len({r['grid_id'] for r in all_equal['selected']})==len({r['slot_id'] for r in all_equal['selected']})==12
    assert all(urlparse(r['body_url']).path=='/fdsnws/event/1/query' for r in all_equal['selected'])
    highest=c.match(plan,rows([999]*24+[None]*8))
    assert highest['maximum_cardinality']==12 and highest['minimum_selected']==999
    # An exhaustive, independent small case exercises sum and lexicographic tie-breaks.
    grids=plan['selected_grids'][:5]
    edges={(row['grid_id'],row['slot_id']):(67 if n in (1,4,7) else 44) for n,row in enumerate(qs[:10])}
    cases=[]
    for candidates in product(*(tuple(key for key in edges if key[0]==grid)+(None,) for grid in grids)):
        items=tuple(x for x in candidates if x is not None)
        if len(items)==3 and len(set(x[1] for x in items))==3:
            cases.append((min(edges[x] for x in items),sum(edges[x] for x in items),items))
    expected=min(cases,key=lambda x:(-x[0],-x[1],x[2]))
    values=[edges.get((q['grid_id'],q['slot_id'])) for q in qs]
    with patch.object(c,'REQUIRED',3):
        actual=c.match(plan,rows(values))
    assert (actual['minimum_selected'],actual['total_count'],tuple((row['grid_id'],row['slot_id'])
                                                                  for row in actual['selected']))==expected


def test_collector(plan_bytes,plan,folder):
    for raw,count in ((b'0',0),(b'999\n',999),(b'30\r\n',30)):
        assert collector.parse_count(raw)==count
    for bad in (b'<html>0',b'1 2',b'03',b'-1',b'3.0',b''):
        rejected('STRICT_COUNT_TEXT_RESPONSE',collector.parse_count,bad)
    authority=c.canonical(dict(count32_GO=True,COUNT32_PLAN_SHA256=c.pin(plan_bytes)['SHA256'],
                               no_body_no_gold=True,published_commit='a'*40))
    check,author=collector.plan_and_authority(plan_bytes,authority,c.pin(authority)['SHA256'])
    assert check==plan
    order=[]
    class Reply:
        status=200
        headers=SimpleNamespace(get_content_type=lambda:'text/plain')
        def __init__(self,raw):self.raw=raw
        def read(self,cap):return self.raw[:cap]
        def __enter__(self):return self
        def __exit__(self,*args):return False
    class OfflineOpener:
        def open(self,request,timeout):
            order.append(request.full_url)
            assert timeout==12 and request.get_method()=='GET' and '/count?' in request.full_url
            if len(order)==2:
                raise OSError('fixture no transport')
            return Reply(b'bad count' if len(order)==3 else b'999\n')
    result=collector.collect(check,author,folder,OfflineOpener())
    assert order==[x['url'] for x in plan['queries']] and result['terminal']==32 and result['completed']==30
    if folder.exists():
        try:collector.collect(check,author,folder,OfflineOpener())
        except FileExistsError:pass
        else:raise AssertionError('whole batch replay accepted')
    assert len(order)==32


if __name__=='__main__':
    frozen=c.D/'FROZEN'
    raw=(frozen/'COUNT32_PLAN.json').read_bytes()
    plan=json.loads(raw)
    with TemporaryDirectory() as root:
        folder=Path(root)
        outcome=c.freeze(folder/'rebuild')
        assert all((folder/'rebuild'/x.name).read_bytes()==x.read_bytes() for x in frozen.iterdir() if x.is_file())
        identity=json.loads((frozen/'HISTORIC_IDENTITY_SUMMARY.json').read_bytes())
        assert identity['legacy_records']==1488 and identity['failed_four_month_catalog_records']==2344
        assert identity['nonpreferred_four_month_retained']==18 and identity['canonical_identity_rows']>2326
        plan_boundary(plan)
        test_match(plan)
        test_collector(raw,plan,folder/'count_fixture')
    print(json.dumps(dict(outcome,checks=['actual_saved_raw_and_identity_byte_rebuild','32_frozen_URLs_and_half_open_boundaries',
                           'exact_12_match_vs_independent_exhaustive_optimizer','failed_and_out_of_range_counts',
                           'mocked_all_32_count_gets_once_and_plan_authority_gate'],
                           actual_HTTP_gets=0,actual_catalog_body_gets=0,actual_gold_reads=0),sort_keys=True))
