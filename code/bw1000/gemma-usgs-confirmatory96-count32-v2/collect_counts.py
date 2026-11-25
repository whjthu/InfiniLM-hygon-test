"""ROOT-approved post-publication collector for the exact 32 official COUNT URLs."""
import argparse
from datetime import datetime, timezone
from http.client import HTTPException
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from capacity import COUNT_ENDPOINT, need, pin, plan_queries, write


EXPECTED='ba270eaf01a7ae0d1bca6ae85568d986602e536077c54c0d896dcf354e9ff1db'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def parse_count(raw):
    need(1<=len(raw)<=32 and re.fullmatch(rb'(?:0|[1-9][0-9]{0,15})\r?\n?',raw) is not None,
         'STRICT_COUNT_TEXT_RESPONSE')
    return int(raw.strip())


def plan_and_authority(plan_raw,authority_raw,authority_sha):
    need(pin(plan_raw)['SHA256']==EXPECTED and pin(authority_raw)['SHA256']==authority_sha,
         'PUBLISHED_EXACT_PLAN_AND_ROOT_APPROVAL_SHA')
    plan=json.loads(plan_raw)
    authority=json.loads(authority_raw)
    need(authority['count32_GO'] is True and authority['COUNT32_PLAN_SHA256']==EXPECTED
         and authority['no_body_no_gold'] is True
         and re.fullmatch('[0-9a-f]{40}',authority['published_commit']) is not None,
         'EXACT_ROOT_POST_PUBLICATION_COUNT_APPROVAL')
    ranking={row['grid_id']:row['design_count'] for row in plan['grid_order_histogram']}
    selected,queries=plan_queries(ranking)
    need(len(queries)==32 and selected==plan['selected_grids'] and queries==plan['queries']
         and all(q['url']==COUNT_ENDPOINT+'?'+urlencode(dict(q['query_params'],format='text')) for q in queries),
         'CANONICAL_OFFICIAL_COUNT_ROUTES_ONLY')
    return plan,authority


def utc():return datetime.now(timezone.utc).isoformat()


def collect(plan,authority,output,opener):
    queries=plan['queries']
    output.mkdir(mode=0o700)
    write(output/'RUN_INTENT.json',dict(request_scope='GEMMA_USGS_CONFIRMATORY96_COUNT_ONLY',
         published_commit=authority['published_commit'],plan_SHA256=EXPECTED,planned=32,query_limit_per_row=1,
         all_queries=[dict(query_id=row['query_id'],url=row['url']) for row in queries],body_gets=0,gold_reads=0))
    receipts=[]
    for number,rule in enumerate(queries,1):
        state=dict(query_id=rule['query_id'],url=rule['url'],status='FAILED',count=None,scheduled_order=number,
                   started_UTC=utc(),request_attempts=1)
        write(output/(rule['query_id']+'-attempt.json'),state)
        try:
            req=Request(rule['url'],headers={'Accept':'text/plain'},method='GET')
            with opener.open(req,timeout=12) as response:
                raw=response.read(33)
                state.update(http_status=response.status,content_type=response.headers.get_content_type(),body_pin=pin(raw))
            need(state['http_status']==200 and state['content_type']=='text/plain','EXPECTED_HTTP_200_TEXT_COUNT_ONLY')
            state.update(status='COMPLETE',count=parse_count(raw))
        except (HTTPError,HTTPException,OSError,ValueError) as exc:
            state['error_type']=type(exc).__name__
            if isinstance(exc,HTTPError):
                state['http_status']=exc.code
                exc.close()
        state['finished_UTC']=utc()
        write(output/(rule['query_id']+'-receipt.json'),state)
        receipts.append(state)
    return dict(receipts=write(output/'COUNTS.json',receipts),planned=32,terminal=len(receipts),
                completed=sum(value['status']=='COMPLETE' for value in receipts),event_body_gets=0,gold_reads=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--authority',required=True,type=Path)
    p.add_argument('--authority-sha',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--emit-only',action='store_true')
    a=p.parse_args()
    with a.plan.open('rb') as stream:
        plan_raw=stream.read(64<<10)
    with a.authority.open('rb') as stream:
        authority_raw=stream.read(16<<10)
    plan,authority=plan_and_authority(plan_raw,authority_raw,a.authority_sha)
    if a.emit_only:
        out=dict(planned=32,HTTP_gets=0,body_gets=0,gold_reads=0,publication=authority['published_commit'])
    else:
        out=collect(plan,authority,a.output,build_opener(NoRedirect))
    print(json.dumps(out,sort_keys=True))


if __name__=='__main__':
    main()
