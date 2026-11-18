"""One fixed historical USGS GeoJSON sample; no charts/QA/model or secondary HTTP."""
from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import re
import ssl
import urllib.error
import urllib.request


def need(ok,why):
    if not ok:raise ValueError(why)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def js(raw):
    def pairs(items):need(len(items)==len({k for k,v in items}),'Duplicate JSON key');return dict(items)
    def constant(value):raise ValueError('Nonfinite JSON constant')
    return json.loads(raw,parse_float=Decimal,object_pairs_hook=pairs,parse_constant=constant)


def contract_gate(c):
    expected='https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2026-09-01T00:00:00Z&endtime=2026-09-30T23:59:59Z&minmagnitude=4.5&eventtype=earthquake&orderby=time-asc&limit=200&nodata=404'
    need(c['URL']==expected and c['query_URL_previously_requested'] is False and c['GETs']==1
        and c['maximum_features']==200 and c['body_cap_bytes']==1048576 and c['rolling_feed'] is False,'Exact new fixed history scope only')


def headers_gate(status,headers,c):
    need(status==200 and headers.get_content_type()=='application/json' and headers.get('Content-Encoding','identity').lower()=='identity','Only plain200 JSON source')
    sizes=headers.get_all('Content-Length',[]);need(len(sizes)<=1,'Duplicate Content-Length');size=None
    if sizes:need(sizes[0].isdigit(),'Invalid Content-Length');size=int(sizes[0]);need(0<size<=c['body_cap_bytes'],'Body cap before read')
    return size


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Redirect forbidden; no fallback source')


def get_once(path,c,remaining,progress):
    contract_gate(c)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    request=urllib.request.Request(c['URL'],headers={'Accept':'application/json','Accept-Encoding':'identity','User-Agent':'bounded-fixed-history-confirmation-source-v1'})
    progress('GET.enter',dict(attempt=1,URL_SHA256=sha(c['URL'].encode())))
    try:response=opener.open(request,timeout=min(45,remaining()))
    except urllib.error.HTTPError as error:
        progress('GET.http_error',dict(status=error.code,body_read=False));error.close();raise
    with response as r:
        progress('GET.headers_observed',dict(status=r.status,Content_Type=r.headers.get('Content-Type'),Content_Length=r.headers.get_all('Content-Length',[])))
        need(r.url==c['URL'],'Response URL changed');size=headers_gate(r.status,r.headers,c)
        progress('GET.headers_qualified',dict(status=200,declared_length=size))
        body=bytearray()
        with path.open('xb') as stream:
            while True:
                remaining();need(len(body)<c['body_cap_bytes'],'Cap reached without EOF')
                chunk=r.read(min(65536,c['body_cap_bytes']-len(body)))
                if not chunk:break
                body.extend(chunk);stream.write(chunk);stream.flush();progress('GET.body',dict(bytes=len(body)))
                if size is not None and len(body)==size:break
        need(body and (size is None or len(body)==size),'Incomplete source body')
    return bytes(body)
