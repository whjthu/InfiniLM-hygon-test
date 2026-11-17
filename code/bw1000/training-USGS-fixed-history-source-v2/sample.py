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
    expected='https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2026-10-01T00:00:00Z&endtime=2026-10-08T00:00:00Z&minmagnitude=4.5&eventtype=earthquake&orderby=time-asc&limit=50&nodata=404'
    need(c['URL']==expected and c['query_URL_previously_requested'] is False and c['GETs']==1
        and c['maximum_features']==50 and c['body_cap_bytes']==1048576 and c['rolling_feed'] is False,'Exact new fixed history scope only')


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
    request=urllib.request.Request(c['URL'],headers={'Accept':'application/json','Accept-Encoding':'identity','User-Agent':'bounded-fixed-history-science-sample-v1'})
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


def audit(payload,c):
    need(isinstance(payload,dict) and payload['type']=='FeatureCollection','Not official GeoJSON collection')
    meta=payload['metadata'];features=payload['features']
    need(type(meta['status']) is int and meta['status']==200 and type(meta['count']) is int and isinstance(features,list)
        and 0<=len(features)==meta['count']<=50,'Response feature denominator/status differs')
    need(type(meta['generated']) is int and meta['generated']>0 and isinstance(meta['api'],str) and meta['api'],'Source vintage/API identity absent')
    start=int(datetime.fromisoformat(c['fixed_start_UTC']).timestamp()*1000);end=int(datetime.fromisoformat(c['fixed_end_UTC']).timestamp()*1000)
    rows=[];ids=set();parents={};prior=None
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    for slot,f in enumerate(features):
        identifier=f['id'];need(isinstance(identifier,str) and re.fullmatch('[A-Za-z0-9_-]{1,128}',identifier) is not None and identifier not in ids,'Missing/duplicate event ID')
        ids.add(identifier);find(identifier);p=f['properties'];g=f['geometry'];errors=[]
        values={};null=[]
        if f.get('type')!='Feature' or g.get('type')!='Point' or p.get('type')!='earthquake':errors.append('wrong_feature_or_event_type')
        coordinates=g.get('coordinates');coordinates=coordinates if isinstance(coordinates,list) and len(coordinates)==3 else [None]*3
        for name,value,low,high in [('magnitude',p.get('mag'),Decimal('4.5'),Decimal('20')),
            ('longitude',coordinates[0],Decimal('-180'),Decimal('180')),('latitude',coordinates[1],Decimal('-90'),Decimal('90')),
            ('depth_km',coordinates[2],Decimal('-100'),Decimal('1000'))]:
            if value is None:null.append(name);values[name]=None
            elif type(value) not in (int,Decimal) or not Decimal(value).is_finite() or not low<=Decimal(value)<=high:errors.append('invalid_'+name);values[name]=None
            else:values[name]=str(value) # exact source decimal, no float round-trip, no imputation
        t=p.get('time');updated=p.get('updated')
        if type(t) is not int or not start<=t<=end:errors.append('time_not_UTC_epoch_ms_in_fixed_window')
        else:
            if prior is not None and t<prior:errors.append('not_time_ascending')
            prior=t
        if type(updated) is not int or type(t) is not int or updated<t:errors.append('invalid_updated_epoch_ms')
        mag_type=p.get('magType');need(isinstance(mag_type,(str,type(None))),'Invalid magnitude type field')
        if not mag_type:null.append('magType')
        aliases=p.get('ids');alias_list=[x for x in aliases.split(',') if x] if isinstance(aliases,str) else []
        if not alias_list or identifier not in alias_list:errors.append('alias_identity_incomplete')
        for alias in alias_list:
            need(re.fullmatch('[A-Za-z0-9_-]{1,128}',alias) is not None,'Invalid alias identity');union(identifier,alias)
        region=None
        if values['longitude'] is not None and values['latitude'] is not None:
            lat,lon=Decimal(values['latitude']),Decimal(values['longitude'])
            region='lat30_'+str(min(5,int((lat+90)//30)))+'_lon60_'+str(min(5,int((lon+180)//60)))
        rows.append(dict(slot=slot,id=identifier,aliases=alias_list,source_net=p.get('net'),source_networks=p.get('sources'),
            time_UTC_epoch_ms=str(t) if isinstance(t,Decimal) else t,updated_UTC_epoch_ms=str(updated) if isinstance(updated,Decimal) else updated,magType=mag_type,review_status=p.get('status'),values=values,
            source_geo_region_key=region,null_fields=null,errors=errors))
    components={}
    for row in rows:row['event_alias_group']=find(row['id']);components.setdefault(row['event_alias_group'],[]).append(row)
    for group in components.values():
        if len(group)>1:
            for row in group:row['errors'].append('multiple_feature_slots_same_alias_event')
    for row in rows:row['status']='QUALIFIED_SOURCE_NUMERIC_RECORD' if not row['errors'] and not row['null_fields'] else 'UNQUALIFIED_RETAINED_SOURCE_SLOT'
    counts=Counter(r['status'] for r in rows)
    return dict(source_schema_qualified=True,metadata_count=meta['count'],actual_event_slots=len(rows),maximum_requested_slots=50,
        qualified_numeric_slots=counts['QUALIFIED_SOURCE_NUMERIC_RECORD'],unqualified_retained_slots=counts['UNQUALIFIED_RETAINED_SOURCE_SLOT'],
        all_source_numeric_slots_qualified=bool(rows) and counts['QUALIFIED_SOURCE_NUMERIC_RECORD']==len(rows),
        sample_limit_reached=len(rows)==50,whole_catalog_window_complete='UNKNOWN_NOT_CLAIMED' if len(rows)==50 else 'ONLY_RETURNED_FIXED_FILTER_QUERY_NOT_GLOBAL_CATALOG',
        generated_epoch_ms=meta['generated'],api_version=meta['api'],event_ids_ordered_SHA256=sha(canonical([r['id'] for r in rows])),
        event_alias_groups=len(components),geographic_groups=len({r['source_geo_region_key'] for r in rows if r['source_geo_region_key']}),
        rows=rows,train_dev_blind_assigned=False,charts_generated=0,QA_generated=0,Dataset_GO=False,training_GO=False,GPU_GO=False)
