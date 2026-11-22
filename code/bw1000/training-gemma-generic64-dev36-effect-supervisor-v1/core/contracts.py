"""Frozen six-kind tasks from actual displayed values; no models or I/O."""
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json

KINDS=['maximum_reported_number','minimum_reported_number','maximum_depth','minimum_depth','depth_change','reported_number_change']
def need(ok,why):
    if not ok:raise ValueError(why)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def display(x):
    value=Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    return format(Decimal('0') if value==0 else value,'.2f')
def displayed_rows(chart):
    need(len(chart['rows'])==6,'frozen actual six-event window required')
    return [dict(label='E%02d'%(i+1),time_UTC=datetime.fromtimestamp(r['time_UTC_epoch_ms']/1000,timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        mag=display(r['values']['magnitude']),depth=display(r['values']['depth_km']),magType=r['magType'],preferred_net=r['preferred_net'],sources=r['sources']) for i,r in enumerate(chart['rows'])]
def questions(chart):
    rows=displayed_rows(chart);mag=[Decimal(r['mag']) for r in rows];depth=[Decimal(r['depth']) for r in rows]
    def labels(values,op):return ' '.join(r['label'] for r,v in zip(rows,values) if v==op(values))
    definitions=[('maximum_reported_number','Which event label(s) have the largest DISPLAYED reported magnitude NUMBER? Compare numbers only, not energy or risk.',labels(mag,max),'labels'),
        ('minimum_reported_number','Which event label(s) have the smallest DISPLAYED reported magnitude NUMBER? Compare numbers only, not energy or risk.',labels(mag,min),'labels'),
        ('maximum_depth','Which event label(s) have the largest DISPLAYED depth in km?',labels(depth,max),'labels'),
        ('minimum_depth','Which event label(s) have the smallest DISPLAYED depth in km?',labels(depth,min),'labels'),
        ('depth_change','Compute DISPLAYED E06 depth minus DISPLAYED E01 depth, in km.',display(depth[-1]-depth[0]),'decimal2'),
        ('reported_number_change','Compute DISPLAYED E06 reported magnitude NUMBER minus DISPLAYED E01 number. This is not an energy or risk calculation.',display(mag[-1]-mag[0]),'decimal2')]
    output=[]
    for index,(kind,q,answer,typ) in enumerate(definitions):
        suffix=' Include every tie; list event labels in table order, separated by spaces.' if typ=='labels' else ' Use exactly two decimal places and a minus sign if negative; retain zero.'
        output.append(dict(id=chart['id']+':q'+str(index),chart_id=chart['id'],group=chart['group'],kind=kind,question=q+suffix+' Answer FINAL: followed by the answer.',
            answer=answer,answer_type=typ,data_status='QUALIFIED',event_ids=[r['label'] for r in rows],display_rows_SHA256=sha(canonical(rows))))
    return output
