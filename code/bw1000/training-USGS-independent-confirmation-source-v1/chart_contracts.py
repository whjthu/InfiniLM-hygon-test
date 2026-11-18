"""Deterministic grouping before questions, display-derived targets; no I/O or network."""
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
import math



def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode()


def sha(b):return hashlib.sha256(b).hexdigest()


def display(x):
    v=Decimal(x);need(v.is_finite(),'Nonfinite display')
    r=v.quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    return format(Decimal('0.00') if r==0 else r,'.2f')


def distance(a,b):
    lat1,lat2=map(math.radians,[float(a['values']['latitude']),float(b['values']['latitude'])])
    dlat=lat2-lat1;dlon=math.radians(float(b['values']['longitude'])-float(a['values']['longitude']))
    h=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088*2*math.asin(min(1,math.sqrt(max(0,h))))


def displayed_rows(chart):
    return [dict(label='E'+str(i+1).zfill(2),date_UTC=datetime.fromtimestamp(r['time_UTC_epoch_ms']/1000,timezone.utc).strftime('%m-%d %H:%M:%S'),
        reported_magnitude=display(r['values']['magnitude']),magType=r['magType'],depth_km=display(r['values']['depth_km'])) for i,r in enumerate(chart['rows'])]


def questions(chart):
    rows=displayed_rows(chart);first,last=rows[0],rows[-1]
    depths=[Decimal(r['depth_km']) for r in rows];mags=[Decimal(r['reported_magnitude']) for r in rows]
    definitions=[('lookup_reported_magnitude','What is the reported magnitude number for E01? Answer with exactly two decimal places.',first['reported_magnitude'],'decimal2'),
        ('lookup_depth','What is the depth in km for E01? Answer with exactly two decimal places.',first['depth_km'],'decimal2'),
        ('depth_change','Compute the last listed row depth minus E01 depth, in km. Use the displayed numbers and answer with exactly two decimal places, with a minus sign if negative.',display(depths[-1]-depths[0]),'decimal2'),
        ('maximum_depth','Which event label(s) have the largest displayed depth in km? Include every tie, separated by one space in table order.',' '.join(r['label'] for r,d in zip(rows,depths) if d==max(depths)),'labels'),
        ('maximum_reported_number','Which event label(s) have the largest displayed reported magnitude NUMBER? Compare displayed numbers only, not physical energy or risk. Include every tie, separated by one space in table order.',' '.join(r['label'] for r,m in zip(rows,mags) if m==max(mags)),'labels')]
    return [dict(id=chart['id']+':'+str(i),chart_id=chart['id'],group=chart['group'],kind=kind,question=q,answer=a,answer_type=t,
        display_rows_SHA256=sha(canonical(rows)),single_event_window=len(rows)==1) for i,(kind,q,a,t) in enumerate(definitions)]
