"""Offline exact task scorer. Full labels denominator, explicit natural EOS, no model."""
import argparse
import json
from pathlib import Path
import re
from contracts import need,sha


def score(labels,predictions):
    need(len({x['id'] for x in labels})==len(labels),'Duplicate label IDs')
    known={x['id'] for x in labels};need(all(x['id'] in known for x in predictions),'Unknown predicted ID')
    buckets={}
    for p in predictions:buckets.setdefault(p['id'],[]).append(p)
    rows=[]
    for q in labels:
        values=buckets.get(q['id'],[]);reason='missing';correct=False
        if q['data_status']!='QUALIFIED':reason='data_failure'
        elif len(values)>1:reason='duplicate_prediction'
        elif len(values)==1:
            p=values[0];text=p.get('text')
            if p.get('status')!='COMPLETED':reason='not_completed'
            elif type(p.get('natural_EOS')) is not bool or p['natural_EOS'] is not True:reason='no_explicit_natural_EOS'
            elif type(p.get('cap_hit')) is not bool or p['cap_hit'] is not False:reason='cap_or_unknown_cap'
            elif not isinstance(text,str):reason='missing_text'
            else:
                normalized=text.strip()
                valid=bool(re.fullmatch(r'-?\d+\.\d{2}',normalized)) if q['answer_type']=='decimal2' else bool(re.fullmatch(r'E\d{2}( E\d{2})*',normalized))
                correct=valid and normalized==q['answer'];reason='correct' if correct else 'wrong_or_parse_failure'
        rows.append(dict(id=q['id'],kind=q['kind'],chart_id=q['chart_id'],correct=correct,reason=reason))
    return dict(denominator=len(labels),correct=sum(r['correct'] for r in rows),accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,failures=sum(not r['correct'] for r in rows),rows=rows,
        metric='Exact displayed-value task accuracy; full failures retained. Not evidence of domain capability or original formal17 metric.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['train','dev'],required=True);a=p.parse_args()
    manifest=json.loads((a.dataset/'manifest-private.json').read_bytes());name=a.split+'/labels-private.jsonl';raw=(a.dataset/name).read_bytes();need(sha(raw)==manifest['outputs'][name]['SHA256'],'Frozen labels changed')
    labels=[json.loads(l) for l in raw.splitlines()];preds=[json.loads(l) for l in a.predictions.read_bytes().splitlines()];result=score(labels,preds)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},sort_keys=True))
