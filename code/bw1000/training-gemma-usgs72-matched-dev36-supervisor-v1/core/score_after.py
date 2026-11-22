"""First gold only after actual108 seal AND independent terminal. No inference or selection."""
import argparse
import json
from pathlib import Path
from effect_contract import *
from content_vs_format import measure
from frozen_score import score
from response_protocol import content_prediction

LABELS=dict(bytes=24409,SHA256='419520b09e58b3f9cef757906e49c20cb8ecaacf04ad6e932f97cafa8bd0c14a')

def validate_seal(independent,folder):
    need(independent['request']==REQUEST and independent['passed'] is independent['independent_passed'] is True and
         independent['all108sealed_before_gold'] is True and independent['dev_gold_read'] is independent['blind_read'] is False,
         'actual independent108 required BEFORE any labels read')
    expected={arm+'-predictions-private.jsonl' for arm in ARMS}|{'generation-finalized.json'}
    need(set(independent['sealed_predictions'])==expected,'exact sealed four files required')
    files={name:read(folder/name,value,cap=4<<20) for name,value in independent['sealed_predictions'].items()}
    final=json.loads(files['generation-finalized.json'])
    need(final['request']==REQUEST and final['passed'] is final['all108sealed_before_gold'] is
         final['all_prediction_files_closed'] is True and final['dev_gold_read'] is False and final['blind_read'] is False,
         'original generation seal differs')
    rows={arm:[json.loads(x) for x in files[arm+'-predictions-private.jsonl'].splitlines()] for arm in ARMS}
    order=None
    for arm,data in rows.items():
        ids=[r['id'] for r in data]
        need(len(ids)==len(set(ids))==36 and (order is None or ids==order),'all108 ordered complete slots required before gold')
        order=ids
        need(all(type(r['natural_EOS']) is type(r['cap_hit']) is bool for r in data),'typed EOS/cap required')
    return rows

def run(a):
    source=verify('SCORE')
    independent_raw=read(a.independent,dict(bytes=a.independent_bytes,SHA256=a.independent_sha))
    independent=json.loads(independent_raw)
    need(independent['source_manifest']==source,'independent source differs')
    rows=validate_seal(independent,a.predictions) # NO labels path touched before this completes.
    need(a.dataset.name=='dataset' and a.dataset.parent.name=='training-USGS-multievent-render144-v2-original',
         'original frozen dev36 local dataset required')
    a.output.mkdir(mode=0o700) # exclusive first-score destination before first gold
    dump(a.output/'first-score-intent.json',dict(request=REQUEST,score_request=SCORE_REQUEST,independent=pin(independent_raw),
        source_manifest=source,planned_each=36,planned_total=108,dev_gold_read_at_intent=False,blind_read=False))
    labelraw=read(a.dataset/'dev/labels-private.jsonl',LABELS)
    labels=[json.loads(x) for x in labelraw.splitlines()]
    need(len(labels)==36 and len({q['chart_id'] for q in labels})==6 and len({q['group'] for q in labels})==2,'frozen36/6charts/2groups required')
    summary=measure(labels,rows)
    byid={q['id']:q for q in labels}
    for arm,values in rows.items():
        need({r['id'] for r in values}==set(byid),'prediction/gold ID mismatch')
        scored=score(labels,[content_prediction(r,byid[r['id']]) for r in values])
        summary['arms'][arm]['by_kind']={kind:dict(denominator=sum(q['kind']==kind for q in labels),
            correct=sum(r['correct'] for r in scored['rows'] if r['kind']==kind)) for kind in {q['kind'] for q in labels}}
        correct={r['id']:r['correct'] for r in scored['rows']}
        summary['arms'][arm]['by_group']={pin(str(group).encode())['SHA256']:dict(
            denominator=sum(q['group']==group for q in labels),correct=sum(correct[q['id']] for q in labels if q['group']==group))
            for group in {q['group'] for q in labels}}
        summary['arms'][arm]['failed_status']=sum(r.get('status')!='COMPLETED' for r in values)
    summary.update(state='GEMMA_DEV36_FIRST_SCORE_EXPOSED',request=REQUEST,score_request=SCORE_REQUEST,
        independent=pin(independent_raw),charts=6,closure_groups=2,planned_predictions=108,
        initial_is_actual851_target_LoRA=True,dev_gold_now_EXPOSED=True,blind_read=False,
        training_updates_added=0,Model_GO=False,formal17=False,performance=False,
        initial_same_does_not_make_equal_tokens=True,uses_same_train_FINAL_format=True)
    dump(a.output/'scored-private.json',dict(summary=summary,source_labels=LABELS,prediction_pins=independent['sealed_predictions']))
    dump(a.output/'summary.json',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('independent','predictions','dataset','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--independent-bytes',type=int,required=True);p.add_argument('--independent-sha',required=True)
    a=p.parse_args();print(json.dumps(run(a),sort_keys=True))
