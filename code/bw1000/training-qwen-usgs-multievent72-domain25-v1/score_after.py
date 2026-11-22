"""First local dev gold only after complete original72 sealed predictions + independent terminal."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

from frozen_score import score
from response_protocol import content_prediction
from prepare_dev import need,pin

LABELS_PIN=dict(bytes=24409,SHA256='419520b09e58b3f9cef757906e49c20cb8ecaacf04ad6e932f97cafa8bd0c14a')
REQUEST='c933ba7359f84e8b9b80ac40d2d30964'


def jsonl(raw):return [json.loads(line) for line in raw.splitlines()]


def run(a):
    claim=json.loads(Path(__file__).with_name('CLAIM.json').read_bytes())
    need(claim['root_GO'] is claim['SCORE_GO'] is True,'operative score scope not adopted')
    raw=a.independent.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==a.independent_sha,'actual original independent terminal pin required')
    independent=json.loads(raw)
    need(independent['request']==REQUEST and independent['passed'] is True and
         independent['actual_updates']==25 and independent['post_clean'] is True and
         independent['all72sealed_before_gold'] is True and independent['metric_gold_read'] is False,
         'full original training/72 predictions must be independently terminal before first gold')
    pins=independent['sealed_predictions'];sealed={}
    for name in ('base-predictions-private.jsonl','domain25-predictions-private.jsonl','generation-finalized.json'):
        p=a.predictions/name
        need(p.is_file() and not p.is_symlink(),'original sealed prediction member missing')
        raw=p.read_bytes();need(pin(raw)==pins[name],'original sealed prediction member changed');sealed[name]=raw
    final=json.loads(sealed['generation-finalized.json'])
    need(final['all72sealed_before_gold'] is final['passed'] is final['all_prediction_files_closed'] is True and
         final['dev_gold_read'] is False and final['blind_read'] is False,'frozen generation seal differs')
    for arm in ('base','domain25'):
        name=arm+'-predictions-private.jsonl';rows=jsonl(sealed[name]);entry=final['arms'][arm]
        need(len(rows)==len({r['id'] for r in rows})==36 and entry['completed']==36 and entry['missing']==0,
             'all36 real prediction slots required before first gold')
        need(all(type(r['natural_EOS']) is type(r['cap_hit']) is bool for r in rows),
             'typed raw-EOS/cap observations required')
    # All72 are durably sealed, exact pins matched and independent consumption passed above.
    labels_raw=(a.dataset/'dev/labels-private.jsonl').read_bytes()
    need(pin(labels_raw)==LABELS_PIN,'original dev36 labels changed')
    labels=jsonl(labels_raw)
    need(len(labels)==len({q['id'] for q in labels})==36 and len({q['chart_id'] for q in labels})==6 and
         len({q['group'] for q in labels})==2,'original complete dev split differs')
    by_id={q['id']:q for q in labels};result={};detailed={}
    for arm in ('base','domain25'):
        rows=jsonl(sealed[arm+'-predictions-private.jsonl'])
        need({r['id'] for r in rows}==set(by_id),'prediction/gold IDs differ')
        normalized=[content_prediction(row,by_id[row['id']]) for row in rows]
        scored=score(labels,normalized);detailed[arm]=scored
        result[arm]={k:v for k,v in scored.items() if k!='rows'}
        result[arm].update(strict_FINAL=sum(r['strict_FINAL_format'] for r in normalized),
            content_parse_valid=sum(r['content_parse_valid'] for r in normalized),
            cap=sum(r['cap_hit'] for r in rows),natural_EOS=sum(r['natural_EOS'] for r in rows),
            by_kind={kind:dict(denominator=6,correct=sum(r['correct'] for r in scored['rows'] if r['kind']==kind))
                     for kind in {q['kind'] for q in labels}})
    base={r['id']:r['correct'] for r in detailed['base']['rows']}
    domain={r['id']:r['correct'] for r in detailed['domain25']['rows']}
    paired=dict(win=sum(domain[k] and not base[k] for k in base),loss=sum(base[k] and not domain[k] for k in base),
                tie=sum(base[k]==domain[k] for k in base),denominator=36)
    summary=dict(state='FIRST_DEV36_SCORE_EXPOSED',request=REQUEST,arms=result,paired=paired,
        independent_terminal_SHA256=a.independent_sha,source_rows72=72,physical_exposures800=800,
        charts=6,closure_groups=2,blind_read=False,domain_scope='new multievent72 Qwen recipe; not original formal17',
        Model_GO=False,formal17=False,performance=False,allocated_DCU=8,visible_inference_DCU=1)
    a.output.mkdir(mode=0o700)
    with (a.output/'scored-private.json').open('x',encoding='utf8') as f:json.dump(dict(summary=summary,details=detailed),f,sort_keys=True)
    with (a.output/'summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,sort_keys=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('independent','predictions','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--independent-sha',required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
