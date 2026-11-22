"""Pure frozen dev36 metrics. ROOT calls only after a separate all108-sealed inference audit.

No I/O, model or gold discovery. Keeping this pure integration point does not claim
that a future three-checkpoint inference was run or independently accepted.
"""
from frozen_score import score
from response_protocol import content_prediction
from contracts import need

ARMS=('base','domain25','generic25')


def measure(labels,predictions):
    need(len(labels)==len({q['id'] for q in labels})==36,'full dev36 labels denominator required')
    need(set(predictions)==set(ARMS),'fixed three checkpoint arms required')
    known={q['id']:q for q in labels};result={};normalized={};correct={}
    for arm in ARMS:
        raw=predictions[arm]
        need(len({r['id'] for r in raw})==len(raw) and all(r['id'] in known for r in raw),'duplicate/foreign prediction')
        rows=[content_prediction(r,known[r['id']]) for r in raw]
        normalized[arm]={r['id']:r for r in rows}
        scored=score(labels,rows);correct[arm]={r['id']:r['correct'] for r in scored['rows']}
        result[arm]=dict(denominator=36,content_correct=scored['correct'],content_accuracy=scored['accuracy'],
            strict_FINAL=sum(r['strict_FINAL_format'] for r in rows),
            content_parse_valid=sum(r['content_parse_valid'] for r in rows),
            missing=36-len(raw),cap_or_unknown=sum(r.get('cap_hit') is not False for r in raw),
            no_natural_EOS=sum(r.get('natural_EOS') is not True for r in raw))
    def paired(left,right):
        both=[]
        for key in known:
            a,b=normalized[left].get(key),normalized[right].get(key)
            if a and b and all(x.get('content_parse_valid') is True and x.get('natural_EOS') is True and
                              x.get('cap_hit') is False for x in (a,b)):both.append(key)
        return dict(full_denominator=36,win=sum(correct[left][k] and not correct[right][k] for k in known),
            loss=sum(correct[right][k] and not correct[left][k] for k in known),
            tie=sum(correct[left][k]==correct[right][k] for k in known),both_parsed=len(both),
            both_parsed_win=sum(correct[left][k] and not correct[right][k] for k in both),
            both_parsed_loss=sum(correct[right][k] and not correct[left][k] for k in both))
    return dict(arms=result,domain_vs_generic=paired('domain25','generic25'),
        domain_vs_base=paired('domain25','base'),format_confounded_caption_control=True,
        semantic_advantage_or_promotion_inferred=False,formal17=False)
