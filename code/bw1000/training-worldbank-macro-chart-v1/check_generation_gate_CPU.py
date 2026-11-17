"""New targeted termination-evidence test only; synthetic216, no source/model data."""
import json
from contracts import need,score
from score_once import generation_gate


def run():
    labels=[dict(id=str(i),data_status='QUALIFIED',answer_type='decimal2',answer='1.00',country_group='synthetic',kind='lookup') for i in range(216)]
    def row(i,**extra):return dict(id=str(i),status='COMPLETED',response='FINAL: 1.00',**extra)
    rows=[row(0,natural_EOS=True,cap_hit=False),row(1,natural_EOS=True,cap_hit=True),row(2,natural_EOS=False,cap_hit=False),
        row(3,cap_hit=False),row(4,natural_EOS=True),row(5,natural_EOS=1,cap_hit=False),row(6,natural_EOS=True,cap_hit=0),
        row(7,natural_EOS='true',cap_hit=False),row(8,natural_EOS=True,cap_hit='false'),
        dict(row(9,natural_EOS=True,cap_hit=False),status='FAILED')]
    values,gate=generation_gate(rows);result=score(labels,values,216)
    need(gate['eligible_rows']==1 and gate['unqualified_rows']==9,'Bad termination evidence was accepted')
    need(result['planned']==216 and result['correct']==1 and result['missing_outputs']==215,'Full216 denominator/failed answers lost')
    need(gate['rejection_counts']==dict(cap_hit=1,no_explicit_natural_EOS=1,missing_or_untyped_termination_flags=6,not_completed=1),'Failure categories differ')
    try:generation_gate([rows[0],rows[0]])
    except ValueError:duplicate_rejected=True
    else:raise AssertionError('Duplicate IDs accepted')
    return dict(passed=True,synthetic_planned=216,supplied=10,eligible=1,unqualified=9,absent=206,correct=1,score_denominator=216,
        rejection_counts=gate['rejection_counts'],duplicate_rejected=duplicate_rejected,real_HTTP=0,real_gold_read=0,model_calls=0,
        old_contract_tests_repeated=False,GPU_GO=False,training_GO=False)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
