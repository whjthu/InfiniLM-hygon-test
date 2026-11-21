"""Only new72 identity/failure accounting boundaries; no native imports or data."""
import copy
import hashlib
import json
from pathlib import Path
from input_contract import require_row,row_sha,validate_progress


def main():
    tests=[]
    def valid(name,value):
        validate_progress(value);tests.append(dict(name=name,accepted=True))
    def invalid(name,callback):
        try:callback()
        except (ValueError,KeyError):tests.append(dict(name=name,rejected=True))
        else:raise AssertionError('invalid boundary accepted: '+name)
    def p(completed,attempted,ea,ec,ca,cc,failed=False):
        return dict(planned_rows=72,completed_rows=completed,attempted_rows=attempted,
            missing_rows=72-completed,unattempted_rows=72-attempted,failed=failed,
            failed_rows=1 if failed and attempted>completed else 0,
            encode_calls_attempted=ea,encode_calls_completed=ec,collator_calls_attempted=ca,collator_calls_completed=cc)
    valid('before_native_attempt',p(0,0,0,0,0,0))
    valid('failure_during_first_full_encode_preserves_attempt',p(0,1,1,0,0,0,True))
    valid('completed72_then_post_failure_keeps_all_calls',p(72,72,144,144,72,72,True))
    valid('actual_success_denominator',p(72,72,144,144,72,72))
    base=p(72,72,144,144,72,72)
    for name,key,value in [('old64_denominator','planned_rows',64),('missing_denominator','missing_rows',1),
                           ('lost_encoder_completion','encode_calls_completed',143),('lost_collator_completion','collator_calls_completed',71),
                           ('fabricated_attempts','attempted_rows',73),('boolean_completed','completed_rows',True)]:
        bad=copy.deepcopy(base);bad[key]=value
        invalid(name,lambda bad=bad:validate_progress(bad))
    row={'id':'synthetic-fixture-only','images':['/new-scope/data/train/media/000.png'],
         'messages':[{'role':'user','content':'synthetic question'},{'role':'assistant','content':'FINAL: 1'}]}
    expected=dict(ordinal=1,row_SHA256=row_sha(row),source_ID_SHA256=hashlib.sha256(row['id'].encode()).hexdigest(),image=row['images'][0])
    require_row(row,expected,1);tests.append(dict(name='exact_new_input_identity',accepted=True))
    bad=copy.deepcopy(row);bad['messages'][1]['content']='FINAL: 2'
    invalid('changed_supervision_rejected',lambda:require_row(bad,expected,1))
    invalid('changed_ordinal_rejected',lambda:require_row(row,expected,2))
    out=Path(__file__).resolve().parents[3]/'local-only/training-internvl-USGS72-native-CPU-v1-source-checks'
    out.mkdir(exist_ok=False)
    result=dict(passed=True,tests=tests,accepted=sum(t.get('accepted',False) for t in tests),
        rejected=sum(t.get('rejected',False) for t in tests),synthetic_contracts_only=True,
        real_native_encoder_calls=0,target_calls=0,Torch_imported=False,GPU=0,training_updates=0)
    (out/'CPU-result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ('passed','accepted','rejected','real_native_encoder_calls','GPU')},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
