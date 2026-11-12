"""Summarize a two-request same-Native/OFF/OFF control without promotion claims."""
import math
from alias_policy import AUDIT_FORWARDS,SCHEDULE,expected_counts
from pilot_inputs import require


def summarize(owners,results):
    require(SCHEDULE==((0,0),(0,0)) and len(owners)==len(results)==2,'incomplete OFF/OFF control')
    for index,(owner,result) in enumerate(zip(owners,results)):
        stats=owner['history_alias'];expected=expected_counts(owner['Native_continuation'],0)
        require(owner['r454_index']==index and stats['mode']==0 and stats['counters']==[expected]*8
                and expected[2:]==[0,0] and result['native_forwards']*21==expected[0],
                'OFF-only eight-owner coverage differs')
    audit=owners[1]['history_alias']['logits_audit']
    require(2<=len(audit)<=AUDIT_FORWARDS and all(math.isfinite(r['max_absolute']) and math.isfinite(r['rms']) for r in audit),
            'invalid matched-prefix repeat audit')
    require(results[0]['case_id']==results[1]['case_id'],'repeat request identity differs')
    same=results[0]['generated_IDs_SHA256']==results[1]['generated_IDs_SHA256']
    eos=all(r['score']['natural_EOS'] for r in results)
    first,repeat=[r['backend_non_eos_decode_tok_s'] for r in results]
    ratio=repeat/first if same and eos and first and repeat else None
    return dict(requests=2,unique_cases=1,alias_modes=[0,0],same_Native=True,
                correct=sum(r['score']['correct'] for r in results),natural_EOS=sum(r['score']['natural_EOS'] for r in results),
                generated_ids_match_in_audit_pair=same,audit_first_forward_limit=AUDIT_FORWARDS,audit_matched_forwards=len(audit),
                audit_bitwise_equal_forwards=sum(r['differing_BF16_values']==0 for r in audit),
                audit_max_absolute=max(r['max_absolute'] for r in audit),audit_rows=audit,
                same_output_natural_EOS_repeat_rate_ratio=ratio,valid_timing_pairs=0,
                stage='same immutable alias-capable Native with branch always OFF; one GSM request repeated',
                interpretation='Nonzero matched-prefix differences show variation with no ON arm. Zero differences in this one repeat do not establish general determinism.',
                broad_quality_qualified=False,performance_qualified=False,client_per_token_decode_qualified=False,promote=False)
