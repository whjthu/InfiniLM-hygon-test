"""Fixed six-case pilot queue validation; CPU-only and no runtime imports."""
import json
def validate_frozen_pilot(model_directory):
    proof=json.loads((model_directory/'fixed-data-r162a.json').read_bytes())
    expected={'case_count':6,'C1_cases':4,'C8_cases':2,'request_executions':20,'unique_question_ids':16,'C1_is_subset_of_C8':True,'not_new32_heldout':True,'cap':512,'context':1024,'W':8,'strategy':'state','W2pair_fixed_ON':True,'W13_fixed_OFF':True,'registry_cache_fixed_ON':True,'model_profile_OFF':True,'EOS_not_forced':True,'root_only_GPU_GO':True}
    assert proof==expected,'fixed pilot metadata changed'
    commands=[json.loads(p.read_bytes())for p in sorted((model_directory/'queue').glob('q*.json'))]
    assert len(commands)==7 and commands[-1]['action']=='stop'
    cases=commands[:-1]
    assert all(c['action']=='case'for c in cases)
    assert [c['C']for c in cases]==[1,1,1,1,8,8]
    assert len({c['id']for c in cases})==6
    for c in cases:
        assert len(c['requests'])==c['C']
        assert c['W']==8 and c['strategy']=='state'and c['cap']==512 and c['W2pair']and not c['model_profile']
        assert not any('W13'in k or 'partial'in k for k in c)
    c1={q['id']for c in cases[:4]for q in c['requests']}
    c8={q['id']for c in cases[4:]for q in c['requests']}
    assert len(c1)==4 and len(c8)==16 and c1<=c8
    assert sum(len(c['requests'])for c in cases)==20
    return {'passed':True,'case_count':6,'request_executions':20,'unique_question_ids':16,'C1_is_subset_of_C8':True}
