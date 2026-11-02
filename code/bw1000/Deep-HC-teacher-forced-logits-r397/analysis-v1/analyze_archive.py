"""stdlib-only audit of sealed fixed-teacher observations. No execution or logits reconstruction."""
import argparse,hashlib,json,math,pathlib

NATIVE='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
OWNER='66e93d47f89d9f8c7645d41427e1ae264e59ddeb3a2cc400abd6aa68db8ccfa6'
TEACHER='596df3e1131e55184edc034f57f8447e025b91a38c7d6a20a69eaa8b61f39513'
PROMPT='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
MODES=[0,1,1,0]

def decode(archive,role):
    f=archive['files'][role];b=f['text'].encode('utf-8')
    assert hashlib.sha256(b).hexdigest()==f['SHA256'] and len(b)==f['bytes'],role
    return json.loads(f['text'])

def pair(sequences,left,right):
    a=sequences[left]['decisions'];b=sequences[right]['decisions']
    exact=[];different_chosen=[];fixed_delta=[];gap_delta=[];relative_delta=[];top1_deltas=[];margin_deltas=[]
    for i,(x,y)in enumerate(zip(a,b)):
        if x['logits_BF16_bytes_SHA256']==y['logits_BF16_bytes_SHA256']:exact.append(i)
        if x['selected_scores']['native_argmax']['id']!=y['selected_scores']['native_argmax']['id']:different_chosen.append(i)
        roles=['correct_original83','incorrect_original83']+(['teacher']if i<83 else[])
        for role in roles:
            assert x['selected_scores'][role]['id']==y['selected_scores'][role]['id']
            fixed_delta.append(abs(y['selected_scores'][role]['score']-x['selected_scores'][role]['score']))
            relative_delta.append(abs((y['selected_scores'][role]['score']-y['top8'][0]['score'])-(x['selected_scores'][role]['score']-x['top8'][0]['score'])))
        top1_deltas.append(y['top8'][0]['score']-x['top8'][0]['score']);margin_deltas.append(y['top1_margin']-x['top1_margin'])
        gx=x['selected_scores']['correct_original83']['score']-x['selected_scores']['incorrect_original83']['score']
        gy=y['selected_scores']['correct_original83']['score']-y['selected_scores']['incorrect_original83']['score'];gap_delta.append(gy-gx)
    return {'sequence_pair':[left,right],'HC_modes':[MODES[left],MODES[right]],'decision_count':84,
            'full_returned_BF16_logit_hash_equal_count':len(exact),'first_unequal_full_logit_hash_index':next((i for i in range(84)if i not in exact),None),
            'native_chosen_ID_difference_indices':different_chosen,'fixed_ID_selected_score_max_abs_delta':max(fixed_delta),
            'fixed_ID_gap_to_top1_max_abs_delta':max(relative_delta),'top1_score_delta_min_max':[min(top1_deltas),max(top1_deltas)],'top1_margin_delta_max_abs':max(map(abs,margin_deltas)),
            'original_pair_gap_delta_max_abs':max(map(abs,gap_delta)),'output83_original_pair_gap_delta':gap_delta[83],
            'unrecorded_full_vector_error':'UNKNOWN: only returned-logit byte SHA and selected/top8 values retained'}

def analyze(archive):
    report=decode(archive,'report');watch=decode(archive,'watch');legacy=decode(archive,'legacy');teacher=decode(archive,'teacher')
    assert report['actual_step']==archive['actual_GPU'] and report['Native_SHA256']==NATIVE
    assert report['source_manifest_SHA256']==OWNER and report['teacher_count']==83 and report['prompt_count']==8489
    ids=teacher['teacher_token_ids'];assert len(ids)==83 and 1 not in ids and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==TEACHER
    assert report['teacher_IDs_SHA256']==TEACHER and report['prompt_IDs_SHA256']==PROMPT
    assert report['context']==32768 and report['prefill_chunk']==1024 and report['observer_enabled']is False
    assert report['HTTP_requests']==0 and report['natural_EOS_claim']is False and report['teacher_forced_stop']is True
    assert report['quality_PASS']is False and report['performance_PASS']is False and report['diagnostic_only']is True
    assert report['completed']is True and report['passed']is True and report['normal_Native_shutdown']is True
    assert report['actual_forwards']==report['actual_forward_attempts']==368 and report['actual_resets']==report['actual_reset_attempts']==4
    assert report['HC_clear_rc']==report['HC_after_clear']==0 and report['observer_disarm_rc']==[0,0] and not report['cleanup_errors']
    assert legacy['engine_worker_shutdown_completed']is True and legacy['shutdown_entered']is True
    assert legacy['library_sha256']==NATIVE
    owners=report['owner_exit_observation'];assert owners['all8_unique_threads_exited']is True
    assert len(owners['owners'])==8 and len({x['tid']for x in owners['owners']})==8 and {x['device']for x in owners['owners']}==set(range(8))
    assert not any(x['task_exists_after_destroy']for x in owners['owners'])
    assert archive['independent_post']['kfd_pids']==[] and archive['independent_post']['owned_or_old_Deep_processes']==[] and archive['independent_post']['HTTP43882_bindable']is True
    assert watch['actual_GPU']==archive['actual_GPU'] and watch['CPU_keeper_actual']==archive['actual_CPU_keeper'] and watch['completed']is True
    assert watch['terminal'][2:4]==['COMPLETED','0:0'] and watch['srun_exit_code']==0 and watch['post_empty']is True and watch['Model_qualified']is False
    assert archive['GPU_accounting']['State']=='COMPLETED' and archive['GPU_accounting']['ExitCode']=='0:0'
    assert archive['keeper_accounting']['State']=='COMPLETED' and archive['keeper_accounting']['ExitCode']=='0:0'
    sequences=report['sequences'];assert len(sequences)==4 and [x['HC_mode']for x in sequences]==MODES
    output83=[];ties=[];sequence_summaries=[]
    for s,mode in zip(sequences,MODES):
        assert s['completed']is True and s['reset_ok']is True and s['prefill_calls']==9 and s['decode_calls']==83 and s['last_total']==8572
        assert len(s['HC_forward_coverage'])==92 and len(s['decisions'])==84
        for k,cov in enumerate(s['HC_forward_coverage']):
            t=1024 if k<8 else 297 if k==8 else 1
            assert cov['input_tokens']==t and cov['mode']==mode and cov['not_hardware_GPU_hit_counter']is True
            expected=[0,0,86,86 if t==1 else 0,86 if t==1 and mode else 0,86 if t!=1 else 0,0]
            assert cov['per_owner_delta']==[expected]*8
        for i,d in enumerate(s['decisions']):
            assert d['output_index']==i and d['finite']is True
            assert d['input_past']==(8192 if i==0 else 8488+i) and d['input_count']==(297 if i==0 else 1)
            assert d['input_IDs_SHA256']==sequences[0]['decisions'][i]['input_IDs_SHA256']
            if i:assert d['input_IDs_SHA256']==hashlib.sha256(json.dumps([ids[i-1]],separators=(',',':')).encode()).hexdigest()
            top=d['top8'];assert len(top)==8 and len({x['id']for x in top})==8 and all(0<=x['id']<129280 and math.isfinite(x['score'])for x in top)
            assert top==sorted(top,key=lambda x:(-x['score'],x['id']))
            assert d['top1_margin']==top[0]['score']-top[1]['score'] and d['top1_margin']>=0
            selected=d['selected_scores'];assert all(math.isfinite(v['score'])for v in selected.values())
            assert selected['native_argmax']['score']==top[0]['score'] and selected['independent_argmax']==top[0]
            assert selected['correct_original83']['id']==30722 and selected['incorrect_original83']['id']==28302
            if i<83:assert selected['teacher']['id']==ids[i]
            if d['top1_margin']==0:ties.append([s['sequence'],i])
        sequence_summaries.append({'sequence':s['sequence'],'HC_mode':mode,'Native_chosen_matches_frozen_teacher_count_first83':sum(d['selected_scores']['native_argmax']['id']==ids[i]for i,d in enumerate(s['decisions'][:83])),
            'BF16_top1_tie_indices':[d['output_index']for d in s['decisions']if d['top1_margin']==0],
            'first_prefill_output0_top1_score':s['decisions'][0]['top8'][0]['score'],'first_prefill_output0_margin':s['decisions'][0]['top1_margin'],
            'HC_eligible_before_output0_each_owner':[sum(c['per_owner_delta'][owner][3]for c in s['HC_forward_coverage'][:9])for owner in range(8)],
            'HC_candidate_completed_before_output0_each_owner':[sum(c['per_owner_delta'][owner][4]for c in s['HC_forward_coverage'][:9])for owner in range(8)]})
        d=s['decisions'][83];output83.append({'sequence':s['sequence'],'HC_mode':mode,'native_chosen':d['selected_scores']['native_argmax'],
            'independent_argmax':d['selected_scores']['independent_argmax'],'top8':d['top8'],'BF16_observable_top1_margin':d['top1_margin'],
            'maximum_BF16_tie_count':d['maximum_tie_count'],'original_correct30722':d['selected_scores']['correct_original83']['score'],
            'original_incorrect28302':d['selected_scores']['incorrect_original83']['score'],
            'original_pair_gap_30722_minus_28302':d['selected_scores']['correct_original83']['score']-d['selected_scores']['incorrect_original83']['score'],
            'returned_BF16_logit_bytes_SHA256':d['logits_BF16_bytes_SHA256']})
    return {'completed':True,'numerical_observation_audit_passed':True,'quality_PASS':False,'performance_PASS':False,'Model_qualified':False,
            'actual_GPU':archive['actual_GPU'],'actual_CPU_keeper':archive['actual_CPU_keeper'],'GPU_accounting':archive['GPU_accounting'],'keeper_accounting':archive['keeper_accounting'],
            'controller_original_Diagnostic_qualified':watch.get('Diagnostic_qualified'),'controller_original_watch_unchanged':True,
            'source_publication':watch['publication_commit'],'Native_SHA256':NATIVE,'teacher_IDs_SHA256':TEACHER,'prompt_IDs_SHA256':PROMPT,
            'actual_sequences':4,'HC_modes':MODES,'actual_forwards':368,'actual_resets':4,'decisions_each':84,'output83':output83,
            'cost_scope':{'GPU_step_elapsed_s':int(archive['GPU_accounting']['ElapsedRaw']),'allocated_GPU_cards':8,'allocated_GPU_card_seconds':8*int(archive['GPU_accounting']['ElapsedRaw']),
                          'legacy_loading_seconds':legacy.get('loading_seconds'),'decision_wrapper_wall_s_each':[sum(d['forward_wall_s']for d in s['decisions'])for s in sequences],
                          'decision_wrapper_scope':'84 observed decisions persequence: Nativeforward plus counterverification/finite/top8diagnostic; excludesfirst8prefillchunks andreportJSONIO. Not pureGPUbusy or fullsequence latency.',
                          'GPU_busy_time':'UNKNOWN'},
            'within_mode_repeats':[pair(sequences,0,3),pair(sequences,1,2)],
            'cross_mode_pairs':[pair(sequences,0,1),pair(sequences,0,2),pair(sequences,3,1),pair(sequences,3,2)],
            'per_sequence_numerical_summary':sequence_summaries,'same_Native_chosen_across_all4_sequence_decision_count':sum(len({s['decisions'][i]['selected_scores']['native_argmax']['id']for s in sequences})==1 for i in range(84)),
            'original_wrong_token28302_at_output83_reproduced':any(x['native_chosen']['id']==28302 for x in output83),
            'BF16_top1_tie_sequence_index_pairs':ties,'eight_actual_owner_threads_exited':True,'independent_per_owner_CPP_success_markers':legacy.get('independent_per_owner_normal_markers','UNKNOWN'),'post_clean':True,
            'limits':['F32 values are exact representation of returned BF16, not Native precast F32 logits or margin.',
                      'Different within-mode hashes/choices are observations, not a numerical failure threshold or identification of hardware/reset cause.',
                      'Cross-mode differences apply to this fixed teacher trajectory. They do not establish free-running quality or original393 error cause.',
                      'Original393 quality FAIL is unchanged. Teacher-forced stop is not natural EOS or an HTTP result.',
                      'Allocated card-time/loading/diagnostic summaries are not GPU busy time and no speedup is claimed.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args();raw=pathlib.Path(a.archive).read_bytes();q=json.loads(raw)
    out=analyze(q.get('archive',q));out['archive_SHA256']=hashlib.sha256(raw).hexdigest();print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
