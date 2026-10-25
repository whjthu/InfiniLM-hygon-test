"""Derive actual phase-call evidence from immutable report; never execute a GPU call."""
def extract(report):
    phases=report.get('phase_calls',[]);offset=0;result=[]
    for case in report.get('speculative_quality_cases',[]):
        start=offset;assert offset<len(phases);prefill=phases[offset];offset+=1;past=case['input_tokens'];cycle_records=[]
        for index,cycle in enumerate(case['cycles']):
            assert offset<len(phases);verify=phases[offset];offset+=1
            expected_width=1 if cycle.get('pair_AR_continuation')or case['K']==0 else cycle['proposal_count']+1
            assert verify['past']==past and verify['input_width']==expected_width,(case['label'],index,past,verify)
            replay=None
            if cycle.get('needs_replay')and not cycle.get('state_only_replay'):
                replay=phases[offset];offset+=1;assert replay['phase']=='replay'and replay['past']==past and replay['input_width']==cycle['accepted']+1
            record={'cycle':index,'before_past':past,'retained_inputs':cycle['accepted']+1 if case['K']and not cycle.get('pair_AR_continuation')else 1,'verify':verify,'state_only_replay':cycle.get('state_only_replay',False),'full_model_replay':replay}
            if cycle.get('terminal'):record['terminal']=True
            cycle_records.append(record);past+=record['retained_inputs']
        evidence=None
        for index,row in enumerate(cycle_records[:-1]):
            if row['state_only_replay']:
                next_=cycle_records[index+1];assert next_['before_past']==row['before_past']+row['retained_inputs']
                evidence={'first_partial_cycle':row['cycle'],'accepted':case['cycles'][row['cycle']]['accepted'],'before_past':row['before_past'],'retained_inputs':row['retained_inputs'],'actual_verify_serial':row['verify']['serial'],'actual_next_forward_serial':next_['verify']['serial'],'actual_next_forward_past':next_['verify']['past'],'actual_next_forward_width':next_['verify']['input_width'],'next_CAPI_maxhost_s':next_['verify']['CAPI_maxhost_s'],'next_forward_completed':True,'explicit_state_and_outer_past_assertions_executed':'Pinned protocol checks phase1 and swapped active/shadow after successful state CAPI; next real CAPI independently validates engine length ledger','phase_source':'actual report phase_calls; no simulated GPU call'};break
        result.append({'label':case['label'],'strategy':case.get('strategy'),'phase_call_range':[start,offset],'first_state_partial_then_next_forward':evidence,'owner_stats56':case.get('state_owner_stats8x7')})
    return {'completed_cases':len(result),'completed_case_phase_calls':offset,'trailing_inprogress_calls':len(phases)-offset,'cases':result}
