"""Observe existing execute_plan callbacks; no Native/GPU or scheduling import.

This is HOST control-flow evidence. No extra forward/vote/sample/discard call is
inserted. The original executor remains the sole owner of callback ordering.
"""

def execute_observed(plan,past,*,prefill,window,forward,vote,sample,discard,record,executor):
    if not prefill:
        return executor(plan,past,prefill=prefill,window=window,forward=forward,
                        vote=vote,sample=sample,discard=discard)
    if type(record)is not list:
        raise TypeError('A caller-owned per-request trace list is required')
    entry=dict(prefill=True,requested_window=window,planned_chunks=len(plan),
               planned_chunk_tokens=[len(values)for _,values in plan],
               forward_calls=[],votes=[],sample_calls=0,execute_discard_calls=0,
               completed=False,canceled=None,error=None,
               scope='HOST callbacks from unchanged executor, not GPU hardware counters')
    record.append(entry)  # Partial state exists before the first real callback.
    def observed_forward(logical_past,values,final):
        call=dict(past=logical_past,tokens=len(values),final=final,returned=False)
        entry['forward_calls'].append(call)
        value=forward(logical_past,values,final)
        call['returned']=True
        return value
    def observed_vote():
        value=vote()
        entry['votes'].append(dict(after_successful_forwards=sum(c['returned']for c in entry['forward_calls']),cancel=value))
        return value
    def observed_sample():
        entry['sample_calls']+=1
        return sample()
    def observed_discard():
        entry['execute_discard_calls']+=1
        return discard()
    primary=None
    try:
        result=executor(plan,past,prefill=prefill,window=window,forward=observed_forward,
                        vote=observed_vote,sample=observed_sample,discard=observed_discard)
        entry['completed']=True;entry['canceled']=result[2]
        return result
    except BaseException as error:
        primary=error;entry['error']=type(error).__name__
        raise
    finally:
        # Metadata derivation must never replace a primary Native/control error.
        try:
            counts=[v['after_successful_forwards']for v in entry['votes']]
            groups=[b-a for a,b in zip(counts,counts[1:])]
            entry['actual_committed_group_chunks']=groups
            entry['successful_forward_count']=sum(c['returned']for c in entry['forward_calls'])
            entry['vote_callbacks']=len(entry['votes'])
            entry['last_unvoted_successful_forwards']=entry['successful_forward_count']-(counts[-1]if counts else 0)
            entry['W4_group_larger_than_W2_observed']=window==4 and any(n>2 for n in groups)
            entry['planned_pair_control_can_differ']=len(plan)>2
            entry['complete_prefill_window_counter_qualified']=bool(entry['completed']and not entry['canceled']and entry['sample_calls']==1 and entry['successful_forward_count']==len(plan))
        except BaseException as secondary:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note('r444 trace-finalization failed: '+repr(secondary))
