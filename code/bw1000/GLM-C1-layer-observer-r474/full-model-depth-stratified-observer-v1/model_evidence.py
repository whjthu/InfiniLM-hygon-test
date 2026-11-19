"""Saved sameModel causal evidence; no replay, GPU calls or token-identity quality rule."""
import hashlib,json,math,re,statistics
from attention_evidence import attention_windows
from ffn_evidence import check_frame, routed_windows
from depth_binding import PAIRS
from resident_trial import schedule
from native_arm import LIBRARY,require
H=lambda b:hashlib.sha256(b).hexdigest()
def audit_owner(owner,inp,eos):
    rank=owner['global_rank'];rows=owner['requests'];planned=schedule()
    require(0<=rank<16 and len(rows)==len(planned)==66 and owner['Native_SHA256']==LIBRARY[1] and owner['normal_close'] and owner['Runtime_weak_expired_CABI_assertion'],'MODEL_OWNER_CLOSURE')
    for row,item in zip(rows,planned):
        pin=inp['private_plan']['requests'][item['case_index']]
        require(all(row[k]==pin[k] for k in ('ordinal','case_id','suite','prompt_tokens','prompt_IDs_SHA256','body_SHA256')) and row['arm']==item['arm'] and row['pair']==item['pair'] and row['original_diagnostic']==item['original_diagnostic'] and row['diagnostic']==item['diagnostic'],'ORIGINAL_PAIR_INPUT_IDENTITY')
        ids=row['generated_token_ids'];natural=bool(ids and ids[-1] in eos)
        require(row['completed'] and row['reset_before'] and row['reset_after'] and row['planned_max_new_tokens']==4096 and 0<len(ids)<=4096 and all(type(t)is int and 0<=t<154880 for t in ids) and not any(t in eos for t in ids[:-1]) and (natural or len(ids)==4096) and row['natural_EOS']==natural and row['cap_hit']==(not natural) and H(json.dumps(ids,separators=(',',':')).encode())==row['generated_token_ids_SHA256'],'OUTPUT_DENOMINATOR')
        admission=row['arm_admission'];require(admission['Native_SHA256']==LIBRARY[1] and admission['arm']==row['arm'] and admission['history_value_only_env']==('1' if row['arm']=='ON' else '0') and admission['actual_TP']==16 and admission['actual_PP']==1 and admission['actual_local_heads']==4 and admission['actual_global_rank']==rank and admission['packed_env']=='0','ACTUAL_ARM_ADMISSION')
    diagnostic=rows[1];obs=diagnostic['observer_report'];count=min(3,len(diagnostic['generated_token_ids'])-1)
    require(diagnostic['arm']=='ON' and diagnostic['diagnostic'] and obs['global_tp_rank']==rank and obs['total_frames_used']==len(obs['samples'])==count and obs['frame_cap']==3 and obs['events_allocated']==512 and not obs['pending_next_decode'],'ONE_OBSERVER_LIFETIME')
    exposure=[]
    depth=[]
    for i,frame in enumerate(obs['samples']):
        check_frame(frame,diagnostic,rank,i)
        for layer in attention_windows(frame,rank):
            branches=set(layer['observed_branches']);tuples=[t for t in layer['tuples'] if t['caller_phase']=='attention_history']
            hit=bool({'selection.complete_prefix','v398.enabled','history.dispatch.standard'}<=branches and len(tuples)==1 and tuples[0]['M']==frame['row_past_count'][0][1]+1 and 1<tuples[0]['M']<=8192 and tuples[0]['N']==1792 and tuples[0]['K']==512 and tuples[0]['row_offset']==rank*1792)
            exposure.append(dict(frame=i,layer=layer['layer'],source_derived_new_method_exposure=hit,direct_call_counter=False,branches=sorted(branches),history_tuples=tuples))
            costs = routed_windows(frame, layer['layer'])
            for w in layer['windows']:
                key = 'attention.'+w['name']+'.'+str(w['occurrence'])
                require(key not in costs, 'DUPLICATE_DEPTH_COST_OCCURRENCE')
                costs[key] = dict(host_s=w['host_envelope_seconds'],
                                  enclosing_stream_s=w['stream_envelope_seconds'])
            depth.append(dict(ordinal=i+1,layer=layer['layer'],windows=costs))
    for i,row in enumerate(rows):
        if i!=1:require(not row['diagnostic'] and 'observer_report' not in row and row['observer_mark_calls_before']==row['observer_mark_calls_after'],'OBSERVER_OUTSIDE_DECLARED_ON_DIAGNOSTIC')
    numerical=None
    if rank==0:
        a,b=rows[0]['diagnostic_logits'],rows[1]['diagnostic_logits'];pairs=[]
        for x in a:
            for y in b:
                if all(x[k]==y[k] for k in ('kind','past','prefix_IDs_SHA256')):
                    require(x['bytes']==y['bytes']==154880*2,'FULL_DIAGNOSTIC_LOGITS_BYTES')
                    pairs.append(dict(kind=x['kind'],past=x['past'],same_input=True,bit_hash_equal=x['BF16_logits_SHA256']==y['BF16_logits_SHA256']))
        numerical=dict(comparisons=pairs,final_prefill_present=any(x['kind']=='prefill' for x in pairs),all_compared_equal=bool(pairs) and all(x['bit_hash_equal'] for x in pairs),scope='Only declared same-input full-logit comparisons, not every decoded intermediate.')
    return dict(rank=rank,planned_requests=66,planned_per_arm=33,planned_screen_per_arm=32,completed_requests=len(rows),observed_frames=count,source_derived_exposed_layer_frames=sum(x['source_derived_new_method_exposure'] for x in exposure),exposure=exposure,depth=depth,exposure_state='OBSERVED_SOURCE_DERIVED' if any(x['source_derived_new_method_exposure'] for x in exposure) else 'UNKNOWN_NO_PROVEN_HIT',numerical=numerical,Model_qualified=False,performance_qualified=False)
def audit_owners(owners,inp,eos):
    require(len(owners)==16 and [o['global_rank'] for o in owners]==list(range(16)),'ALL16_OWNER_DENOMINATOR')
    results=[audit_owner(o,inp,eos) for o in owners]
    require(all(x==o['model_value_evidence'] for x,o in zip(results,owners)),'SAVED_EVIDENCE_RECOMPUTATION')
    return dict(planned_ranks=16,planned_frames=48,observed_frames=sum(x['observed_frames'] for x in results),full_coverage=all(x['observed_frames']==3 for x in results),all_ranks_source_exposure=all(x['source_derived_exposed_layer_frames']>0 for x in results),owners=results,numerical=results[0]['numerical'],depth_summary=depth_summary(results),clock_scope='Nested host/stream envelopes; diagnostics excluded from32screen timings; fixed OFF/ON order is disclosed.')


def depth_summary(results):
    """Summarize each distinct layer and phase over up to 16 dependent ranks."""
    require([row['rank'] for row in results] == list(range(16)), 'DEPTH_RANK_DENOMINATOR')
    layers = []
    for ordinal, pair in enumerate(PAIRS, 1):
        for layer in pair:
            observed = {}
            for owner in results:
                rows = [x['windows'] for x in owner['depth']
                        if (x['ordinal'], x['layer']) == (ordinal, layer)]
                require(len(rows) <= 1, 'DUPLICATE_RANK_LAYER_SAMPLE')
                if rows:
                    observed[owner['rank']] = rows[0]
            costs = {}
            names = set().union(*(row.keys() for row in observed.values()))
            for name in sorted(names):
                values = [row[name] for row in observed.values() if name in row]
                host = [w['host_s'] for w in values]
                stream = [w['enclosing_stream_s'] for w in values if w['enclosing_stream_s'] is not None]
                require(all(math.isfinite(t) and t>=0 for t in host + stream), 'INVALID_DEPTH_COST')
                costs[name] = dict(host_rank_samples=len(host), stream_rank_samples=len(stream),
                                   host_median_s=statistics.median(host), host_max_s=max(host),
                                   stream_median_s=statistics.median(stream) if stream else None,
                                   stream_max_s=max(stream) if stream else None)
            layers.append(dict(ordinal=ordinal,layer=layer,observed_ranks=len(observed),
                               planned_ranks=16,missing_ranks=sorted(set(range(16))-set(observed)), costs=costs))
    return dict(layers=layers, planned_layer_rank_frames=96,
                observed_layer_rank_frames=sum(x['observed_ranks'] for x in layers),
                clock_scope='Same-owner nested host/stream envelopes with dependent ranks and possible waits.')


def sampled_metrics(row,eos,special):
    ids=row['generated_token_ids'];ns=row['sampled_ns'];enter=row['request_enter_ns']
    require(type(enter)is int and len(ids)==len(ns) and all(type(t)is int and t>=enter for t in ns) and all(a<b for a,b in zip(ns,ns[1:])), 'SAMPLED_CLOCK_ORDER')
    visible=[t for token,t in zip(ids,ns) if token not in eos and token not in special]
    rate=(len(visible)-1)*1e9/(visible[-1]-visible[0]) if len(visible)>1 else None
    ttft=(visible[0]-enter)/1e9 if visible else None
    require(row['visible_nonEOS_nonspecial']==len(visible) and row['backend_decode_tok_s']==rate and row['backend_sampled_TTFT_s']==ttft,'SAMPLED_CLOCK_RECOMPUTATION')
    return dict(visible_tokens=len(visible),sameowner_backend_decode_tok_s=rate,sameowner_TTFT_s=ttft,HTTP_SLO=False)
