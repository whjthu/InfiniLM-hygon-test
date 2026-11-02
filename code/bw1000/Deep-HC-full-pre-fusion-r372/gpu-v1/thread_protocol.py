"""One process, eight persistent owners; every phase shares one begin/end barrier."""
import threading,time,os,traceback

PHASES=[dict(kind='baseline',mode=0,repeats=1),dict(kind='candidate',mode=1,repeats=1),dict(kind='warm',mode=0,repeats=1),dict(kind='warm',mode=1,repeats=1)]+[dict(kind='measurement',mode=mode,repeats=32,trial=i)for i,mode in enumerate([0,1,1,0]*5)]

def run(setup,execute,finish,phases=PHASES,timeout=120,observe=None):
    """setup/execute/finish run on the same owner thread. No parent affinity changes."""
    ready=threading.Barrier(9);begin=threading.Barrier(9);end=threading.Barrier(9)
    states=[None]*8;errors=[None]*8;results=[None]*8;closes=[None]*8
    current=[None];metadata=[None]*8;records=[];threads=[];started=time.perf_counter()
    def owner(rank):
        metadata[rank]=dict(rank=rank,TID=threading.get_native_id(),initial_allowed=sorted(os.sched_getaffinity(0)))
        try:
            try:states[rank]=setup(rank)
            except BaseException as exc:errors[rank]=dict(stage='setup',error=repr(exc),traceback=traceback.format_exc())
            ready.wait(timeout)
            while True:
                begin.wait(timeout);phase=current[0]
                if phase is None:break
                try:
                    if errors[rank] is None:results[rank]=execute(rank,states[rank],phase)
                except BaseException as exc:errors[rank]=dict(stage=dict(phase),error=repr(exc),traceback=traceback.format_exc())
                finally:end.wait(timeout)
        except BaseException as exc:
            if errors[rank] is None:errors[rank]=dict(stage='barrier',error=repr(exc),traceback=traceback.format_exc())
            ready.abort();begin.abort();end.abort()
        finally:
            try:closes[rank]=finish(rank,states[rank])
            except BaseException as exc:closes[rank]=dict(normal_close=False,error=repr(exc),traceback=traceback.format_exc())
            metadata[rank]['final_allowed']=sorted(os.sched_getaffinity(0))
    for rank in range(8):
        t=threading.Thread(target=owner,args=(rank,),name=f'HC363-owner-{rank}',daemon=False);threads.append(t);t.start()
    protocol_error=None
    try:
        ready.wait(timeout)
        if not any(errors):
            for phase in phases:
                # The process-wide adapter flag is constant from begin through end.
                os.environ['INFINILM_DEEP_HC_FULL_R372']=str(phase['mode'])
                current[0]=dict(phase);results[:]=[None]*8
                tick=time.perf_counter();begin.wait(timeout);end.wait(timeout);toc=time.perf_counter()
                records.append(dict(phase=dict(phase),whole_phase_wall_s=toc-tick,owners=list(results)))
                if observe is not None:observe(records[-1])
                if any(errors):break
    except BaseException as exc:
        protocol_error=repr(exc);ready.abort();begin.abort();end.abort()
    finally:
        current[0]=None
        try:begin.wait(timeout)
        except threading.BrokenBarrierError:pass
        for t in threads:t.join(timeout)
    return dict(passed=not any(errors)and protocol_error is None and len(records)==len(phases)and all(not t.is_alive()for t in threads)and all(c and c.get('normal_close')for c in closes),records=records,owners=metadata,errors=errors,closes=closes,protocol_error=protocol_error,threads_joined=[not t.is_alive()for t in threads],whole_batch_wall_s=time.perf_counter()-started,parent_final_allowed=sorted(os.sched_getaffinity(0)))

def aligned_measurements(records):
    """No cross-trial maxima: each envelope uses exactly the same trial on all owners."""
    out=[]
    for record in records:
        phase=record['phase']
        if phase['kind']!='measurement':continue
        owners=record['owners'];assert len(owners)==8 and all(v and v['trial']==phase['trial']and v['mode']==phase['mode']and v['repeats']==32 for v in owners)
        out.append(dict(trial=phase['trial'],mode=phase['mode'],max8_CAPI_wall_s=max(v['CAPI_wall_s']for v in owners),max8_native_component_s=max(v['native_component_s']for v in owners),max8_local_loop_wall_s=max(v['local_loop_wall_s']for v in owners),whole_phase_wall_s=record['whole_phase_wall_s'],owner_loop_envelope_s=max(v['loop_end_s']for v in owners)-min(v['loop_begin_s']for v in owners),aligned_repeat_sum_max8_CAPI_wall_s=sum(max(v['repeat_CAPI_wall_s'][i]for v in owners)for i in range(32)),owners=owners))
    assert [v['trial']for v in out]==list(range(len(out)))
    return out
