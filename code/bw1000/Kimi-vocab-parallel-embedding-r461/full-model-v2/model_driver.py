"""Typed AR-only C1 driver. No pair, capture, replay, speculative commit or HTTP."""
import ctypes as C
import hashlib,json,time

SYMBOL='modern_kimi_engine_forward_ar_storage_r456'
compact=lambda v:json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()
H=lambda v:hashlib.sha256(v).hexdigest()

def declare(lib):
    f=getattr(lib,SYMBOL);f.argtypes=[C.c_void_p,C.c_int,C.c_size_t,C.c_size_t,C.c_int]+[C.c_void_p]*9;f.restype=C.c_int
    for name in ['load','finish','reset']:
        fn=getattr(lib,'modern_kimi_engine_'+name+'_r2475');fn.argtypes=[C.c_void_p]+([C.c_char_p]if name=='load'else[]);fn.restype=C.c_int
    lib.modern_kimi_shard_head_expected_r9183.argtypes=[C.c_char_p];lib.modern_kimi_shard_head_expected_r9183.restype=C.c_int
    lib.modern_kimi_shard_head_error_r9183.restype=C.c_char_p
    return f

def generate(lib,ctx,node,control,prompt,cap,deadline,record,memory_guard=None):
    import numpy as np
    assert prompt and len(prompt)+cap<=32768 and cap==4096
    assert all(type(v)is int and 0<=v<163840 for v in prompt)
    f=declare(lib);calls=[];generated=[];ticks=[];past=0
    def check(rc):
        if rc:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
    def reset(label):
        check(lib.modern_kimi_engine_reset_r2475(ctx));control.collect(label,dict(node=node))
    def call(kind,ids):
        assert kind in(0,1)and 0<len(ids)<=512 and(kind==0 or len(ids)==1 and past>0)
        assert time.monotonic()<deadline,'Original service deadline; no extension'
        arrays=[np.array([1],np.int32),np.array([past],np.int32),np.array([past+len(ids)],np.int32),np.array([0,len(ids)],np.int32),np.array(ids,np.int64),np.arange(past,past+len(ids),dtype=np.int64),np.full((1,163840),0x7fc0,np.uint16),np.array([-1],np.int64),np.array([-1],np.int32)]
        record('forward_before',dict(serial=len(calls),kind=kind,past=past,width=len(ids),IDs_SHA256=H(compact(ids))))
        started=time.perf_counter();check(f(ctx,kind,len(ids),1,3,*[C.c_void_p(a.ctypes.data)for a in arrays]));wall=time.perf_counter()-started
        logits,out,flags=arrays[-3:];assert int(flags[0])==(3 if node==0 else 1)
        assert not np.any((logits&0x7f80)==0x7f80),'Nonfinite actual full Model BF16 logits'
        packet=dict(node=node,kind=kind,past=past,width=len(ids),IDs_SHA256=H(compact(ids)),wall_s=wall,selected=int(out[0])if node==0 else None,logits_SHA256=H(logits.tobytes()))
        peers=control.collect('typed-forward:'+str(len(calls)),packet)
        assert len({v['IDs_SHA256']for v in peers})==1,'All TP node inputs must agree'
        token=peers[0]['selected'];assert type(token)is int and 0<=token<163840
        calls.append(dict(kind=kind,past=past,width=len(ids),CAPI_maxhost_s=max(v['wall_s']for v in peers),per_node=peers,TP_logits_bitwise_equal_diagnostic=len({v['logits_SHA256']for v in peers})==1));record('forward_completed',calls[-1]);return token
    primary=None;ended='exception';started=time.monotonic()
    reset('request-reset-before')
    try:
        while past<len(prompt):
            part=prompt[past:past+512];token=call(0,part);past+=len(part)
            if memory_guard:memory_guard('prefill:'+str(past))
            record('prefill_progress',dict(committed=past,total=len(prompt),calls=len(calls)))
        prefill_done=time.monotonic()
        generated.append(token);ticks.append(prefill_done);record('emit',dict(token=token,index=0,monotonic=prefill_done))
        while token not in{163585,163586}and len(generated)<cap:
            token=call(1,[token]);past+=1;generated.append(token);ticks.append(time.monotonic());record('emit',dict(token=token,index=len(generated)-1,monotonic=ticks[-1]))
            if memory_guard and len(generated)%64==0:memory_guard('decode:'+str(len(generated)))
        ended='eos'if token in{163585,163586}else'length'
    except BaseException as error:primary=error;raise
    finally:
        try:reset('request-reset-after')
        except BaseException as cleanup:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note('reset: '+repr(cleanup))
    if memory_guard:memory_guard('request-end')
    visible=[t for t,v in zip(ticks,generated)if v not in{163585,163586}]
    return dict(generated_ids=generated,emit_monotonic=ticks,calls=calls,finish=ended,natural_EOS=ended=='eos',cap_hit=ended=='length',normal_reset=True,prompt_tokens=len(prompt),prompt_IDs_SHA256=H(compact(prompt)),prefill_wall_s=prefill_done-started,backend_TTFT_from_generation_start_s=ticks[0]-started,visible_tokens=len(visible),backend_visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 and visible[-1]>visible[0]else None,scope='Native driver enqueue/start to all-node return/emit, not HTTP or public client latency')
