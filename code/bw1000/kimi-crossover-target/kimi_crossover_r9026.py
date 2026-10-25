"""Fixed causal crossover; diagnostic only, never speculative acceptance or TPS.

Native trace is mandatory. Call startup(lib) immediately after binding, BEFORE
engine create/weight loading. root must bind the exact compiled provider first.
"""
import ctypes
import hashlib
import json
import time

ARMS = {'A': [1]*32, 'B': [1]*32, 'C': [1]*24+[8],
        'D': [8]*3+[1]*8, 'E': [8]*4}
POSITIONS = (536, 537)


def read_pair_rows(lib, ctx, *, ordinary):
    """The controlled ordinary epoch deliberately rejects all pair-state APIs."""
    if ordinary:
        return None
    out = (ctypes.c_int32 * 4)()
    if lib.modern_kimi_pair_rows_r9006(ctx, 1, out):
        raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
    return list(out)


def blocks(arm):
    past = 512
    for width in ARMS[arm]:
        yield past, width
        past += width
    assert past == 544


def startup(lib):
    """Reject absent hooks before loading. No GPU/ctx creation in this function."""
    names = ('modern_kimi_diag_capabilities_r9008', 'modern_kimi_diag_arm_r9008',
             'modern_kimi_diag_read_r9008')
    for name in names:
        if not hasattr(lib, name):
            raise RuntimeError('crossover trace hook missing BEFORE weights: '+name)
    cap = getattr(lib, names[0]); cap.argtypes=[]; cap.restype=ctypes.c_char_p
    arm = getattr(lib, names[1]); arm.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]; arm.restype=ctypes.c_int
    read = getattr(lib, names[2]); read.argtypes=[ctypes.c_void_p]; read.restype=ctypes.c_char_p
    contract = json.loads(cap())
    required = ('actual_final_hidden', 'actual_layer_boundaries',
                'actual_recurrent_state_metadata', 'actual_gpu_calls')
    if contract.get('abi') != 'kimi-crossover-r9008-v1' or any(contract.get(k) is not True for k in required):
        raise RuntimeError('crossover trace hook not ready BEFORE weights: '+json.dumps(contract))
    if contract.get('positions') != list(POSITIONS):
        raise RuntimeError('wrong diagnostic position contract BEFORE weights')
    return contract


def run(g):
    import numpy as np
    lib,ctx,r,control,report,save=[g[k] for k in ('lib','ctx','r','control','report','save')]
    node,tail,emitter,vocab=[g[k] for k in ('node','tail','emitter','vocab')]
    # A missing pre-weight startup receipt is a launcher error, not permission to
    # discover hooks only after an expensive load.
    cap=g.get('crossover_preweight_capabilities')
    if cap is None: raise RuntimeError('mandatory pre-weight crossover startup receipt missing')
    if cap != startup(lib): raise RuntimeError('trace capability changed after loading')
    def ck(code):
        if code: raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
    def rows():
        return read_pair_rows(lib,ctx,ordinary=False)
    def trace():
        raw=lib.modern_kimi_diag_read_r9008(ctx)
        if raw is None: raise RuntimeError('trace read returned NULL')
        # Native owns returned memory until next diagnostic/native call. Copy and
        # parse now; no raw pointer is persisted in report.
        value=json.loads(bytes(raw))
        if value.get('error'):raise RuntimeError('native trace error: '+str(value['error']))
        if value.get('finite') is not True:raise RuntimeError('non-finite or missing native trace finite receipt')
        return value
    f=r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json'
    if hashlib.sha256(f.read_bytes()).hexdigest()!='ab686988a0952fe094d76773902e784e5d4be577c44c57c23493d702cc1d5625':
        raise RuntimeError('fixed prompt receipt changed')
    prompt=json.loads(f.read_text())['natural_input_ids'][0]
    assert len(prompt)==512
    report.update(crossover_arms=[],crossover_calls=[],trained_spec_speed_claim=False,
                  oracle_is_diagnostic_only=True,scope='fixed512 teacherforced A-E causal crossover; no acceptance/serving/performance qualification',
                  crossover_trace_capabilities=cap)
    generation=0;serial=0
    def mode(arm):
        nonlocal generation
        out=ctypes.c_uint64();ordinary=arm=='A'
        ck(lib.modern_kimi_pair_mode_r9007(ctx,int(ordinary),ctypes.byref(out)));generation=out.value
        ck(lib.modern_kimi_diag_arm_r9008(ctx,arm.encode(),1))
        packets=control.collect('crossover-mode:'+arm,{'generation':generation,'ordinary':ordinary})
        assert all(p['generation']==generation and p['ordinary']==ordinary for p in packets)
    def call(arm,ids,past,prefill=False):
        nonlocal serial
        ids=np.asarray(ids,dtype=np.int64);N=len(ids);paired=arm!='A' and not prefill;n_out=N if paired else 1
        row=np.array([1],np.int32);pp=np.array([past],np.int32);total=np.array([past+N],np.int32)
        off=np.array([0,N],np.int32);pos=np.arange(past,past+N,dtype=np.int64)
        before=read_pair_rows(lib,ctx,ordinary=arm=='A')
        read=np.array([before[0] if before is not None else 1],np.int32)
        write=np.array([before[1] if before is not None else 9],np.int32)
        logits=np.full((n_out,vocab),0x7fc0,np.uint16);out=np.full(n_out,-1,np.int64);flags=np.array([-1],np.int32)
        arrays={'read':read.tolist() if paired else None,'write':write.tolist() if paired else None,
                'logical_KV_rows':row.tolist(),'past':pp.tolist(),'total':total.tolist(),
                'offsets':off.tolist(),'input_ids':ids.tolist(),'positions':pos.tolist(),
                'flags_argument':3,'count_argument':1,'tokens_argument':N,
                'dtypes':{'read_write_rows_past_total_offsets':'int32','ids_positions':'int64','logits':'BF16 raw uint16','output':'int64'}}
        pre_trace=trace();ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
        start=time.perf_counter()
        if arm=='A':
            ck(lib.modern_kimi_pair_ordinary_forward_r9007(ctx,generation,N,1,3,*[ptr(a) for a in (row,pp,total,off,ids,pos,logits,out,flags)]))
        elif prefill:
            ck(lib.modern_kimi_pair_prefill_r9006(ctx,N,*[ptr(a) for a in (row,ids,pos,logits,out,flags)]))
        else:
            assert before[2]==past and before[3]==1
            ck(lib.modern_kimi_pair_forward_r9006(ctx,N,1,3,*[ptr(a) for a in (read,write,row,pp,total,off,ids,pos,logits,out,flags)]))
        seconds=time.perf_counter()-start
        assert int(flags[0])==int(tail)+2*int(node==emitter)
        after_verify=read_pair_rows(lib,ctx,ordinary=arm=='A');post_trace=trace();selected=None;head=[]
        if tail:
            fp=(logits.astype(np.uint32)<<16).view(np.float32)
            if not np.isfinite(fp).all():raise RuntimeError('non-finite actual BF16 head logits')
            computed=np.argmax(fp,axis=-1).tolist()
            if node==emitter:
                if out.tolist()!=computed:raise RuntimeError('CAPI IDs do not match actual BF16 logits argmax')
                selected=out.tolist()
            for j in range(n_out):
                position=past+j if paired else past+N-1
                if position in POSITIONS:
                    order=np.lexsort((np.arange(vocab),-fp[j]))[:2]
                    head.append({'input_position':position,'generated_index':position-511,
                                 'top2_ids':order.tolist(),'top2_BF16_bits':logits[j,order].tolist(),
                                 'top2_values':fp[j,order].tolist(),'margin':float(fp[j,order[0]]-fp[j,order[1]]),
                                 'logits_8094_17375':fp[j,[8094,17375]].tolist(),
                                 'logits_8094_17375_BF16_bits':logits[j,[8094,17375]].tolist()})
        packets=control.collect('crossover-call:'+str(serial),{'node':node,'selected':selected,'seconds':seconds,
                                'head':head,'trace_before':pre_trace,'trace_after':post_trace,'flags':int(flags[0])})
        selected=packets[emitter]['selected'];assert selected is not None and len(selected)==n_out
        after_commit=None
        if paired:
            # Full accept is an explicit causal intervention. It does NOT mean
            # proposals matched outputs and must never become acceptance data.
            ck(lib.modern_kimi_pair_commit_r9006(ctx,1,N-1,1));after_commit=rows()
            assert after_commit==[before[1],before[0],past+N,1]
        entry={'arm':arm,'serial':serial,'phase':'prefill' if prefill else 'decode','arrays':arrays,
               'rows_before':before,'rows_after_forward':after_verify,'rows_after_full_commit':after_commit,
               'actual_computed_greedy_ids':selected,'native_forward_calls_this_owner':1,
               'actual_gpu_calls_source':'actual native diagnostic callback invocation count; not hardware kernel count',
               'owners':packets,'diagnostic_maxhost_seconds':max(p['seconds'] for p in packets),
               'full_accept_causal_control':paired}
        report['crossover_calls'].append(entry);serial+=1;save();return selected
    reference=None
    for arm in ARMS:
        mode(arm);anchor=call(arm,prompt,0,True);computed=[]
        for past,width in blocks(arm):
            start=past-512
            # A alone constructs reference. Every other arm consumes frozen A
            # inputs even after a numerical output mismatch.
            inp=[anchor[-1] if not computed else computed[-1]] if arm=='A' else reference[start:start+width]
            computed.extend(call(arm,inp,past))
        if arm=='A':reference=anchor+computed;assert len(reference)==33
        expected=reference[1:]
        mismatches=[{'generated_index':j+1,'input_position':512+j,'actual':x,'ordinary':y}
                    for j,(x,y) in enumerate(zip(computed,expected)) if x!=y]
        report['crossover_arms'].append({'arm':arm,'widths':ARMS[arm],'anchor':anchor,'actual_ids':computed,
                 'expected_A_ids':expected,'anchor_equal_A':anchor==reference[:1],
                 'ID_differences':mismatches,'all_IDs_equal_A':computed==expected,
                 'fixed_A_teacherforced_inputs':arm!='A','real_acceptance_claim':False})
        report['phase']='crossover:'+arm+':complete';save()
    ck(lib.modern_kimi_diag_arm_r9008(ctx,b'closed',0))
    report.update(crossover_complete=True,crossover_all_IDs_equal=all(a['all_IDs_equal_A'] for a in report['crossover_arms']),
                  serving_protocol_complete=True,allposition_finalIDs_EOS_pass=False,
                  untested=['forced rejection','replay','page255','EOS terminal skip','post-crossover original AR restore'],
                  performance_claim=False)
    save()
