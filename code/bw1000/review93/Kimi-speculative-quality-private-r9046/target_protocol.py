"""True pinned-draft -> fast-W8 target -> prefix commit/replay online loop."""
import ctypes as C,json,pathlib,time,os,re
from policy import greedy
from live_taps import LiveTaps,assemble
from wire import connect,listen,send,recv

def startup(lib):
    from target_binding import startup as check
    return check(lib)
def run(g):
    import numpy as np
    lib,ctx,r,control,report,save=[g[k]for k in ['lib','ctx','r','control','report','save']];node,tail,emitter,vocab=[g[k]for k in ['node','tail','emitter','vocab']]
    report.update(scope='TruepinnedDFlash2span8 + originalfastW8 Math0, greedyprefix acceptance; no historicalsequence identity requirement; realdecodecycle excludesprefill',speculative_quality_cases=[],phase_calls=[],lossless_sampling_claim=False)
    live=LiveTaps(lib,ctx,node);peer=draft=listener=None;generation=0;serial=0;rpcseq=0;nonce=os.environ['KIMI_DRAFT_NONCE_R9046'];deadline=g['experiment_started_r9046']+2500
    def ck(code):
        if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
    def rpc(cmd,body=b'',**fields):
        nonlocal rpcseq
        tick=time.perf_counter();send(draft,{'cmd':cmd,'seq':rpcseq,**fields},body);reply,blob=recv(draft);assert reply['seq']==rpcseq and not blob;rpcseq+=1;return reply,time.perf_counter()-tick
    def mode(ordinary,label):
        nonlocal generation
        out=C.c_uint64();ck(lib.modern_kimi_pair_mode_r9007(ctx,int(ordinary),C.byref(out)));generation=out.value;packets=control.collect('mode:'+label,{'generation':generation,'ordinary':ordinary});assert all(p['generation']==generation and p['ordinary']==ordinary for p in packets)
    def rows():
        out=(C.c_int32*4)();ck(lib.modern_kimi_pair_rows_r9006(ctx,1,out));return list(out)
    def call(ids,past,phase,capture,on_selected=None):
        nonlocal serial
        ids=np.asarray(ids,np.int64);N=len(ids);n_out=1 if phase in ['ordinary','prefill']else N
        row=np.asarray([1],np.int32);pp=np.asarray([past],np.int32);total=np.asarray([past+N],np.int32);off=np.asarray([0,N],np.int32);pos=np.arange(past,past+N,dtype=np.int64);logits=np.full((n_out,vocab),0x7fc0,np.uint16);out=np.full(n_out,-1,np.int64);flags=np.asarray([-1],np.int32);ptr=lambda a:C.c_void_p(a.ctypes.data)
        live.begin(capture);tick=time.perf_counter()
        if phase=='ordinary':ck(lib.modern_kimi_pair_ordinary_forward_r9007(ctx,generation,N,1,3,*[ptr(a)for a in [row,pp,total,off,ids,pos,logits,out,flags]]))
        elif phase=='prefill':ck(lib.modern_kimi_pair_prefill_r9006(ctx,N,ptr(row),ptr(ids),ptr(pos),ptr(logits),ptr(out),ptr(flags)))
        else:
            active,shadow,actual_past,state=rows();assert actual_past==past and state in [1,3];read=np.asarray([active],np.int32);write=np.asarray([shadow],np.int32);ck(lib.modern_kimi_pair_forward_r9006(ctx,N,1,3,*[ptr(a)for a in [read,write,row,pp,total,off,ids,pos,logits,out,flags]]))
        wall=time.perf_counter()-tick;assert flags[0]==int(tail)+2*int(node==emitter)
        selected=None
        if tail:
            assert not np.any((logits&0x7f80)==0x7f80),'nonfinite target logits'
            if node==emitter:selected=out.tolist();assert all(0<=x<vocab for x in selected)
        packets=control.collect('target-call:'+str(serial),{'selected':selected,'CAPI_s':wall});selected=list(packets[emitter]['selected']);assert len(selected)==n_out
        if on_selected is not None:on_selected(selected)
        features=None;featuretick=time.perf_counter()
        if capture:
            own=live.read(past,N)
            if node==2:send(peer,{'call':serial,**own[0]},own[1])
            elif node==0:
                other,body=recv(peer);assert other['call']==serial and other['node']==2;features=assemble([own,(other,body)],past,N)
            else:assert own[0]['records']==[]and not own[1]
        featurewall=time.perf_counter()-featuretick
        result={'serial':serial,'phase':phase,'past':past,'input_width':N,'CAPI_maxhost_s':max(p['CAPI_s']for p in packets),'feature_harvest_local_s':featurewall,'capture':capture,'selected':selected.copy()};report['phase_calls'].append(result);serial+=1
        return selected,features,result
    def set_draft_reset(label):
        reply=None
        if node==0:reply,wall=rpc('reset');assert reply['reset']
        control.collect('draft-reset:'+label,{'node':node})
    def draft_query(features,past,n,anchor,K,label):
        packet=None
        if node==0:
            reply,rpcwall=rpc('query',features,past=past,rows=n,anchor=anchor,K=K);assert reply['greedy_only']and len(reply['proposals'])==K;packet={**reply,'RPC_wall_s':rpcwall}
        packets=control.collect('draft-query:'+label,{'packet':packet});return packets[0]['packet']
    def decode(inp,label,K,cap):
        mode(K==0,label);set_draft_reset(label);first_event={}
        def first_emit(tokens):first_event.update(stamp=time.perf_counter(),tokens=list(tokens))
        begin=time.perf_counter();z,features,pre=call(inp,0,'ordinary'if K==0 else'prefill',K>0,on_selected=first_emit);first=first_event['stamp'];p=len(inp);context_past=0;append_rows=len(inp);cycles=[];timestamps=[first];native_s=0
        while len(z)<cap and z[-1]not in eos:
            tick=time.perf_counter();event={}
            def ordinary_emit(tokens):event.update(stamp=time.perf_counter(),tokens=list(tokens))
            if K==0:
                emitted,_,metrics=call([z[-1]],p,'ordinary',False,on_selected=ordinary_emit);p+=1;row={'accepted':0,'proposal_count':0,'verify_s':metrics['CAPI_maxhost_s'],'draft_RPC_s':0,'draft_query_s':0,'append_H2D_s':0,'replay_s':0,'commit_local_s':0,'useful_emitted':sum(x not in eos for x in emitted)};native_s+=metrics['CAPI_maxhost_s']
            else:
                packet=draft_query(features,context_past,append_rows,z[-1],K,label+':'+str(len(cycles)));proposal=packet['proposals'];context_past+=append_rows;before=rows()
                def verify_emit(target_ids):
                    decision=greedy(z[-1],proposal,target_ids,eos,cap-len(z));event.update(decision=decision,tokens=list(decision.emitted),stamp=time.perf_counter())
                target,newfeatures,metrics=call([z[-1]]+proposal,p,'verify',True,on_selected=verify_emit);decision=event['decision'];emitted=list(decision.emitted);commit_tick=time.perf_counter();ck(lib.modern_kimi_pair_commit_r9006(ctx,1,decision.accepted,int(not decision.terminal)));commitwall=time.perf_counter()-commit_tick;replaywall=0
                if decision.needs_replay:
                    replay,newfeatures,replaymetrics=call(decision.commit_inputs,p,'replay',True);replaywall=replaymetrics['CAPI_maxhost_s']
                    # No replay-vs-verify token identity gate: execution shapes may round differently.
                if not decision.terminal:assert rows()[:3]==[before[1],before[0],p+len(decision.commit_inputs)]and rows()[3]==1
                else:assert rows()[3]==4
                p+=len(decision.commit_inputs);append_rows=len(decision.commit_inputs);features=newfeatures
                row={'accepted':decision.accepted,'proposal_count':K,'verify_s':metrics['CAPI_maxhost_s'],'draft_RPC_s':packet['RPC_wall_s'],'draft_query_s':packet['draft_query_s'],'append_H2D_s':packet['append_H2D_s'],'replay_s':replaywall,'commit_local_s':commitwall,'needs_replay':decision.needs_replay,'terminal':decision.terminal,'useful_emitted':sum(x not in eos for x in emitted)};native_s+=metrics['CAPI_maxhost_s']+replaywall
            # Emit callback precedes feature transport/commit/replay; their cost remains in
            # the next ITL and total decode wall. No retroactive token timestamp.
            emitted_time=event['stamp'];assert event['tokens']==emitted;z.extend(emitted);timestamps.extend([emitted_time]*len(emitted));row['emit_since_cycle_start_s']=emitted_time-tick;row['cycle_user_wall_s']=time.perf_counter()-tick;row['emit_burst']=len(emitted);cycles.append(row)
            if len(cycles)%8==0:report['phase']=label+':'+str(len(z));save()
            if time.monotonic()>deadline:raise TimeoutError('reserved cleanup deadline reached within case')
        end=time.perf_counter();useful=sum(x not in eos for x in z);decode_useful=sum(c['useful_emitted']for c in cycles);decode_wall=end-first;gaps=[b-a for a,b in zip(timestamps,timestamps[1:])];q={'label':label,'K':K,'input_tokens':len(inp),'generated_ids':z,'generated_tokens':len(z),'EOS':z[-1]in eos,'truncated':z[-1]not in eos and len(z)>=cap,'prefill_local_wall_s':first-begin,'decode_user_wall_s':decode_wall,'useful_decode_tokens':decode_useful,'decode_user_tok_s':decode_useful/decode_wall if decode_wall else None,'first_last_user_tok_s':(useful-1)/(timestamps[useful-1]-timestamps[0])if useful>1 else None,'E2E_useful_output_tok_s':useful/(end-begin),'native_target_window_s':native_s,'native_target_only_tok_s':decode_useful/native_s if native_s else None,'cycles':cycles,'mean_useful_outputs_per_cycle':decode_useful/len(cycles)if cycles else None,'proposal_acceptance':sum(c['accepted']for c in cycles)/(K*len(cycles))if K and cycles else None,'ITL_max_s':max(gaps)if gaps else None,'ITL_P50_s':float(np.percentile(gaps,50))if gaps else None,'ITL_P95_s':float(np.percentile(gaps,95))if gaps else None,'ITL_P99_s':float(np.percentile(gaps,99))if gaps else None,'clock_scope':'single emitter token-ready/emit clock; not HTTP; prefill excluded from decode; bursts count real emitted tokens'};report['speculative_quality_cases'].append(q);save();return q
    fixed=json.loads((r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_text());eos=set(fixed['eos_token_ids'])
    try:
        if node==0:
            listener=listen(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));peer,_=listener.accept();hello,body=recv(peer);assert hello=={'cmd':'feature-peer','nonce':nonce}and not body;send(peer,{'ready':True});draft=connect(os.environ['KIMI_DRAFT_HOST_R9046'],int(os.environ['KIMI_DRAFT_PORT_R9046']));send(draft,{'cmd':'hello','nonce':nonce});ready,body=recv(draft);assert ready['ready']and ready['block_size']==8 and not body;report['draft_handshake']=ready
        elif node==2:
            peer=connect(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));send(peer,{'cmd':'feature-peer','nonce':nonce});ready,body=recv(peer);assert ready['ready']and not body
        control.collect('live-transport-ready',{'node':node});report['transport']='persistentbinaryTCPonib0;targetcollectivesremainIBext;CPUstagingnotGDR';save()
        # Two input families, one loaded target and one loaded drafter, no token-equality gate.
        for prompt_index,inp in enumerate(fixed['natural_input_ids'][:2]):
            for K in [0,2,4,7]:
                budget=control.collect('case-budget:natural'+str(prompt_index)+'K'+str(K),{'allow':time.monotonic()+80<=deadline if node==0 else None})
                if not budget[0]['allow']:report.setdefault('skipped_budget',[]).append({'prompt':prompt_index,'K':K});continue
                decode(inp,'natural'+str(prompt_index)+'-K'+str(K),K,64)
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(g['model']),trust_remote_code=True);samplefile=pathlib.Path(__file__).with_name('quality_samples.json');samples=json.loads(samplefile.read_text());report['quality_sample_source']=samples['source'];report['quality_smoke_only']=True
        for i,sample in enumerate(samples['samples']):
            inp=tok.apply_chat_template([{'role':'user','content':sample['question']+'\nSolve this problem. End with #### followed by the numeric answer.'}],tokenize=True,add_generation_prompt=True)
            if not 1<=len(inp)<=512:raise ValueError('quality input outside bounded capability')
            for K in [0,7]:
                budget=control.collect('case-budget:quality'+str(i)+'K'+str(K),{'allow':time.monotonic()+120<=deadline if node==0 else None})
                if not budget[0]['allow']:report.setdefault('skipped_quality_budget',[]).append({'sample':i,'K':K});continue
                q=decode(inp,'gsm8k'+str(i)+'-K'+str(K),K,256);text=tok.decode(q['generated_ids'],skip_special_tokens=True);matches=re.findall(r'####\s*(-?[0-9][0-9,.]*)',text);answer=matches[-1].replace(',','').rstrip('.')if matches else None;q['quality']={'task':'GSM8K','sample_index':i,'expected_answer':sample['answer'],'extracted_answer':answer,'correct':answer==sample['answer'],'response':text,'truncated':q['truncated'],'scope':'tiny smoke; not a benchmark no-regression proof'};save()
        report.update(serving_protocol_complete=True,trained_spec_speed_claim='MEASURED_IN_REPORT_PENDING_INDEPENDENT_AUDIT',full_sampling_lossless_claim=False,quality_gate='taskanswers_and_later_benchmark; no oldARtokenidentity');save()
    finally:
        live.begin(False)
        if node==0 and draft is not None:
            try:rpc('close')
            finally:draft.close()
        if peer is not None:peer.close()
        if listener is not None:listener.close()
