"""True pinned-draft -> fast-W8 target -> prefix commit/replay online loop."""
import ctypes as C,json,pathlib,time,os,re,hashlib
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
    live=LiveTaps(lib,ctx,node);peer=draft=listener=None;generation=0;serial=0;rpcseq=0;nonce=os.environ['KIMI_DRAFT_NONCE_R9046'];deadline=g['experiment_started_r9046']+int(os.environ.get('KIMI_PROTOCOL_BUDGET_R9116','5000'));drafts={};active_draft='dflash2';baselines={};case_serial=0;command_ids=set()
    for name,args in [('modern_kimi_state_mode_r9114',[C.c_void_p,C.c_int]),('modern_kimi_state_replay_r9114',[C.c_void_p,C.c_int]),('modern_kimi_state_stats_r9114',[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t])]:getattr(lib,name).argtypes=args
    def ck(code):
        if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
    def rpc(cmd,body=b'',**fields):
        nonlocal rpcseq
        tick=time.perf_counter();sock,seq=drafts[active_draft];send(sock,{'cmd':cmd,'seq':seq,**fields},body);reply,blob=recv(sock);assert reply['seq']==seq and not blob;drafts[active_draft]=(sock,seq+1);return reply,time.perf_counter()-tick
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
                other,body=recv(peer);assert other['call']==serial and other['node']==2;features=assemble([own,(other,body)],past,N,live.modal)
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
    def decode(inp,label,K,cap,strategy='oldspec',baseline=None):
        state_on=strategy in ['state','adaptive'];mode(K==0,label);ck(lib.modern_kimi_state_mode_r9114(ctx,int(state_on)));set_draft_reset(label)if K else None;first_event={};fallback=False;fallback_at=None;adaptive_wall=0.;adaptive_useful=0;spec_cycles=0
        def first_emit(tokens):first_event.update(stamp=time.perf_counter(),tokens=list(tokens))
        begin=time.perf_counter();z,features,pre=call(inp,0,'ordinary'if K==0 else'prefill',K>0,on_selected=first_emit);first=first_event['stamp'];p=len(inp);context_past=0;append_rows=len(inp);cycles=[];timestamps=[first];native_s=0
        while len(z)<cap and z[-1]not in eos:
            tick=time.perf_counter();event={};cycle_K=min(K,cap-len(z)-1,max(0,1023-p))if K else 0
            def ordinary_emit(tokens):event.update(stamp=time.perf_counter(),tokens=list(tokens))
            if K==0:
                emitted,_,metrics=call([z[-1]],p,'ordinary',False,on_selected=ordinary_emit);p+=1;row={'accepted':0,'proposal_count':0,'verify_s':metrics['CAPI_maxhost_s'],'draft_RPC_s':0,'draft_query_s':0,'append_H2D_s':0,'replay_s':0,'commit_local_s':0,'useful_emitted':sum(x not in eos for x in emitted)};native_s+=metrics['CAPI_maxhost_s']
            elif fallback or cycle_K==0:
                before=rows();assert before[2]==p and before[3]==1
                def pair_ar_emit(target_ids):
                    decision=greedy(z[-1],[],target_ids,eos,cap-len(z));event.update(decision=decision,tokens=list(decision.emitted),stamp=time.perf_counter())
                _,_,metrics=call([z[-1]],p,'pair_AR_continuation',False,on_selected=pair_ar_emit);decision=event['decision'];emitted=list(decision.emitted);ct=time.perf_counter();ck(lib.modern_kimi_pair_commit_r9006(ctx,1,0,int(not decision.terminal)));commitwall=time.perf_counter()-ct;p+=1
                assert rows()[3]==(4 if decision.terminal else 1)
                row={'accepted':0,'proposal_count':0,'verify_s':metrics['CAPI_maxhost_s'],'draft_RPC_s':0,'draft_query_s':0,'append_H2D_s':0,'replay_s':0,'commit_local_s':commitwall,'useful_emitted':sum(x not in eos for x in emitted),'pair_AR_continuation':True};native_s+=metrics['CAPI_maxhost_s']
            else:
                packet=draft_query(features,context_past,append_rows,z[-1],cycle_K,label+':'+str(len(cycles)));proposal=packet['proposals'];context_past+=append_rows;before=rows()
                def verify_emit(target_ids):
                    decision=greedy(z[-1],proposal,target_ids,eos,cap-len(z));event.update(decision=decision,tokens=list(decision.emitted),stamp=time.perf_counter())
                target,newfeatures,metrics=call([z[-1]]+proposal,p,'verify',True,on_selected=verify_emit);decision=event['decision'];emitted=list(decision.emitted);commit_tick=time.perf_counter();ck(lib.modern_kimi_pair_commit_r9006(ctx,1,decision.accepted,int(not decision.terminal)));commitwall=time.perf_counter()-commit_tick;replaywall=0
                if decision.needs_replay:
                    if state_on:
                        st=time.perf_counter();ck(lib.modern_kimi_state_replay_r9114(ctx,1));statewall=time.perf_counter()-st;packets=control.collect('state-replay:'+str(serial),{'wall':statewall});replaywall=max(v['wall']for v in packets)
                        if node==0:newfeatures=newfeatures[:len(decision.commit_inputs)*(43008 if live.modal else 35840)*2]
                    else:
                        replay,newfeatures,replaymetrics=call(decision.commit_inputs,p,'replay',True);replaywall=replaymetrics['CAPI_maxhost_s']
                    # Shape-dependent roundoff is not a historical token identity gate.
                if not decision.terminal:assert rows()[:3]==[before[1],before[0],p+len(decision.commit_inputs)]and rows()[3]==1
                else:assert rows()[3]==4
                p+=len(decision.commit_inputs);append_rows=len(decision.commit_inputs);features=newfeatures
                row={'accepted':decision.accepted,'proposal_count':cycle_K,'verify_s':metrics['CAPI_maxhost_s'],'draft_RPC_s':packet['RPC_wall_s'],'draft_query_s':packet['draft_query_s'],'append_H2D_s':packet['append_H2D_s'],'replay_s':replaywall,'commit_local_s':commitwall,'needs_replay':decision.needs_replay,'terminal':decision.terminal,'useful_emitted':sum(x not in eos for x in emitted)};native_s+=metrics['CAPI_maxhost_s']+replaywall
            # Emit callback precedes feature transport/commit/replay; their cost remains in
            # the next ITL and total decode wall. No retroactive token timestamp.
            emitted_time=event['stamp'];assert event['tokens']==emitted;z.extend(emitted);timestamps.extend([emitted_time]*len(emitted));row['emit_since_cycle_start_s']=emitted_time-tick;row['cycle_user_wall_s']=time.perf_counter()-tick;row['emit_burst']=len(emitted);row['state_only_replay']=bool(state_on and not fallback and row.get('needs_replay'));cycles.append(row)
            if strategy=='adaptive'and not fallback and row['proposal_count']>0:
                spec_cycles+=1;adaptive_wall=time.perf_counter()-first;adaptive_useful+=row['useful_emitted'];decision_packet=None
                if node==emitter:decision_packet={'switch':spec_cycles>=2 and adaptive_useful/adaptive_wall<baseline*1.05,'observed_tok_s':adaptive_useful/adaptive_wall,'AR_tok_s':baseline}
                decisions=control.collect('adaptive:'+str(serial)+':'+str(len(cycles)),{'decision':decision_packet});economics=decisions[emitter]['decision'];row['adaptive_economics']=economics
                if economics['switch']and z[-1]not in eos and len(z)<cap:
                    assert rows()[3]==1 and rows()[2]==p;ck(lib.modern_kimi_state_mode_r9114(ctx,0));fallback=True;fallback_at=len(cycles);row['switch_to_pair_AR']=True
            if len(cycles)%8==0:report['phase']=label+':'+str(len(z));save()
            if time.monotonic()>deadline:raise TimeoutError('reserved cleanup deadline reached within case')
        end=time.perf_counter();useful=sum(x not in eos for x in z);decode_useful=sum(c['useful_emitted']for c in cycles);decode_wall=end-first;gaps=[b-a for a,b in zip(timestamps,timestamps[1:])];q={'label':label,'K':K,'strategy':strategy,'draft':active_draft if K else None,'adaptive_fallback_cycle':fallback_at,'AR_calibration_tok_s':baseline,'input_tokens':len(inp),'generated_ids':z,'generated_tokens':len(z),'EOS':z[-1]in eos,'truncated':z[-1]not in eos and len(z)>=cap,'prefill_local_wall_s':first-begin,'decode_user_wall_s':decode_wall,'useful_decode_tokens':decode_useful,'decode_user_tok_s':decode_useful/decode_wall if decode_wall else None,'first_last_user_tok_s':(useful-1)/(timestamps[useful-1]-timestamps[0])if useful>1 else None,'E2E_useful_output_tok_s':useful/(end-begin),'native_target_window_s':native_s,'native_target_only_tok_s':decode_useful/native_s if native_s else None,'cycles':cycles,'mean_useful_outputs_per_cycle':decode_useful/len(cycles)if cycles else None,'proposal_acceptance':sum(c['accepted']for c in cycles)/sum(c['proposal_count']for c in cycles)if any(c['proposal_count']for c in cycles)else None,'ITL_max_s':max(gaps)if gaps else None,'ITL_P50_s':float(np.percentile(gaps,50))if gaps else None,'ITL_P95_s':float(np.percentile(gaps,95))if gaps else None,'ITL_P99_s':float(np.percentile(gaps,99))if gaps else None,'clock_scope':'single emitter token-ready/emit clock; not HTTP; prefill excluded from decode; bursts count real emitted tokens'};stats=(C.c_uint64*56)();ck(lib.modern_kimi_state_stats_r9114(ctx,stats,56));q['state_owner_stats8x7']=list(stats);report['speculative_quality_cases'].append(q);save();return q
    fixed=json.loads((r/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_text());eos=set(fixed['eos_token_ids'])
    try:
        if node==0:
            listener=listen(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));peer,_=listener.accept();hello,body=recv(peer);assert hello=={'cmd':'feature-peer','nonce':nonce}and not body;send(peer,{'ready':True});draft=connect(os.environ['KIMI_DRAFT_HOST_R9046'],int(os.environ['KIMI_DRAFT_PORT_R9046']));send(draft,{'cmd':'hello','nonce':nonce});ready,body=recv(draft);assert ready['ready']and ready['block_size']==8 and not body;report['draft_handshake']=ready;drafts['dflash2']=(draft,0)
        elif node==2:
            peer=connect(os.environ['KIMI_ENGINE_HEAD'],int(os.environ['KIMI_FEATURE_PORT_R9046']));send(peer,{'cmd':'feature-peer','nonce':nonce});ready,body=recv(peer);assert ready['ready']and not body
        control.collect('live-transport-ready',{'node':node});report['transport']='persistentbinaryTCPonib0;targetcollectivesremainIBext;CPUstagingnotGDR';save()
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(g['model']),trust_remote_code=True);samples=json.loads(pathlib.Path(__file__).with_name('quality_samples.json').read_text());report['quality_sample_source']=samples['source'];report['quality_smoke_only']=True
        def execute(command):
            nonlocal active_draft,case_serial
            assert command.get('schema')==1 and command.get('action')=='case'
            label=command['id'];assert re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',label)and label not in command_ids;command_ids.add(label);strategy=command['mode'];assert strategy in ['ar','oldspec','state','adaptive'];K=0 if strategy=='ar'else int(command.get('K',7));assert K in [0,2,4,7];cap=int(command.get('cap',256));assert 2<=cap<=512
            active_draft=command.get('draft','dflash2');assert active_draft in ['dflash2','modal'];live.modal=active_draft=='modal'
            if K and node==0 and active_draft not in drafts:
                assert active_draft=='modal','missing default draft'
                sock=connect(os.environ['KIMI_MODAL_HOST_R9116'],int(os.environ['KIMI_MODAL_PORT_R9116']));send(sock,{'cmd':'hello','nonce':os.environ.get('KIMI_MODAL_NONCE_R9116',nonce)});ready,body=recv(sock);assert ready['ready']and ready['block_size']==16 and not body;drafts['modal']=(sock,0);report['modal_handshake']=ready
            sample=None
            if 'quality' in command:
                spec=command['quality'];assert spec.get('task')=='GSM8K';sample={'question':spec['question'],'answer':str(spec['answer']),'sample_index':int(spec['sample_index'])};assert len(sample['question'])<=8192
                inp=tok.apply_chat_template([{'role':'user','content':sample['question']+'\nSolve this problem. End with #### followed by the numeric answer.'}],tokenize=True,add_generation_prompt=True)
            else:inp=command['input_ids']
            assert isinstance(inp,list)and len(inp)>0 and all(type(x)is int and 0<=x<vocab for x in inp)
            if len(inp)>512 or len(inp)+cap>1024:
                report.setdefault('capacity_skipped_commands',[]).append({'id':label,'input_tokens':len(inp),'cap':cap,'reason':'bounded capability; original question preserved, no truncation'});save();return
            key=hashlib.sha256(json.dumps(inp).encode()).hexdigest()
            if strategy=='adaptive'and key not in baselines:
                calibration=decode(inp,label+'-ARcal',0,min(cap,32),'ar');baselines[key]=calibration['decode_user_tok_s']
            report['active_command']={'id':label,'case_serial':case_serial,'SHA256':hashlib.sha256(json.dumps(command,sort_keys=True).encode()).hexdigest()};case_serial+=1;save()
            q=decode(inp,label,K,cap,strategy,baselines.get(key));q['command']=command
            if strategy=='ar':baselines[key]=q['decode_user_tok_s']
            if sample:
                text=tok.decode(q['generated_ids'],skip_special_tokens=True);matches=re.findall(r'####\s*(-?[0-9][0-9,.]*)',text);answer=matches[-1].replace(',','').rstrip('.')if matches else None;q['quality']={'task':'GSM8K','sample_index':sample['sample_index'],'expected_answer':sample['answer'],'extracted_answer':answer,'correct':answer==sample['answer'],'response':text,'truncated':q['truncated'],'scope':'pilot; aggregate accuracy only after declared sample set complete'}
            save()
        initial=[]
        # Same prompt and cap for all strategies; oldspec is the frozen fullModel replay control.
        for index,sample in enumerate(samples['samples']):
            for strategy in ['ar','oldspec','state','adaptive']:
                initial.append({'schema':1,'action':'case','id':'gsm8k'+str(index)+'-'+strategy,'mode':strategy,'K':7,'cap':256,'quality':{'task':'GSM8K','question':sample['question'],'answer':sample['answer'],'sample_index':index}})
        for index,inp in enumerate(fixed['natural_input_ids'][:2]):
            for strategy in ['ar','oldspec','state','adaptive']:
                initial.append({'schema':1,'action':'case','id':'natural'+str(index)+'-'+strategy,'mode':strategy,'K':7,'cap':64,'input_ids':inp})
        for command in initial:
            allow=control.collect('initial-budget:'+command['id'],{'allow':time.monotonic()+150<deadline if node==0 else None})[0]['allow']
            if not allow:report.setdefault('skipped_budget',[]).append(command['id']);continue
            execute(command)
        queue_name=os.environ.get('KIMI_CASE_QUEUE_R9116');seen=set();poll=0
        if queue_name:
            assert re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',queue_name);queue=r/'results'/queue_name;report['resident_queue']=str(queue);report['phase']='resident_ready';save()
            while True:
                packet=None
                if node==0:
                    if time.monotonic()+150>=deadline:packet={'action':'stop','reason':'reserved cleanup deadline'}
                    else:
                        for path in sorted(queue.glob('q*.json')):
                            if path.name not in seen:
                                raw=path.read_bytes();assert len(raw)<=131072;packet=json.loads(raw);seen.add(path.name);break
                        if packet is None:time.sleep(.5)
                packet=control.collect('resident-poll:'+str(poll),{'command':packet})[0]['command'];poll+=1
                if packet is None:continue
                if packet.get('action')=='stop':report['resident_stop']=packet;break
                execute(packet)
        report.update(serving_protocol_complete=True,trained_spec_speed_claim='MEASURED_IN_REPORT_PENDING_INDEPENDENT_AUDIT',full_sampling_lossless_claim=False,quality_gate='taskanswers_and_later_benchmark; no oldARtokenidentity');save()
    finally:
        live.begin(False)
        if node==0 and draft is not None:
            try:
                for draft_name in list(drafts):
                    active_draft=draft_name;rpc('close')
            finally:
                for sock,seq in drafts.values():sock.close()
        if peer is not None:peer.close()
        if listener is not None:listener.close()
