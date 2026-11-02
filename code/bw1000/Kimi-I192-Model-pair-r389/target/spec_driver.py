"""Four engine-level requests; no HTTP/SLO or stable speedup qualification."""
import hashlib
import json
import time
from publisher import Publisher, integer
from batch_policy_r9128 import greedy


def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


class BoundedNative:
    """Check the original deadline/cancel on all leaders before each Native operation."""
    def __init__(self,native,deadline,cancelled,clock=time.monotonic):
        self._native=native;self._deadline=deadline;self._cancelled=cancelled;self._clock=clock;self._serial=0
    def __getattr__(self,name): return getattr(self._native,name)
    def check(self,operation):
        status='ok'
        try:
            if self._clock()>=self._deadline: status='timeout'
            elif self._cancelled(): status='cancelled'
        except BaseException: status='cancel_check_failed'
        rows=self.control.collect('long-driver-gate:%d:%s'%(self._serial,operation),
                                  dict(node=self.node,status=status))
        self._serial+=1
        if len(rows)!=4 or {r['node'] for r in rows}!=set(range(4)):
            raise RuntimeError('Four target leader cancellation records required')
        if any(r['status']!='ok' for r in rows):
            raise RuntimeError('Original deadline/cancellation boundary stopped all leaders')
    def call(self,*args,**kwargs):
        self.check('forward');return self._native.call(*args,**kwargs)
    def call_append(self,*args,**kwargs):
        self.check('append');return self._native.call_append(*args,**kwargs)
    def reset(self,*args,**kwargs):
        self.check('reset');return self._native.reset(*args,**kwargs)
    def seal_prompt(self,*args,**kwargs):
        self.check('seal');return self._native.seal_prompt(*args,**kwargs)
    def commit(self,*args,**kwargs):
        self.check('commit');return self._native.commit(*args,**kwargs)
    def replay(self,*args,**kwargs):
        self.check('replay');return self._native.replay(*args,**kwargs)


def _row(native,label,expected):
    rows=native.control.collect('long-driver-row:'+label,dict(node=native.node,rows=native.rows([1])))
    if len(rows)!=4 or {r['node'] for r in rows}!=set(range(4)) or any(r['rows'] != [expected] for r in rows):
        raise RuntimeError('Actual target row divergence in AR arm')


def generate_ar(native,tokens,emit,eos):
    """Same W512 true-input prefill, W1 greedy decode, capture OFF throughout."""
    native.reset(False)
    first=None
    for past in range(0,len(tokens),512):
        chunk=list(tokens[past:past+512])
        callback=(lambda selected:emit(tuple(selected))) if past+len(chunk)==len(tokens) else None
        if past==0:
            selected,_,_=native.call('prefill',[1],[0],[chunk],False,callback)
        else:
            selected,_,_=native.call_append(1,past,chunk,capture=False,on_selected=callback)
        _row(native,'AR-prompt-%d'%past,[1,2,past+len(chunk),1])
        first=selected[0]
    native.seal_prompt(len(tokens),speculate=False)
    past=len(tokens);token=first;count=1
    while token not in eos and count<128:
        before=native.rows([1])[0]
        decisions=[]
        def chosen(selected):
            d=greedy(token,[],selected,eos,128-count)
            decisions.append(d);emit(d.emitted)
        native.call('verify',[1],[past],[[token]],False,chosen)
        if len(decisions)!=1: raise RuntimeError('AR selected-ID callback count')
        d=decisions[0];native.commit([1],[d])
        if d.needs_replay: raise RuntimeError('W1 AR cannot require partial replay')
        if not d.terminal:
            past+=len(d.commit_inputs)
            _row(native,'AR-decode-%d'%count,[before[1],before[0],past,1])
        token=d.emitted[-1];count+=len(d.emitted)
    return 'eos' if token in eos else 'length'


def cycle_metrics(decision,eos):
    # accepted is the raw match scan before output budget/EOS truncation.
    accepted_prefix=decision.emitted[:min(decision.accepted,len(decision.emitted))]
    return dict(raw_matched_proposals=decision.accepted,
        accepted_visible_proposals=sum(token not in eos for token in accepted_prefix),
        useful_emitted_tokens=sum(token not in eos for token in decision.emitted),
        emitted_tokens=len(decision.emitted),committed_inputs=len(decision.commit_inputs),
        needs_replay=decision.needs_replay,terminal=decision.terminal)


def _validate_fixtures(frozen):
    # Validate all original frozen bytes with the immutable337 validator first.
    from pair_driver import _validate_fixtures as original
    fixtures=original(frozen)
    if len(fixtures['retained8635']['prompt_tokens'])!=8635: raise ValueError('Exact8635 input')
    return fixtures


def run_pair(native,frozen,connect_draft,grade,*,deadline,cancelled,save,enabled=False,clock=time.monotonic):
    """Caller owns exact source binding, target construction, capture peers and cleanup.

    Start the draft once before the first Spec. Target stays resident for all6.
    connect_draft runs on all leaders but returns a socket client only on node0.
    No request is repeated on failure. grade runs on node0 and must be pinned.
    """
    if enabled is not True: raise RuntimeError('Long pair driver default OFF')
    fixtures=_validate_fixtures(frozen)
    from counters import CounterNative
    counted=CounterNative(native,save)
    n=BoundedNative(counted,deadline,cancelled,clock)
    report=dict(completed=False,requests=[],target_cards=32,draft_cards=1,max_requests=6,
        requests_are_engine_level=True,HTTP=False,client_SLO_qualified=False,
        stable_performance_gain_qualified=False,load_and_draft_init_s_are_separate=True)
    counted.report=report
    publisher=None
    try:
        for ordinal,(name,mode) in enumerate([['retained8635','spec']]*6):
            n.check('request-start')
            fixture=fixtures[name]
            counted.begin_request(ordinal)
            identity=dict(ordinal=ordinal,name=name,mode=mode,prompt_SHA256=fixture['prompt_ID_SHA256'],GROUPED=counted.mode,warm=ordinal<2)
            matches=n.control.collect('long-driver-request-%d'%ordinal,identity)
            if len(matches)!=4 or any(m!=identity for m in matches): raise ValueError('Request identity mismatch')
            if mode=='spec' and publisher is None:
                tick=clock();rpc=connect_draft()
                n.check('draft-connected')
                publisher=Publisher(n,rpc,enabled=True)
                report['draft_connect_load_handshake_local_s']=clock()-tick
            ids=[];visible=[];events=[];start=clock();call_start=len(n.calls)
            row=dict(identity,input_tokens=len(fixture['prompt_tokens']),output_budget=128,
                     all_service_cards=33 if mode=='spec' else 32,completed=False)
            report['requests'].append(row);save(report)
            def emit(values):
                if not values: raise ValueError('Empty emitter burst')
                tick=clock()
                for token in values:
                    integer(token,0,n.vocab-1)
                    if len(ids)>=128 or ids and ids[-1] in fixture['eos']:
                        raise ValueError('Emission beyond budget or after EOS')
                    ids.append(token)
                    if token not in fixture['eos']: visible.append((token,tick-start))
                events.append(dict(at_s=tick-start,tokens=list(values)))
            cycles=[]
            if mode=='AR':
                finish=generate_ar(n,fixture['prompt_tokens'],emit,set(fixture['eos']))
            else:
                before_costs=len(publisher.costs)
                anchor=publisher.prefill(fixture['prompt_tokens'],ordinal+1,emit)
                while anchor not in fixture['eos'] and len(ids)<128:
                    tick=clock();d=publisher.step(anchor,128-len(ids),set(fixture['eos']),emit)
                    cycles.append(dict(cycle_metrics(d,set(fixture['eos'])),cycle_local_wall_s=clock()-tick))
                    if d.terminal: break
                    anchor=d.pending_anchor
                if publisher.phase!='retired': publisher.retire()
                finish='eos' if ids[-1] in fixture['eos'] else 'length'
                row['publisher_costs']=publisher.costs[before_costs:]
            generation_done=clock()
            signature=dict(ids_SHA256=ids_sha(ids),tokens=len(ids),finish=finish)
            signatures=n.control.collect('long-driver-output-%d'%ordinal,signature)
            if len(signatures)!=4 or any(s!=signature for s in signatures): raise ValueError('Four target outputs differ')
            n.check('quality')
            quality=None
            if n.node==0:
                try: quality=dict(passed=True,details=grade(fixture,ids,finish))
                except BaseException as error: quality=dict(passed=False,error_type=type(error).__name__)
            grades=n.control.collect('long-driver-quality-%d'%ordinal,dict(node=n.node,quality=quality))
            quality=next(g['quality'] for g in grades if g['node']==0)
            row.update(signature,generated_ids=ids,emitter_events=events,emitter_scope='node0 backend selected IDs before capture/commit; no HTTP client',
                TTFT_s=visible[0][1] if visible else None,
                backend_decode_tok_s=(len(visible)-1)/(visible[-1][1]-visible[0][1]) if len(visible)>1 and visible[-1][1]>visible[0][1] else None,
                visible_tokens=len(visible),generation_local_wall_s=generation_done-start,
                quality=quality,cycles=cycles,Native_calls=n.calls[call_start:])
            # Reset/drain is part of the per-arm completion record, never a reload.
            n.reset(False)
            reset=n.control.collect('long-driver-drained-%d'%ordinal,dict(node=n.node,reset=True))
            if len(reset)!=4 or {v['node'] for v in reset}!=set(range(4)) or not all(v['reset'] for v in reset): raise RuntimeError('Four leader reset acknowledgement required')
            row.update(completed=True,normal_reset=True,request_through_reset_local_wall_s=clock()-start,I192=counted.end_request())
            save(report)
            if not quality or quality['passed'] is not True: raise RuntimeError('Task answer/finish failed; stop remaining requests')
        publisher.close()
        report['completed']=True;report['normal_draft_protocol_close']=True;save(report)
        return report
    except BaseException as error:
        report['error_type']=type(error).__name__;save(report)
        if publisher is not None and publisher.rpc is not None: publisher.rpc.abort()
        raise  # original controller must destroy32owners and close the sole draft, no retry
