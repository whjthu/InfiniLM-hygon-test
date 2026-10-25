"""Opt-in current native trace -> ordered PP feature wire -> DSpark worker slice.

No target/draft load. No old six-tap feature input. Existing native trace_hidden
and StageControl/draft_rpc_transport supply capture and transport; no new CAPI.
Backbone/checkpoint construction is deliberately injected and still unqualified.
"""
import base64,contextlib,hashlib,json,os,pathlib,re
from kimi_dspark_adapter_r8965 import (TARGET_LAYERS,CONSUMER_LAYERS,HIDDEN_SIZE,
    STREAM,VERIFY_WIDTH,DRAFT_QUERY_TOKENS,check_config,check_feature_packet,markov_greedy_block,confidence_features)

MAX_FEATURE_TOKENS=512
TRACE_KEYS=('INFINILM_KIMI_TRACE_DIR','INFINILM_KIMI_TRACE_LAYERS',
 'INFINILM_KIMI_TRACE_BOUNDARY_POINTS_R8506','INFINILM_KIMI_TRACE_BOUNDARY_LAYERS_R8506',
 'INFINILM_KIMI_TRACE_HASH_ONLY_R8506','INFINILM_KIMI_TRACE_FULL_FIRST',
 'INFINILM_KIMI_TRACE_FULL_LAST','INFINILM_KIMI_TRACE_PACKED_REQUEST_R8508')

@contextlib.contextmanager
def target_capture(directory, packed_request=0):
    """Use around ONE serialized existing native forward; never mutate live669.

    Caller supplies a new per-node /local directory per immutable job. All owner
    calls finish before restoring flags and collecting their CPU-owned files.
    current trace_hidden writes full BF16 for TP-local0/6; only0 is exported.
    """
    assert isinstance(packed_request,int) and packed_request>=0
    directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=False)
    old={k:os.environ.get(k) for k in TRACE_KEYS}
    values=dict(zip(TRACE_KEYS,(str(directory),','.join(map(str,CONSUMER_LAYERS)),
       'attn_res',','.join(map(str,CONSUMER_LAYERS)),'0','0','92',str(packed_request))))
    os.environ.update(values)
    try:yield directory
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

def local_features(directory, *, node, stage, ranges, tp_nodes, tokens, jobseq):
    """Read actual current trace naming/journal, including PP-node vs PP-stage.

    Only the first node of each TP group publishes rank0: other TP shards are
    diagnostic copies, not independent taps. Metadata is required before bytes.
    """
    assert node//tp_nodes==stage and 0<tokens<=MAX_FEATURE_TOKENS
    if node%tp_nodes:return []
    directory=pathlib.Path(directory);records=[]
    journal=directory/f'stage{node}-rank0.jsonl'
    wanted=[x for x in CONSUMER_LAYERS if ranges[stage][0]<=x<ranges[stage][1]]
    lines=[json.loads(v)for v in journal.read_text().splitlines()]if wanted else []
    for consumer in wanted:
        candidates=[v for v in lines if v['layer']==consumer and v['point']=='attn_res']
        assert len(candidates)==1,(consumer,'missing_or_duplicate_capture')
        q=candidates[0];assert q['rank']==0 and q['stage']==node and q['hidden']==HIDDEN_SIZE
        assert q['tokens']==tokens and q['packed_target_only'] is True
        assert q['selected_shape']==[1,tokens,HIDDEN_SIZE]
        assert len(q['requests'])==1 and q['requests'][0]['nonfinite']==0
        p=directory/f'stage{node}-rank0-layer{consumer}-attn_res-full-bf16.bin'
        raw=p.read_bytes();assert len(raw)==tokens*HIDDEN_SIZE*2
        records.append(dict(jobseq=jobseq,stage=stage,node=node,rank=0,
             target_layer=consumer-1,consumer_layer=consumer,tokens=tokens,
             stream_semantics=STREAM,raw_SHA256=hashlib.sha256(raw).hexdigest(),
             raw_b64=base64.b64encode(raw).decode()))
    return records

def gather_features(control, local, *, jobseq, tokens):
    """Existing ordered all-frontend control collect, same jobseq on every PP.

    Bound: <=5*512*7168*2=36,700,160 raw bytes, base64 ~48.9MB.
    This is a diagnostic transport cost, not zero-cost GPU PP or serving speed.
    """
    packets=control.collect('dspark-five-taps:'+str(jobseq),local)
    rows=sorted([v for group in packets for v in group],key=lambda x:x['consumer_layer'])
    assert [v['consumer_layer']for v in rows]==list(CONSUMER_LAYERS)
    assert [v['target_layer']for v in rows]==list(TARGET_LAYERS)
    assert all(v['jobseq']==jobseq and v['tokens']==tokens and v['stream_semantics']==STREAM for v in rows)
    for v in rows:
        raw=base64.b64decode(v['raw_b64'],validate=True)
        assert len(raw)==tokens*HIDDEN_SIZE*2 and hashlib.sha256(raw).hexdigest()==v['raw_SHA256']
    return rows

def target_forward_with_features(forward, *, directory, control, node, stage,
                                 ranges, tp_nodes, tokens, jobseq,
                                 checked_math_policy, checked_native_sha256):
    """Concrete insertion around existing owner-serialized target CAPI callback.

    Current CPU223/2c native only is source-bound here. Do not silently use the
    old six-tap Math11 target or claim that sealed669/f6f was altered/tested.
    Caller obtains these checked values from its actual loaded-library report
    and the pre-forward math-scope check, not from a draft request.
    """
    assert checked_math_policy==0
    assert checked_native_sha256=='2c2961d29d56788c4e59061b241dc108bc0e9bff3153ed8d7aa6f3f575b83444'
    with target_capture(directory,0):
        result=forward() # Existing CAPI invokes owner stream and returns stage-local.
    local=local_features(directory,node=node,stage=stage,ranges=ranges,
                         tp_nodes=tp_nodes,tokens=tokens,jobseq=jobseq)
    return result,gather_features(control,local,jobseq=jobseq,tokens=tokens)

def committed_packet(features, *, captured_tokens, retain_tokens, prefix_start,
                     committed_target_past, anchor, case, reset):
    """Crop verification features to actual accepted prefix BEFORE RPC.

    prefix_start is the draft's previously committed prefix, not target's
    temporary speculative cache length. Neither rejected future IDs nor gold
    enter this function. Anchor is the currently committed target decision.
    """
    assert 0<=retain_tokens<=captured_tokens<=MAX_FEATURE_TOKENS
    assert 0<=prefix_start and prefix_start+retain_tokens<=committed_target_past
    assert isinstance(anchor,int)and anchor>=0 and type(reset)is bool
    if captured_tokens:
        assert [v['consumer_layer']for v in features]==list(CONSUMER_LAYERS)
    else:assert not features
    parts=[]
    for v in features:
        raw=base64.b64decode(v['raw_b64'],validate=True)
        assert len(raw)==captured_tokens*HIDDEN_SIZE*2 and hashlib.sha256(raw).hexdigest()==v['raw_SHA256']
        parts.append(raw[:retain_tokens*HIDDEN_SIZE*2])
    body=b''.join(parts)if retain_tokens else b''
    header=dict(draft_kind='kimi-k3-dspark',target_layers=list(TARGET_LAYERS),
       stream_semantics=STREAM,W=VERIFY_WIDTH,draft_query_tokens=DRAFT_QUERY_TOKENS,feature_tokens=retain_tokens,
       prefix_start=prefix_start,committed_target_past=committed_target_past,
       anchor=anchor,case=case,reset=reset)
    check_feature_packet(header,body)
    return header,body

class DraftPrefix:
    """Actual worker replacement for six-tap/body unpack and prefix ledger.

    Torch is imported only on execution. Trained DSpark module, target shared
    head, official rope/Markov/confidence weights must be loaded separately.
    """
    def __init__(self):self.case=None;self.past=0
    def unpack(self,req,body,*,torch,device):
        count,_=check_feature_packet(req,body)
        if req['reset']:self.case=req['case'];self.past=0
        assert self.case==req['case'] and req['prefix_start']==self.past
        assert self.past+count<=req['committed_target_past']
        import numpy as np
        chunk=count*HIDDEN_SIZE*2
        taps=[torch.from_numpy(np.frombuffer(body[k*chunk:(k+1)*chunk],np.uint16).copy()).view(torch.bfloat16).reshape(1,count,HIDDEN_SIZE) for k in range(5)]
        return torch.cat(taps,dim=-1).to(device) # [1,count,35840], not old43008
    def finish(self,req,cache):
        """Noise cache is cropped to ONLY committed features, not proposals."""
        self.past+=req['feature_tokens'];assert cache.get_seq_length()==self.past+DRAFT_QUERY_TOKENS
        cache.crop(self.past);assert cache.get_seq_length()==self.past

def worker_forward(req,body,*,state,torch,device,cache,embed,backbone,shared_head,
                   markov_head,confidence_head):
    """Wire-ready worker call, after strict new DSpark checkpoint construction.

    backbone must implement original noise+feature -> final hidden and its
    trained rope/cache semantics. This does NOT instantiate old DFlashModel.
    """
    features=state.unpack(req,body,torch=torch,device=device)
    assert cache.get_seq_length()==state.past
    token_ids=torch.full((1,DRAFT_QUERY_TOKENS),int(backbone.mask_token_id),dtype=torch.long,device=device)
    token_ids[0,0]=req['anchor'];noise=torch.nn.functional.embedding(token_ids,embed)
    positions=torch.arange(state.past,state.past+req['feature_tokens']+DRAFT_QUERY_TOKENS,device=device).view(1,-1)
    hidden=backbone(position_ids=positions,noise_embedding=noise,target_hidden=features,past_key_values=cache,use_cache=True)
    assert hidden.shape[1]==DRAFT_QUERY_TOKENS
    base_logits=shared_head(hidden)
    assert torch.isfinite(hidden).all() and torch.isfinite(base_logits).all()
    anchor=torch.tensor([req['anchor']],device=device,dtype=torch.long)
    tokens,previous=markov_greedy_block(base_logits,anchor,markov_head)
    conf_features=confidence_features(hidden,previous,markov_head,torch.stack,lambda xs,dim:torch.cat(xs,dim))
    # Do not guess confidence activation/threshold semantics or use it to alter
    # target acceptance; return raw trained head output until official binding.
    confidence=confidence_head(conf_features)
    assert torch.isfinite(confidence).all()
    state.finish(req,cache)
    return dict(tokens=torch.stack(tokens,1).cpu().tolist()[0],
                confidence_raw=confidence.detach().cpu().tolist(),cached_prefix=state.past)

class DSparkRPCWorker:
    """Drop-in TCP request body branch after NEW strict checkpoint load.

    Replace old worker lines `chunk=count*7168*2 ... for tap in range(6)`
    through `cache.crop(past)` with this processor. Existing accept/armed/ack
    framing stays unchanged. No checkpoint URL/fallback to old model is hidden.
    """
    def __init__(self,*,config,torch,device,cache_factory,embed,backbone,
                 shared_head,markov_head,confidence_head):
        self.descriptor=check_config(config)
        self.state=DraftPrefix();self.cache=None;self.cache_factory=cache_factory
        self.kw=dict(torch=torch,device=device,embed=embed,backbone=backbone,
            shared_head=shared_head,markov_head=markov_head,confidence_head=confidence_head)
    def process(self,req,body):
        check_feature_packet(req,body) # Reject legacy before modifying cache.
        if req['reset']:self.cache=self.cache_factory()
        assert self.cache is not None
        with self.kw['torch'].inference_mode():
            return worker_forward(req,body,state=self.state,cache=self.cache,**self.kw)
    def serve_one(self,client,index):
        from draft_rpc_transport_r4846 import send,receive
        cmd,empty=receive(client);assert not empty and cmd=={'index':index,'mode':'tcp'}
        send(client,{'armed':index});req,body=receive(client)
        reply=self.process(req,body);send(client,dict(index=index,completed=True,**reply))
        ack,empty=receive(client);assert not empty and ack=={'received':index}

def rpc_request(sock,req,body,index):
    """Replacement for old request_draft's six fixed body parts."""
    from draft_rpc_transport_r4846 import send,receive
    check_feature_packet(req,body);send(sock,{'index':index,'mode':'tcp'})
    assert receive(sock)[0]=={'armed':index};send(sock,req,body)
    reply,empty=receive(sock);assert not empty and reply['index']==index and reply['completed']
    assert len(reply['tokens'])==7 and reply['cached_prefix']==req['prefix_start']+req['feature_tokens']
    send(sock,{'received':index});return reply
