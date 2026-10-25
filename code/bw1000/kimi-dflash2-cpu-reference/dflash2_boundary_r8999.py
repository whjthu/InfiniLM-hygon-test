"""Default-OFF LightSeek DFlash2 boundary. No downloads, model load or GPU setup.

Reference control/NumPy functions are CPU-testable. They are not a trained MLA
forward or a qualification of current TP16/Math0 target verification.
"""
import copy
import numpy as np
TAPS=(19,37,66,78,90);H=7168;V=163840;W=8;K=7;MASK=163592
STREAM='completed-layer-prefix-before-next-AttnRes-and-final-norm'

def check_config(c):
    d=c['dflash_config']
    assert c['architectures']==['DFlash2DraftModel']
    assert tuple(d['target_layer_ids'])==TAPS and d['block_size']==W and d['attention_mode']=='mla'
    assert c['hidden_size']==H and c['vocab_size']==V and c['draft_vocab_size']==V
    assert c['num_hidden_layers']==5 and c['num_target_layers']==93 and c['fc_norm']is False
    assert d['mask_token_id']==MASK and (d['conv_kernel_size'],d['conv_group_size'],d['selector_rank'],d['selector_top_k'])==(2,16,256,16)
    assert (c['q_lora_rank'],c['kv_lora_rank'],c['qk_nope_head_dim'],c['qk_rope_head_dim'],c['v_head_dim'])==(1536,512,128,64,128)
    assert c['layer_types']==['sliding_attention']*4+['full_attention'] and c['sliding_window']==4096 and c['is_causal']is False
    assert c['rope_parameters']['rope_theta']==50000 and c['rope_parameters']['factor']==32
    return {'feature_width':5*H,'draft_query_rows':8,'proposal_rows':list(range(1,8)),'verify_width':8,'stream':STREAM,'tokenizer':'Reuse target integer IDs, embedding and head; independent tokenizer equivalence NOT established'}

def feature_packet(taps,*,layers=TAPS,stream=STREAM):
    assert tuple(layers)==TAPS and stream==STREAM,'Reject oldRadix/Modal or next-consumer-AttnRes feature semantics'
    assert len(taps)==5 and all(x.dtype==np.uint16 and x.ndim==3 and x.shape[0]==1 and x.shape[2]==H for x in taps)
    assert len({x.shape for x in taps})==1
    return np.concatenate(taps,axis=-1) # BF16 bits; no per-tap norm when fc_norm=False

def rms_f32(x,weight,eps=1e-5):
    x=np.asarray(x,np.float32);return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)+np.float32(eps))*weight

def context_reference(taps,fc,hidden_norm):
    """Source order: concat -> projection -> context RMSNorm, not per-tap norm.

    Float32 CPU equation reference only; BF16 operator rounding is diagnostic.
    """
    return rms_f32(np.concatenate(taps,axis=-1)@fc.T,hidden_norm)

def grouped_convolve_reference(hidden,dynamic,base,group_size):
    b,t,h=hidden.shape;kernel=base.shape[0];assert h%group_size==0
    x=hidden.reshape(b,t,h//group_size,group_size);d=dynamic.reshape(b,t,kernel,h//group_size,1);out=np.zeros_like(x)
    for offset in range(kernel):
        shifted=np.zeros_like(x)
        if offset==0:shifted=x
        elif offset<t:shifted[:,offset:]=x[:,:-offset]
        out=out+base[offset].reshape(1,1,h//group_size,group_size)*shifted+d[:,:,offset]*shifted
    return out.reshape(hidden.shape)

def selector_greedy_reference(hidden,logits,anchor,predecessor,successor,projection,top_k=16):
    """DFlash2 candidate path, distinct from DSpark vanilla Markov/confidence.

    top-k tie ordering is provider-specific: unique-logit CPU fixture only.
    """
    b,t,v=logits.shape;assert 1<=top_k<=v
    candidates=np.argsort(-logits,axis=-1,kind='stable')[...,:top_k];projected=hidden@projection.T;prev=np.asarray(anchor).copy();path=[]
    for j in range(t):
        ids=candidates[:,j];unary=np.take_along_axis(logits[:,j],ids,-1)
        score=unary+np.sum((predecessor[prev]*projected[:,j])[:,None,:]*successor[ids],axis=-1)
        prev=ids[np.arange(b),np.argmax(score,axis=-1)];path.append(prev.copy())
    return np.stack(path,axis=1)

def serving_to_training_state(state,target_embedding):
    """Inverse of producer's explicit serving export; strict load must follow.

    State is supplied by an authorized caller. No AutoModel/from_pretrained or
    network fallback. Embedding/head are target-owned and not in draft export.
    """
    aliases=(('fc.','context_proj.'),('hidden_norm.','context_norm.'),('norm.','final_norm.'))
    result={}
    for key,value in state.items():
        # Current creator export drops this key, but the actual checkpoint
        # header has not been read. Preserve it if present, never silently drop.
        dst=next((new+key[len(old):]for old,new in aliases if key.startswith(old)),key)
        assert dst not in result;result[dst]=value
    result.setdefault('embed_tokens.weight',target_embedding)
    return result

def load_existing_torchspec(model,state,target_embedding,*,enabled=False):
    if not enabled:raise RuntimeError('DFlash2 defaultOFF; no implicit draft load')
    assert type(model).__name__=='DFlash2DraftModel' and type(model).__module__=='torchspec.models.draft.dflash2'
    assert model.config.attention_mode=='mla' and tuple(model.config.target_layer_ids)==TAPS
    assert tuple(target_embedding.shape)==(V,H)
    supplied=serving_to_training_state(state,target_embedding)
    expected=model.state_dict();assert set(expected)==set(supplied),{'missing':sorted(set(expected)-set(supplied)),'extra':sorted(set(supplied)-set(expected))}
    assert all(tuple(expected[k].shape)==tuple(v.shape)for k,v in supplied.items())
    model.load_state_dict(supplied,strict=True,assign=True)
    assert not any(x.is_meta for x in model.parameters()),'Unmaterialized draft weights'
    assert not any(x.is_meta for x in model.buffers()),'Recreate canonical MLA/YaRN nonpersistent buffers before execution'
    return model.eval() # Actual tokenizer/embed/head equivalence remains unqualified.

def verify_plan(anchor,proposals,past,active,shadow,*,kv_row=1):
    assert len(proposals)<=K and past>=0 and active!=shadow and min(active,shadow,kv_row)>0
    tokens=[int(anchor),*map(int,proposals)];assert all(0<=x<V for x in tokens)
    assert past+len(tokens)<=1024,'Existing r3632 four256page CAPI bound; do not silently truncate'
    positions=list(range(past,past+len(tokens)));first_page=1+(kv_row-1)*4
    return {'tokens':tokens,'positions':positions,'past':past,'total':past+len(tokens),'offsets':[0,len(tokens)],'KV_row':kv_row,'mamba_read':active,'mamba_write':shadow,'sample_all_positions':True,'block_table':[first_page+j for j in range(4)],'slot_mapping':[(first_page+x//256)*256+x%256 for x in positions]}

def existing_spec_CAPI_arrays(plan):
    """Eight input pointers after ctx/tokens/count/flags, in actualr3632 order.

    Caller supplies final3 output pointers separately. This creates CPU arrays
    only; it does not resolve/call a DSO or change current target math policy.
    """
    return tuple(np.ascontiguousarray(v,dtype=t)for v,t in [([plan['mamba_read']],np.int32),([plan['mamba_write']],np.int32),([plan['KV_row']],np.int32),([plan['past']],np.int32),([plan['total']],np.int32),(plan['offsets'],np.int32),(plan['tokens'],np.int64),(plan['positions'],np.int64)])

def greedy_commit(anchor,proposals,target_ids,past,active,shadow,*,eos=(),remaining=128):
    assert len(target_ids)==len(proposals)+1 and 1<=remaining
    accepted=0
    while accepted<len(proposals)and proposals[accepted]==target_ids[accepted]and proposals[accepted]not in eos:accepted+=1
    emitted=(list(proposals[:accepted])+[int(target_ids[accepted])])[:remaining]
    for j,token in enumerate(emitted):
        if token in eos:emitted=emitted[:j+1];break
    stopped=emitted[-1]in eos or len(emitted)==remaining
    retained=accepted+1
    return {'accepted_drafts':accepted,'emitted':emitted,'retained_input_tokens':retained,'next_past':past+retained,'replay_inputs':([anchor,*proposals][:retained]if retained<len(target_ids)and not stopped else None),'replay_from':active,'replay_into':shadow,'next_active':shadow,'next_shadow':active,'terminal':stopped,'correction_is_unconsumed_next_anchor':True}

def execute_paired_target_round(forward,anchor,proposals,past,active,shadow,*,eos=(),remaining=128,enabled=False):
    """Inject existing serialized target callback; never invent rollback-by-crop.

    callback receives the concrete metadata above and returns per-input target
    decisions. Its native KDA+convolution state implementation needs separate
    current-DSO/math/topology qualification; this adapter does not supply it.
    """
    if not enabled:raise RuntimeError('DFlash2 defaultOFF; current target verification unqualified')
    plan=verify_plan(anchor,proposals,past,active,shadow);target=forward(plan)
    commit=greedy_commit(anchor,proposals,target,past,active,shadow,eos=eos,remaining=remaining)
    if commit['replay_inputs']is not None:
        replay=verify_plan(commit['replay_inputs'][0],commit['replay_inputs'][1:],past,active,shadow)
        actual=forward(replay);assert actual[-1]==target[commit['retained_input_tokens']-1],'replay correction differs'
    return commit
