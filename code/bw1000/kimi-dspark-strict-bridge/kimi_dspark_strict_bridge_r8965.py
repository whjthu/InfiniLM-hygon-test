"""Strict bridge to retained reviewed RadixArk3c5bac source; no AutoModel.

No network/download, no old Modal fallback, no GPU setup. Full tensor reads are
only performed by explicit future caller supplying a state_dict. This module
can validate the real 62-tensor header using a full-size meta model.
"""
import pathlib,hashlib,json,sys,inspect
PIN='3c5bac301d9cf392706189d82ed947feca6c2f0f'
WEIGHT_SHA='ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca'
SOURCE_SHA={'dspark.py':'805928129fe98c809f4cd130ef91c8942ef3cc27392980250d0a967cec106351',
 'specforge/modeling/draft/dflash.py':'1b8fcdf0e1b84eb0faeeabce2e1c8cd5eb469f50892ae31695451cd03f3f6a80'}

def classes(base):
    base=pathlib.Path(base).resolve()
    for n,h in SOURCE_SHA.items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,n
    # Reject a previously imported namesake, don't silently switch source.
    if 'dspark'in sys.modules:assert pathlib.Path(sys.modules['dspark'].__file__).resolve()==base/'dspark.py'
    if str(base)not in sys.path:sys.path.insert(0,str(base))
    from dspark import DSparkConfig,DSparkDraftModel
    return DSparkConfig,DSparkDraftModel

def validate_config(raw):
    from kimi_dspark_adapter_r8965 import check_config
    result=check_config(raw)
    assert raw['num_attention_heads']==64 and raw['num_key_value_heads']==16 and raw['head_dim']==64
    assert raw['intermediate_size']==14336 and raw['vocab_size']==163840
    assert raw['enable_confidence_head']and raw['confidence_head_with_markov']
    assert raw['rope_parameters']['original_max_position_embeddings']==65536
    assert raw['max_position_embeddings']==1048576
    return result

def instantiate_meta(base,raw):
    import torch
    validate_config(raw);Config,Model=classes(base);cfg=Config(**raw);cfg._attn_implementation='eager'
    with torch.device('meta'):model=Model(cfg)
    assert tuple(model.fc.weight.shape)==(7168,35840)
    assert tuple(model.markov_head.markov_w1.weight.shape)==(163840,256)
    assert tuple(model.markov_head.markov_w2.weight.shape)==(163840,256)
    assert tuple(model.confidence_head.proj.weight.shape)==(1,7424)
    assert tuple(model.confidence_head.proj.bias.shape)==(1,)
    return model

def strict_header(model,metadata):
    actual={k:{'shape':list(v.shape),'dtype':'BF16'}for k,v in model.state_dict().items()}
    assert len(actual)==62 and set(actual)==set(metadata),{'missing':sorted(set(actual)-set(metadata)),'extra':sorted(set(metadata)-set(actual))}
    assert all(v['dtype']=='BF16'and v['shape']==actual[k]['shape']for k,v in metadata.items())
    return actual

def load_strict_state(model,state):
    """Use only when futurecaller explicitly reads real pinned weights."""
    import torch
    strict_header(model,{k:{'shape':list(v.shape),'dtype':'BF16'if v.dtype==torch.bfloat16 else str(v.dtype)}for k,v in state.items()})
    model.load_state_dict(state,strict=True,assign=True)
    # Rotary inv_freq is nonpersistent and meta init is not real initialization.
    model.rotary_emb=type(model.rotary_emb)(model.config)
    assert not any(v.is_meta for v in model.parameters())and not any(v.is_meta for v in model.buffers())
    assert model.rotary_emb.inv_freq.dtype==torch.float32
    return model.eval()

def wire_worker(model,*,torch,device,cache_factory,target_embed,target_head):
    from kimi_dspark_five_tap_r8965 import DSparkRPCWorker
    cfg=model.config.to_dict();validate_config(cfg)
    assert not any(p.is_meta for p in model.parameters()),'Actualweightsnotloaded'
    assert tuple(target_embed.shape)==(163840,7168)and tuple(target_head.shape)==(163840,7168)
    return DSparkRPCWorker(config=cfg,torch=torch,device=device,cache_factory=cache_factory,
        embed=target_embed,backbone=model,shared_head=lambda x:torch.nn.functional.linear(x,target_head),
        markov_head=model.markov_head,confidence_head=model.confidence_head)

def cpu_meta_check(root):
    import torch
    root=pathlib.Path(root);source=root/'models/Kimi-K3-DSpark-3c5bac30-r2699';base=root/'candidates/dspark-reference-3c5bac30-r2764'
    stage=json.loads((root/'results/kimi-draft-r2699-stage.json').read_text());raw=json.loads((source/'config.json').read_text())
    assert stage['completed']and stage['revision']==PIN and stage['files']['model.safetensors']['sha256']==WEIGHT_SHA
    assert(source/'model.safetensors').stat().st_size==4498585858
    model=instantiate_meta(base,raw);keys=strict_header(model,stage['tensor_metadata'])
    signature=str(inspect.signature(model.forward));assert all(k in signature for k in ['position_ids','noise_embedding','target_hidden','past_key_values'])
    rejected=[]
    for name in ['markov_head.markov_w1.weight','confidence_head.proj.bias']:
        bad=dict(stage['tensor_metadata']);bad.pop(name)
        try:strict_header(model,bad)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Missingheadaccepted')
    # Existing qualified2764tiny already exercised actual CPU forward and heads.
    prior=json.loads((root/'results/dspark-reference-r2764-tiny.json').read_text());assert prior['completed']and prior['configured_RoPE']=='yarn'
    return dict(CPU_only=True,GPU=False,passed=True,torch_version=torch.__version__,
       no_GPU_initialization=not torch.cuda.is_initialized(),pin=PIN,
       full_model_meta=True,keys=keys,parameter_count=sum(v.numel()for v in model.parameters()),
       forward_signature=signature,missing_heads_rejected=rejected,
       original_verified_header_SHA256=hashlib.sha256((root/'results/kimi-draft-r2699-stage.json').read_bytes()).hexdigest(),
       retained_weights_present=True,full_weight_rehash=False,weight_values_read=False,
       reused_CPU_forward_proof_SHA256=hashlib.sha256((root/'results/dspark-reference-r2764-tiny.json').read_bytes()).hexdigest(),
       draft_query_tokens=7,query_rows_used=list(range(7)),target_verify_width=8,
       checkpoint_numeric_qualification='NOT_QUALIFIED_THIS_RUN_META_AND_HEADER_ONLY',
       RoPE='Exactpinnedconfig10000/YaRN16 usesretained2764actualCPUforwardproof; no duplicatedRoPEcampaign')
