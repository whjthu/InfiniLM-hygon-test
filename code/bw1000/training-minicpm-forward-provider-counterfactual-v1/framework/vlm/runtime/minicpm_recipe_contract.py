"""Validate Trainer arguments and real PEFT model state through their own APIs."""
def validate_native_recipe(model,args,expected_numel):
    assert args.max_steps==25 and args.gradient_accumulation_steps==4
    assert args.per_device_train_batch_size==1 and args.warmup_steps==5 and args.bf16
    active=model.active_adapter
    assert isinstance(active,str) and active in model.peft_config
    config=model.peft_config[active]
    assert config.r==16 and config.lora_alpha==32 and config.lora_dropout==0.05
    named=sorted((n,p) for n,p in model.named_parameters() if p.requires_grad)
    assert named and all('.lora_A.' in n or '.lora_B.' in n for n,_ in named)
    assert sum(p.numel() for _,p in named)==expected_numel
    core=model.get_base_model()
    assert not core.is_gradient_checkpointing
    assert core.model.language_model.config._attn_implementation=='sdpa'
    frozen={}
    for path in ('model.vision_tower','model.merger'):
        params=list(core.get_submodule(path).parameters())
        assert params and not any(p.requires_grad for p in params),path
        frozen[path]={'tensors':len(params),'parameters':sum(p.numel() for p in params)}
    return named,frozen
