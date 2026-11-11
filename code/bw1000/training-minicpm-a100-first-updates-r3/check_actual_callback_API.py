"""Use real installed Trainer/PEFT classes with a small CPU fixture, no checkpoint."""
import json,os,types
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
import torch
from transformers import Seq2SeqTrainingArguments
from peft import LoraConfig,get_peft_model
from swift.callbacks import TrainerCallback
from recipe_model_contract import validate_native_recipe
class Language(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q_proj=torch.nn.Linear(4,4,bias=False)
        self.config=types.SimpleNamespace(_attn_implementation='sdpa')
    def forward(self,x):return self.q_proj(x)
class Small(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.language_model=Language()
        self.model.vision_tower=torch.nn.Linear(4,4);self.model.merger=torch.nn.Linear(4,4)
        self.is_gradient_checkpointing=False
    def forward(self,x):return self.model.language_model(x)
args=Seq2SeqTrainingArguments(output_dir='/tmp/minicpm-callback-CPU-fixture-unused',max_steps=25,gradient_accumulation_steps=4,per_device_train_batch_size=1,warmup_steps=5,bf16=True,use_cpu=True,report_to='none')
assert not hasattr(args,'lora_rank')
callback=TrainerCallback(args,types.SimpleNamespace());assert callback.args is args
model=get_peft_model(Small(),LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj']))
named,frozen=validate_native_recipe(model,callback.args,128)
assert len(named)==2 and set(frozen)=={'model.vision_tower','model.merger'}
bad=0
def reject():
    global bad
    try:validate_native_recipe(model,args,128)
    except AssertionError:bad+=1
    else:raise AssertionError('bad recipe accepted')
cfg=model.peft_config[model.active_adapter]
for key,value in [('r',4),('lora_alpha',4),('lora_dropout',0)]:
    prior=getattr(cfg,key);setattr(cfg,key,value);reject();setattr(cfg,key,prior)
for key,value in [('max_steps',24),('gradient_accumulation_steps',1),('warmup_steps',0),('bf16',False)]:
    prior=getattr(args,key);setattr(args,key,value);reject();setattr(args,key,prior)
core=model.get_base_model()
for module in (core.model.vision_tower,core.model.merger):
    parameter=next(module.parameters());parameter.requires_grad_(True);reject();parameter.requires_grad_(False)
prior=core.model.language_model.config._attn_implementation;core.model.language_model.config._attn_implementation='eager';reject();core.model.language_model.config._attn_implementation=prior
assert not torch.cuda.is_initialized() and all(p.device.type=='cpu' for p in model.parameters())
print(json.dumps({'state':'REAL_TRAINER_ARGUMENT_AND_PEFT_CPU_CONTRACT_PASS','normal_cases':1,'rejected_bad_cases':bad,'actual_Trainer_arguments_type':type(args).__name__,'actual_PEFT_config_type':type(cfg).__name__,'checkpoint_loaded':False,'GPU_initialized':False,'training_forward_backward':False}))
