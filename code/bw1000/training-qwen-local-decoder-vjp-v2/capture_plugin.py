"""Stop after two native accumulation microbatches and replay two captured layers in place."""
import hashlib
import json
import os
from pathlib import Path

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_update_capture import cpu

from layer_vjp import clone_tree, need, record, replay, save_operands, versions


KEY='qwen_native_resume_witness'
TARGET={1:(1,33),4:(2,29)}


class DecoderCapture(TrainerCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        self.rank=int(os.environ['RANK'])
        need(torch.version.hip and self.rank==int(os.environ['LOCAL_RANK'])==torch.cuda.current_device()
             and int(os.environ['WORLD_SIZE'])==8 and 0<=self.rank<8,'Wrong original eight-rank lease')
        need(state.global_step==20 and args.max_steps==25 and args.gradient_accumulation_steps==4 and args.bf16,
             'Original training recipe changed')
        self.output=Path(args.output_dir)
        self.deadline=float(os.environ['NATIVE_RESUME_DEADLINE'])
        reference=Path(os.environ['QWEN_VJP_REFERENCES'])
        raw=reference.read_bytes()
        need(hashlib.sha256(raw).hexdigest()==os.environ['QWEN_VJP_REFERENCES_SHA256'],'Input reference identity differs')
        self.expected=json.loads(raw)[str(self.rank)]
        self.capture={}
        self.inputs=[]
        self.micro=0
        self.armed=False
        self.substeps=0
        model=kwargs.get('model',self.trainer.model)
        self.model=model
        self.handles=[model.register_forward_pre_hook(self.model_entry,with_kwargs=True)]
        self.layer=None
        if self.rank in TARGET:
            self.target_micro,self.target_layer=TARGET[self.rank]
            language=model.get_base_model().get_submodule('model.language_model')
            need(len(language.layers)==36,'Unexpected Qwen decoder count')
            self.layer=language.layers[self.target_layer]
            need(self.layer.training and self.layer.gradient_checkpointing is True,'Native layer checkpoint mode differs')
            self.parameter_versions=versions(self.layer)
            self.handles += [self.layer.register_forward_pre_hook(self.layer_entry,with_kwargs=True),
                             self.layer.register_forward_hook(self.layer_exit,with_kwargs=True)]
        self.original=self.trainer.training_step
        self.instance_method='training_step' in vars(self.trainer)
        self.hook=self.training_step
        self.trainer.training_step=self.hook
        return control

    def training_step(self,model,*args,**kwargs):
        need(self.trainer.state.global_step==20 and self.micro<2 and self.micro==self.substeps,
             'Native diagnostic crossed the declared accumulation boundary')
        need(isinstance(model,torch.nn.parallel.DistributedDataParallel)
             and not model.require_backward_grad_sync and not self.trainer.accelerator.sync_gradients,
             'Diagnostic microbatch would enter DDP gradient synchronization')
        self.micro+=1
        self.armed=True
        try:
            loss=self.original(model,*args,**kwargs)
        finally:
            self.armed=False
        need(len(self.inputs)==self.micro,'Missing or duplicate native model entry')
        return loss

    def model_entry(self,_module,args,kwargs):
        if not self.armed:
            return
        need(len(self.inputs)==self.micro-1,'Duplicate native forward')
        actual=fingerprint(cpu(dict(args=args,kwargs=kwargs)))['SHA256']
        need(actual==self.expected[self.micro-1],'Native input differs from original resumed branch')
        self.inputs.append(actual)

    def layer_entry(self,_layer,args,kwargs):
        if not self.armed or self.micro!=self.target_micro:
            return
        need(bool(args) and tuple(args[0].shape)==(1,283,4096) and args[0].dtype==torch.bfloat16
             and tuple(args[0].stride())==(1159168,4096,1),'Selected original decoder input shape differs')
        if self.capture:
            if 'recompute_input' in self.capture:
                raise ValueError('Duplicate selected decoder recompute')
            self.capture['recompute_input']=clone_tree(args[0])
            return
        x=args[0]
        self.capture.update(args=clone_tree(args),kwargs=clone_tree(kwargs),cpu_rng=torch.get_rng_state(),
                            gpu_rng=torch.cuda.get_rng_state(x.device),autocast_enabled=torch.is_autocast_enabled('cuda'),
                            autocast_dtype=torch.get_autocast_dtype('cuda'))
        self.handles.append(x.register_hook(self.save_input_gradient))

    def layer_exit(self,_layer,_args,_kwargs,output):
        if not self.armed or self.micro!=self.target_micro or 'output' in self.capture:
            return
        need(isinstance(output,torch.Tensor) and output.requires_grad,'Missing actual selected decoder output')
        self.capture['output']=clone_tree(output)
        self.handles.append(output.register_hook(self.save_output_gradient))

    def save_input_gradient(self,grad):
        need('grad_in' not in self.capture,'Duplicate native decoder input gradient')
        self.capture['grad_in']=clone_tree(grad)

    def save_output_gradient(self,grad):
        need('grad_out' not in self.capture,'Duplicate native decoder output gradient')
        self.capture['grad_out']=clone_tree(grad)

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        raise ValueError('Decoder capture must stop before the optimizer')

    def on_substep_end(self,args,state,control,**kwargs):
        self.substeps+=1
        need(self.substeps==self.micro and self.substeps<=2 and state.global_step==20,'Native substep boundary differs')
        if self.substeps==2:
            control.should_training_stop=True
            control.should_save=False
            control.should_evaluate=False
            control.should_log=False
        return control

    def on_epoch_end(self,args,state,control,**kwargs):
        if self.substeps==2:
            control.should_save=False
            control.should_evaluate=False
            control.should_log=False
        return control

    def on_train_end(self,args,state,control,**kwargs):
        need(self.substeps==self.micro==2 and state.global_step==20 and len(self.inputs)==2,
             'Native diagnostic did not stop at two local gradients and zero updates')
        need(self.trainer.training_step is self.hook,'Native training callback changed')
        if self.instance_method:
            self.trainer.training_step=self.original
        else:
            del self.trainer.training_step
        for h in self.handles:
            h.remove()
        result=dict(rank=self.rank,completed_microbatches=2,global_step=20,optimizer_updates=0,inputs_matched=2)
        if self.layer is not None:
            need(set(self.capture)=={'args','kwargs','cpu_rng','gpu_rng','autocast_enabled','autocast_dtype',
                                    'recompute_input','output','grad_in','grad_out'},'Selected layer evidence incomplete')
            need(versions(self.layer)==self.parameter_versions,'Layer weights changed during native capture')
            result.update(layer=self.target_layer,micro=self.target_micro,
                          tensors={k:record(self.capture[k]) for k in ('output','grad_in','grad_out','recompute_input')},
                          input_tensor=record(self.capture['args'][0]),
                          private_operand=save_operands(self.output,self.rank,self.capture))
            result['replay']=replay(self.layer,self.capture,self.output,self.rank,self.deadline)
        with (self.output/f'vjp-rank-{self.rank:02d}.json').open('x',encoding='utf-8') as f:
            json.dump(result,f,allow_nan=False)
        # Drain the owned worker group after rank-local tests before Swift tears down the communicator.
        torch.distributed.barrier()
        return control


if KEY in callbacks_map:
    raise ValueError('Native Qwen callback already registered')
callbacks_map[KEY]=DecoderCapture
