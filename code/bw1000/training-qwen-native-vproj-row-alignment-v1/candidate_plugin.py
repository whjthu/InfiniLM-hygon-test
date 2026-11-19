"""Opt-in replay of the first real training forward without returning to Trainer."""
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from swift.callbacks import TrainerCallback, callbacks_map
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from replay_state import ForwardState, describe, digest, rng, restore_rng
from detail_trace import DetailTrace
from text_base_preference import ARMS, TextBasePreference
from vproj_replay import SITE, replay as profile_vproj
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.checkpointing_lifecycle import validate_checkpointing_lifecycle

KEY='qwen_native_initial_forward_repeat'
RANK_BOUNDARIES={0,1,4,5}
VISION_ORDER=('cublaslt',)
DECODER_FOCUS = (6, 10)


def need(ok,reason):
    if not ok:raise ValueError(reason)


def checked_archive(folder,name,record):
    p=folder/name
    a=p.lstat()
    need(stat.S_ISREG(a.st_mode) and record['bytes']==a.st_size, 'ORIGINAL_TENSOR_FILE_SIZE')
    sha=hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda:f.read(1<<20),b''):sha.update(v)
    b=p.lstat()
    need((a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
         and sha.hexdigest()==record['SHA256'], 'ORIGINAL_TENSOR_FILE_DIGEST')
    return torch.load(p,map_location='cpu',weights_only=True,mmap=True)


class LayerBoundaries:
    def __init__(self,core):
        text=core.get_submodule('model.language_model')
        need(len(text.layers)==36,'ORIGINAL_TEXT_LAYER_COUNT')
        self.points={}
        self.handles=[]
        def before(name):
            def mark(_module,args,kw):
                need(name not in self.points,'DUPLICATE_LAYER_INPUT')
                self.points[name]=describe(args[0] if args else kw['hidden_states'])
            return mark
        def after(name):
            def mark(_module,_args,out):
                need(name not in self.points,'DUPLICATE_LAYER_OUTPUT')
                self.points[name]=describe(out[0] if isinstance(out,(tuple,list)) else out)
            return mark
        for i,m in enumerate(text.layers):
            name=f'decoder_{i:02d}'
            self.handles.append(m.register_forward_pre_hook(before(name+'.input'),with_kwargs=True))
            self.handles.append(m.register_forward_hook(after(name+'.output')))
        self.handles.extend([text.norm.register_forward_hook(after('final_norm')),
                             core.get_output_embeddings().register_forward_hook(after('lm_head'))])

    def finish(self):
        need(len(self.points)==74,'INCOMPLETE_LAYER_TRACE')
        return self.points

    def close(self):
        for h in self.handles:h.remove()


class VisionFC2Preference:
    """Set one preference inside the synchronous FC2 dispatch and restore it on exit."""

    def __init__(self, vision):
        self.arm = None
        self.calls = []
        self.originals = []
        self.set_or_get = torch.backends.cuda.preferred_blas_library
        self.expected = torch.backends.cuda._BlasBackends
        self.inherited = self.set_or_get()
        need(len(vision.blocks) == 27, 'FULL_VISION_FC2_COUNT')
        for layer in vision.blocks:
            fc2 = layer.mlp.linear_fc2
            need(type(fc2) is torch.nn.Linear and fc2.bias is not None
                 and list(fc2.weight.shape) == [1152, 4304]
                 and fc2.weight.dtype == torch.bfloat16, 'ACTUAL_VISION_FC2_TYPE')
        for index, layer in enumerate(vision.blocks):
            fc2 = layer.mlp.linear_fc2
            original = fc2.forward
            self.originals.append((fc2, original))
            fc2.forward = self.wrap(index, original)

    def wrap(self, index, original):
        def call(operand):
            need(self.arm in VISION_ORDER and index == len(self.calls) % 27, 'FC2_NATIVE_CALL_ORDER')
            need(operand.ndim == 2 and operand.shape[0] > 0 and operand.shape[1] == 4304
                 and operand.dtype == torch.bfloat16 and operand.device.type == 'cuda',
                 'FC2_NATIVE_BF16_OPERAND')
            previous = self.set_or_get()
            need(previous == self.inherited, 'FC2_EXTERNAL_PREFERENCE_CHANGE')
            try:
                got = self.set_or_get(self.arm)
                need(got == self.expected[self.arm] == self.set_or_get(), 'FC2_PREFERENCE_READBACK')
                out = original(operand)
            finally:
                self.set_or_get(previous)
            need(self.set_or_get() == previous, 'FC2_PREFERENCE_RESTORE')
            need(out.dtype == operand.dtype and tuple(out.shape) == (operand.shape[0],1152),
                 'FC2_NATIVE_BF16_OUTPUT')
            self.calls.append(dict(layer=index, rows=operand.shape[0], stride=list(operand.stride()),
                                   dtype=str(operand.dtype), actual_preference=str(got)))
            return out
        return call

    def finish(self):
        need(27 <= len(self.calls) <= 108 and len(self.calls) % 27 == 0, 'FC2_FULL_VISION_CALLS')
        result = list(self.calls)
        self.calls.clear()
        return result

    def close(self):
        for module, original in self.originals:
            module.forward = original


class NativeInitialForwardRepeat(TrainerCallback):
    def __init__(self,swift_args,trainer):
        super().__init__(swift_args,trainer)
        self.trainer=trainer

    def on_train_begin(self,args,state,control,**kw):
        self.epoch=float(os.environ['FIRST_UPDATE_EPOCH'])
        self.deadline=float(os.environ['FIRST_UPDATE_DEADLINE'])
        need(self.epoch<=time.time()<self.deadline,'NATIVE_REPLAY_DEADLINE')
        rank=int(os.environ['RANK'])
        local=int(os.environ['LOCAL_RANK'])
        need(0<=rank<8 and rank==local==torch.cuda.current_device() and
             torch.cuda.device_count()==8 and torch.version.hip and
             torch.distributed.is_initialized() and torch.distributed.get_world_size()==8,
             'ORIGINAL_DISTRIBUTED_TOPOLOGY')
        need(state.global_step==0 and args.max_steps==25,'ORIGINAL_REQUEST_ENTRY')
        path=Path(os.environ['FIRST_UPDATE_BINDING'])
        raw=path.read_bytes()
        need(hashlib.sha256(raw).hexdigest()==os.environ['FIRST_UPDATE_BINDING_SHA256'],'TRAIN_RECIPE_BINDING')
        binding=json.loads(raw)
        model=kw.get('model',self.trainer.model)
        prepare=getattr(self.trainer,'_prepare_gradient_checkpointing',None)
        named,_frozen=validate_native_recipe(model,args,binding,checkpointing_method=prepare)
        core=model.get_base_model()
        validate_checkpointing_lifecycle(core,args,binding,prepare)
        callbacks=self.trainer.callback_handler.callbacks
        recorders=[cb for cb in callbacks if type(cb).__name__=='NativeFirstUpdates']
        need(len(recorders)==1 and callbacks.index(recorders[0])<callbacks.index(self),'ORIGINAL_RECORDER_NOT_FIRST')
        capture=recorders[0].capture
        need(capture is not None and not capture.detached and len(capture.handles)==2
             and capture.inputs==capture.outputs==capture.updates==0,'ORIGINAL_RECORDER_NOT_ATTACHED')
        capture.detach()
        need(capture.detached and not capture.handles
             and self.trainer._clip_grad_norm==capture.original_clip,'ORIGINAL_RECORDER_DETACH_FAILED')
        self.named=list(named)
        self.call_count=0
        self.math_routes=0
        self.detail=None
        self.vision_calls=[]
        vision=core.get_submodule('model.visual')
        layers=[m for m in vision.modules() if isinstance(m,qwen.Qwen3VLVisionAttention)]
        need(bool(layers) and all(m.config._attn_implementation=='sdpa' for m in layers),
             'ACTUAL_VISION_SDPA_PROVIDER_REQUIRED')
        self.vision_layers=len(layers)
        blas_source=Path(torch.backends.cuda.__file__).read_bytes()
        need(hashlib.sha256(blas_source).hexdigest()=='94f8fde8aa08235c9d06c0508dc68d94a004bf336a24130241df35bd28b68d6a',
             'EXACT_BLAS_PYTHON_PROVIDER')
        self.fc2_preferences=VisionFC2Preference(vision)
        self.text_base=TextBasePreference(core, rank)
        registry=qwen.ALL_ATTENTION_FUNCTIONS
        source=registry['sdpa']
        need(source is sdpa_attention_forward,'FIXED_SDPA_PROVIDER')
        torch.use_deterministic_algorithms(True,warn_only=False)

        @functools.wraps(source)
        def route(module,query,key,value,attention_mask,*a,**k):
            if isinstance(module,qwen.Qwen3VLVisionAttention):
                return self.vision_route(source,module,query,key,value,attention_mask,*a,**k)
            if not isinstance(module,qwen.Qwen3VLTextAttention):
                return source(module,query,key,value,attention_mask,*a,**k)
            need(query.dtype==torch.bfloat16 and query.device.type=='cuda','ORIGINAL_TEXT_QUERY')
            self.math_routes+=1
            with sdpa_kernel(SDPBackend.MATH):
                need(torch.backends.cuda.math_sdp_enabled() and not (
                    torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled()
                    or torch.backends.cuda.cudnn_sdp_enabled()),'EXCLUSIVE_MATH_SDPA')
                if self.detail:self.detail.observe_sdpa_input(module,query,key,value,attention_mask)
                result=source(module,query,key,value,attention_mask,*a,**k)
                if self.detail:self.detail.observe_sdpa_output(module,result)
                return result

        registry.register('sdpa',route)
        need(registry['sdpa'] is route,'ACTUAL_SDPA_DISPATCH')
        original=model.forward

        @functools.wraps(original)
        def forward(*values,**options):
            self.call_count+=1
            need(self.call_count==1 and state.global_step==0 and model.training and
                 torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),'FIRST_NATIVE_TRAIN_FORWARD')
            evidence=self.replay(rank,model,core,original,values,options)
            out=Path(os.environ['QWEN_REPEAT_OUTPUT'])/f'repeat-rank-{rank:02d}.json'
            with out.open('x',encoding='utf8') as f:
                json.dump(evidence,f,allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            # All eight children exit successfully from the first model call, prior to Trainer backward.
            torch.distributed.barrier(device_ids=[local])
            torch.distributed.destroy_process_group()
            raise SystemExit(0)

        model.forward=forward
        return control

    def vision_route(self,source,module,query,key,value,attention_mask,*args,**kwargs):
        need(module.config._attn_implementation=='sdpa' and query.device.type=='cuda'
             and query.ndim==key.ndim==value.ndim==4 and attention_mask is None
             and kwargs.get('is_causal') is False,'ACTUAL_VISION_SDPA_CALL_BOUNDARY')
        def execute():
            flags=dict(math=torch.backends.cuda.math_sdp_enabled(),
                       flash=torch.backends.cuda.flash_sdp_enabled(),
                       efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                       cudnn=torch.backends.cuda.cudnn_sdp_enabled())
            need(len(self.vision_calls)<512,'VISION_SDPA_CALL_BUDGET')
            self.vision_calls.append(dict(q=list(query.shape),k=list(key.shape),v=list(value.shape),
                 dtype=str(query.dtype),dropout=kwargs.get('dropout',0.0),allowed=flags))
            return source(module,query,key,value,attention_mask,*args,**kwargs)
        return execute()

    def replay(self,rank,model,core,original,values,options):
        check=lambda:need(self.epoch<=time.time()<self.deadline,'REPLAY_FORWARD_DEADLINE')
        folder=Path('/replay-capture')/f'rank-{rank:02d}'
        manifest=json.loads(Path(os.environ['QWEN_REPEAT_PINS']).read_bytes())
        record=manifest['ranks'][str(rank)]['files']
        before=checked_archive(folder,'initial.pt',record['initial.pt'])
        try:
            need(before['global_step']==0 and set(before['parameters'])=={n for n,_ in self.named}
                 and len(self.named)==504,'ORIGINAL_TRAINABLE_NAMES')
            for n,p in self.named:
                host=before['parameters'][n]
                actual=p.detach().cpu()
                need(actual.dtype==host.dtype and actual.shape==host.shape and torch.equal(actual,host),
                     'ORIGINAL_INITIAL_TRAINABLE_VALUE')
        finally:
            del before
        archive=checked_archive(folder,'forward-01-input.pt',record['forward-01-input.pt'])
        need(archive['global_step']==0 and archive['micro_ordinal']==1,'ARCHIVED_FIRST_FORWARD')
        archive_hash=digest((archive['args'],archive['kwargs']))
        need(digest((values,options))==archive_hash,'ACTUAL_FIRST_INPUT_DIFFERS_FROM_ORIGINAL')
        native_rng=digest(rng())
        rng_state=archive['RNG_before_forward']
        target_rng_hash=digest(rng_state)
        del archive
        guards=digest((values,options),version=True)
        model_state=ForwardState(model)
        need(any(k=='rope_deltas' for _n,k in model_state.attrs),'ORIGINAL_ROPE_DELTAS_UNOBSERVED')
        hooks=LayerBoundaries(core) if rank in RANK_BOUNDARIES else None
        points=[]
        self.detail=DetailTrace(core,rank,DECODER_FOCUS,self.text_base.branches(rank)) if rank in (1,5) else None
        try:
            for occurrence,base_arm in enumerate(ARMS):
                check()
                self.fc2_preferences.arm='cublaslt'
                self.text_base.arm=base_arm
                self.vision_calls.clear()
                if occurrence: model_state.restore()
                need(digest((values,options),version=True)==guards and
                     all(p.grad is None for _,p in self.named),'NATIVE_MUTATION_BEFORE_FORWARD')
                restore_rng(rng_state)
                start_hash=digest(rng())
                need(start_hash==target_rng_hash,'ACTUAL_RNG_RESTORE_FAILED')
                initial_calls=self.math_routes
                if hooks:hooks.points.clear()
                output=original(*values,**options)
                fc2_calls=self.fc2_preferences.finish()
                text_calls=self.text_base.finish()
                need(len(self.vision_calls)>=self.vision_layers,'VISION_BACKEND_NOT_EXERCISED')
                need(output is not None and 'loss' in output and 'logits' in output,'NATIVE_TRAINING_OUTPUT')
                loss=float(output['loss'].detach().item())
                logits=output['logits']
                need(math.isfinite(loss) and logits.shape[-1]==151936 and
                     bool(torch.isfinite(logits).all()) and self.math_routes>initial_calls,
                     'FINITE_ORIGINAL_OUTPUT_AND_MATH_ROUTE')
                output_hash=describe(logits)
                del logits,output
                model_state.immutable()
                need(digest((values,options),version=True)==guards and all(p.grad is None for _,p in self.named),
                     'NATIVE_MUTATION_AFTER_FORWARD')
                detailed, retained = self.detail.finish() if self.detail else ([], 0)
                points.append(dict(loss=loss,logits=output_hash,boundaries=dict(hooks.finish()) if hooks else {},
                                   fc2_preference='cublaslt',fc2_calls=fc2_calls,
                                   vproj_layout=base_arm,text_base_calls=text_calls,
                                   vision_calls=list(self.vision_calls),
                                   detailed=detailed,retained_logical_tensor_bytes=retained,
                                   math_sdpa_calls=self.math_routes-initial_calls,entry_rng=start_hash,
                                   post_rng=digest(rng()),post_model_state=model_state.state_digest()))
        finally:
            if hooks:hooks.close()
            if self.detail:self.detail.close()
            self.text_base.close()
            self.fc2_preferences.close()
            check()
        target_records = []
        isolated = None
        if rank == 1:
            target = self.text_base.target
            need(tuple(r['layout'] for r in target) == ARMS and self.text_base.operands is not None,
                 'ACTUAL_VPROJ_ONCE_PER_ARM')
            target_records = [dict(layout=r['layout'], input=describe(r['input']),
                                   output=describe(r['output'])) for r in target]
            isolated = profile_vproj(*self.text_base.operands, check)
        else:
            need(not self.text_base.target and self.text_base.operands is None,
                 'NO_OTHER_RANK_VPROJ_INTERVENTION')
        self.text_base.target.clear()
        self.text_base.operands = None
        def compare(a,b):
            need(a['math_sdpa_calls']==b['math_sdpa_calls'],'MATH_ROUTE_COUNT_CHANGED')
            need(list(a['boundaries'])==list(b['boundaries']),'LAYER_ORDER_CHANGED')
            need([x['boundary'] for x in a['detailed']]==[x['boundary'] for x in b['detailed']],
                 'DETAILED_CALL_ORDER_CHANGED')
            first=next((k for k in a['boundaries'] if a['boundaries'][k]!=b['boundaries'][k]),None)
            first_detail=next((x['boundary'] for x,y in zip(a['detailed'],b['detailed']) if x!=y),None)
            return dict(first_different_boundary=first,first_different_detailed_boundary=first_detail,
                        detailed_count=len(a['detailed']),logits_equal=a['logits']==b['logits'],loss_equal=a['loss']==b['loss'])
        shapes=[[{k:v for k,v in receipt.items() if k!='allowed'} for receipt in p['vision_calls']] for p in points]
        need(all(s==shapes[0] for s in shapes[1:]),'VISION_OPERAND_GEOMETRY_CHANGED')
        fc2_geometry=[[{k:v for k,v in call.items() if k!='actual_preference'} for call in row['fc2_calls']]
                      for row in points]
        need(all(g==fc2_geometry[0] for g in fc2_geometry[1:]),'FC2_PAIRED_OPERAND_GEOMETRY_CHANGED')
        need(all(row['vision_calls']==points[0]['vision_calls'] for row in points[1:]),
             'NON_FC2_VISION_SDPA_ROUTE_CHANGED')
        text_geometry=[[{k:v for k,v in call.items() if k!='preference'} for call in row['text_base_calls']]
                       for row in points]
        need(all(g==text_geometry[0] for g in text_geometry[1:]),'ORIGINAL_TEXT_BASE_GEOMETRY')
        comparisons=dict(native_rows=compare(points[0],points[3]),
                         aligned_rows=compare(points[1],points[2]),
                         cross_first=compare(points[0],points[1]))
        return dict(rank=rank,original_input_sha=archive_hash,matched_original_trainable_values=504,
            matched_initial_rng_to_live_native=native_rng==target_rng_hash,original_rng_sha=target_rng_hash,
            original_recorder_validated_and_detached=True,pinned_rank_rng_restored_for_every_forward=True,registered_and_known_mutable_state_bytes=model_state.bytes,
            entry_mutable_state_sha=model_state.state_hash,retained_grad_enabled=True,backward_calls=0,optimizer_calls=0,
            vproj_layout_pair_comparisons=comparisons, actual_target_records=target_records,
            captured_vproj_operator=isolated, target_site=SITE if rank == 1 else None,
            matched_vproj_inputs_across_arms=(len({r['input']['SHA256'] for r in target_records}) == 1)
            if rank == 1 else None,
            retained_vproj_capture_bytes=sum(r['input']['shape'][1]*(4096+1024)*2 for r in target_records),
            base_inventory=self.text_base.inventory, peft_linear_source_SHA256=self.text_base.lora_provider,
            inherited_blas_preference=str(self.text_base.inherited),
            decoder_focus=list(DECODER_FOCUS) if rank in (1,5) else [],
            actual_library_kernel_proven=False,vision_attention_blocks=self.vision_layers,
            records=points)


need(KEY not in callbacks_map,'REPLAY_CALLBACK_ALREADY_REGISTERED')
callbacks_map[KEY]=NativeInitialForwardRepeat
