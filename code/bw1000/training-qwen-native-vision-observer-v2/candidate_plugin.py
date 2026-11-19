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
from vision_trace import VisionTrace
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.checkpointing_lifecycle import validate_checkpointing_lifecycle

KEY='qwen_native_initial_forward_repeat'
RANK_BOUNDARIES={0,1,4,5}


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
        registry=qwen.ALL_ATTENTION_FUNCTIONS
        source=registry['sdpa']
        need(source is sdpa_attention_forward,'FIXED_SDPA_PROVIDER')
        torch.use_deterministic_algorithms(True,warn_only=False)

        @functools.wraps(source)
        def route(module,query,key,value,attention_mask,*a,**k):
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
        self.detail=VisionTrace(core,rank) if rank in (0,1,5) else None
        try:
            for arm in range(4):
                check()
                if arm: model_state.restore()
                need(digest((values,options),version=True)==guards and
                     all(p.grad is None for _,p in self.named),'NATIVE_MUTATION_BEFORE_FORWARD')
                restore_rng(rng_state)
                start_hash=digest(rng())
                need(start_hash==target_rng_hash,'ACTUAL_RNG_RESTORE_FAILED')
                initial_calls=self.math_routes
                if hooks:hooks.points.clear()
                output=original(*values,**options)
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
                observed=self.detail.finish() if self.detail else None
                points.append(dict(loss=loss,logits=output_hash,boundaries=dict(hooks.finish()) if hooks else {},
                                   detailed=observed['records'] if observed else [],
                                   vision_output_grad_flags=observed['grad_flags'] if observed else None,
                                   math_sdpa_calls=self.math_routes-initial_calls,entry_rng=start_hash,
                                   post_rng=digest(rng()),post_model_state=model_state.state_digest()))
        finally:
            if hooks:hooks.close()
            if self.detail:self.detail.close()
        check()
        def compare(a,b):
            need(a['math_sdpa_calls']==b['math_sdpa_calls'],'MATH_ROUTE_COUNT_CHANGED')
            need(list(a['boundaries'])==list(b['boundaries']),'LAYER_ORDER_CHANGED')
            need([x['boundary'] for x in a['detailed']]==[x['boundary'] for x in b['detailed']],
                 'DETAILED_CALL_ORDER_CHANGED')
            first=next((k for k in a['boundaries'] if a['boundaries'][k]!=b['boundaries'][k]),None)
            first_detail=next((x['boundary'] for x,y in zip(a['detailed'],b['detailed']) if x!=y),None)
            return dict(first_different_boundary=first,first_different_detailed_boundary=first_detail,
                        detailed_count=len(a['detailed']),logits_equal=a['logits']==b['logits'],loss_equal=a['loss']==b['loss'])
        if self.detail:
            need(all(p['vision_output_grad_flags']==points[0]['vision_output_grad_flags'] for p in points),
                 'VISION_AUTOGRAD_BOUNDARY_CHANGED')
        native_pair=compare(points[0],points[3])
        repeat_pair=compare(points[1],points[2])
        return dict(rank=rank,original_input_sha=archive_hash,matched_original_trainable_values=504,
            matched_initial_rng_to_live_native=native_rng==target_rng_hash,original_rng_sha=target_rng_hash,
            original_recorder_validated_and_detached=True,pinned_rank_rng_restored_for_every_forward=True,registered_and_known_mutable_state_bytes=model_state.bytes,
            entry_mutable_state_sha=model_state.state_hash,retained_grad_enabled=True,backward_calls=0,optimizer_calls=0,
            native_pair=native_pair,repeat_native_pair=repeat_pair,
            vision_fields_replaced=False,
            records=points)


need(KEY not in callbacks_map,'REPLAY_CALLBACK_ALREADY_REGISTERED')
callbacks_map[KEY]=NativeInitialForwardRepeat
