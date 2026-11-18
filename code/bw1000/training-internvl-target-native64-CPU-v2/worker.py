"""New target native64+metaPEFT ordinary CPU qualification; no historical query replay."""
import argparse
import copy
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import random
import resource
import signal
import sys
import time
from types import SimpleNamespace
from common import D,P,RUN,read,pin,save,require,admission,verify

W=RUN
S=D
rows=[]
phase='admission'
attempts=[]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    require(time.time()<WORK_DEADLINE and time.monotonic()<MONO_END,'ORIGINAL_WORK_DEADLINE_EXHAUSTED')


def progress(failed=False):
    save(RUN/'caller-progress.json',dict(planned_rows=64,completed_rows=len(rows),missing_rows=64-len(rows),failed=failed),replace=True)


def guard_open(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise RuntimeError('network access forbidden in offline native CPU worker')
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if path=='/dev/kfd' or path.startswith(('/dev/nvidia','/dev/dri/renderD')):
            raise RuntimeError('explicit Python device open forbidden in CPU caller scope')
        if path.endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
            attempts.append(hashlib.sha256(os.fsencode(path)).hexdigest())
            raise RuntimeError('weight or capture file read forbidden in this target CPU scope')


def verify_input_files():
    metadata=read(D/'MODEL_METADATA.json');require(len(metadata['files'])==13,'exact13 metadata')
    for name,expected in metadata['files'].items():
        check();require(pin(Path(P['model_metadata_root'])/name)=={k:expected[k] for k in ('bytes','SHA256')},'model metadata identity differs')
    require(pin(P['dataset'])==dict(bytes=P['dataset_bytes'],SHA256=P['dataset_SHA256']),'actual original HIP dataset differs')
    original=read(D/'ORIGINAL_INPUT_FILES.json')
    # Only image/plugin assets are read; the original cuda JSONL is not substituted for HIP JSONL.
    for item in original['files']:
        if item['name']=='original.cuda.jsonl':continue
        check();path=Path(P['assets_root'])/item['name']
        require(pin(path)=={k:item[k] for k in ('bytes','SHA256')},'exact shared input member differs')


def own_device_fds():
    found=[]
    for item in Path('/proc/self/fd').iterdir():
        try:target=os.readlink(item)
        except FileNotFoundError:continue
        if target=='/dev/kfd' or target.startswith(('/dev/dri/renderD','/dev/nvidia')):found.append(item.name)
    return found


def prepare_metadata_view():
    root=RUN/P['model_view_basename'];root.mkdir(mode=0o700)
    metadata=read(D/'MODEL_METADATA.json')
    for name,expected in metadata['files'].items():
        require(Path(name).name==name and name.endswith(('.json','.txt','.jinja')),'metadata member format')
        source=Path(P['model_metadata_root'])/name
        require(pin(source)=={k:expected[k] for k in ('bytes','SHA256')},'original13 metadata differs')
        raw=source.read_bytes()
        require(dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())=={k:expected[k] for k in ('bytes','SHA256')},'metadata copy changed')
        with (root/name).open('xb') as stream:stream.write(raw)
    require({p.name for p in root.iterdir()}==set(metadata['files']),'metadata only13 view')


def run_native(pre):
    global phase,F
    F=Path(P['framework_root']);root=RUN/P['model_view_basename']
    prior=read(S/'INPUT_REFERENCE.json');reference=prior['identities'];require(len(reference)==64,'original premodel64 reference')
    fm=read(D/'FRAMEWORK_FILES.json')
    for name,expected in fm['files'].items():
        check();require(pin(F/name)=={k:expected[k] for k in ('bytes','SHA256')},'actual framework source closure differs')
    sys.path.insert(0,str(F))
    binding=json.loads((S/'NATIVE_BINDING.json').read_bytes());assert binding['spec']['case']=='T03.sft' and len(binding['trainable_manifest'])==504
    parameters=sum(p['numel'] for p in binding['trainable_manifest']);assert parameters==43646976
    from tools.hygon_vlm_recipe import build_launch
    from vlm.runtime.template_inputs import settings_from_launch
    from vlm.runtime.native_recipe_contract import parameter_manifest,validate_native_recipe
    from vlm.runtime.checkpointing_lifecycle import provider_identity,validate_checkpointing_lifecycle
    dataset=Path(P['dataset'])
    launch=build_launch('T03.sft',(S/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath(str(dataset)),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
    settings=settings_from_launch(launch);assert settings==prior['capture']['settings'] and settings['template_type'] is None
    phase='actual-native-caller-setup'
    import numpy as np
    import torch
    assert torch.cuda.__dict__.get('_initialized') is False;torch.set_num_threads(2)
    def no_device(*args,**kwargs):raise RuntimeError('device initialization forbidden in CPU caller scope')
    torch.cuda._lazy_init=no_device
    random.seed(settings['seed']);np.random.seed(settings['seed']);torch.random.default_generator.manual_seed(settings['seed'])
    from swift.model import get_model_processor
    from swift.template import get_template
    empty,processor=get_model_processor(str(root),model_type='internvl',load_model=False,download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16)
    assert empty is None and processor.model_meta.template=='internvl_hf'
    template=get_template(processor,template_type=None,max_length=4096,max_pixels=settings['max_pixels'],padding_side='right',padding_free=False);template.set_mode('train')
    assert template.sequence_parallel_size==1 and not template.padding_free
    eos=processor.tokenizer.eos_token_id;eos_ids=eos if isinstance(eos,list) else [eos]
    require(bool(eos_ids) and all(type(x) is int for x in eos_ids),'native EOS identity missing')
    phase='actual-meta-native-PEFT-and-language-GC'
    from transformers import InternVLConfig,InternVLForConditionalGeneration
    config=InternVLConfig.from_pretrained(root,local_files_only=True)
    config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
    with torch.device('meta'):core=InternVLForConditionalGeneration(config)
    core.to(dtype=torch.bfloat16);core.requires_grad_(False);core.model_meta=processor.model_meta;core.model_info=processor.model_info
    from dataclasses import dataclass
    from swift.arguments.tuner_args import TunerArguments
    from swift.pipelines.train.tuner import get_target_modules,prepare_adapter
    @dataclass
    class NativeTunerArguments(TunerArguments):
        model_meta:object=None
        tuner_type:str='lora'
    tuner=NativeTunerArguments(lora_rank=16,lora_alpha=32,lora_dropout=.05,freeze_vit=True,model_meta=processor.model_meta)
    require(tuner.freeze_aligner is True and tuner.freeze_llm is False,'original freeze boundary')
    tuner.task_type='causal_lm';tuner.tuner_type='lora';tuner.tuner_backend='peft';tuner.use_swift_lora=False
    target=get_target_modules(tuner,core)
    with torch.device('meta'):model=prepare_adapter(tuner,core,task_type='CAUSAL_LM')
    from swift.trainers import Seq2SeqTrainingArguments
    from swift.trainers.mixin import SwiftMixin
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-output'),use_cpu=True,report_to='none',vit_gradient_checkpointing=False,**binding['spec']['training'])
    method=SwiftMixin._prepare_gradient_checkpointing
    method(SimpleNamespace(args=args),core)
    from providers import method_identity,bind_encode,model_source_identities
    actual_model_sources=model_source_identities(model,binding['model_source_SHA256'])
    # Source and structure are measured by this CPU scope; A100 runtime identity is not inherited.
    measured=copy.deepcopy(binding)
    measured['target_modules']=target
    measured['trainable_manifest']=parameter_manifest(sorted((n,p) for n,p in model.named_parameters() if p.requires_grad))
    require(measured['trainable_manifest']==binding['trainable_manifest'],'actual PEFT named504 dtype/shape scope differs')
    measured['native_checkpointing_lifecycle']['provider']=provider_identity(method)
    verified,frozen=validate_native_recipe(model,args,measured,checkpointing_method=method)
    lifecycle=validate_checkpointing_lifecycle(core,args,measured,method)
    require(len(verified)==504 and sum(p.numel() for _,p in verified)==43646976,'actual native PEFT tensor/parameter count')
    require(all(p.device.type=='meta' for p in model.parameters()),'only meta model and adapter allowed')
    save(RUN/'native-PEFT-structure.json',dict(trainable_manifest=measured['trainable_manifest'],target_modules=target,frozen=frozen,lifecycle=lifecycle,actual_model_sources=actual_model_sources,
         actual_trainable_tensors=504,actual_trainable_parameters=43646976,initial_values_qualified=False))
    holder=SimpleNamespace(args=args,template=template,model=core)
    assert args.use_logits_to_keep is None
    use_logits=SwiftMixin.get_use_logits_to_keep(holder,template.sequence_parallel_size==1)
    assert use_logits is True and args.use_logits_to_keep is True
    from vlm.runtime.native_caller_budget import plan_capture,validate_capture_conditions
    conditions=dict(case='T03.sft',hidden_size=4096,vocab_size=151936,language_cache=False,hidden_state_stacks=False,attention_maps=False)
    guard=validate_capture_conditions(core,conditions)
    phase='actual-native-method-provider-chain'
    providers={'encode':bind_encode(template,torch.inference_mode)}
    for name,method in [('collator',template.data_collator),('base_collator',template._data_collator),('logit_policy',SwiftMixin.get_use_logits_to_keep),('logit_selection',SwiftMixin.prepare_logits_to_keep),('native_prepare_inputs',Seq2SeqTrainer._prepare_inputs),('post_encode',template._post_encode),('prepare_adapter',prepare_adapter),('GC',SwiftMixin._prepare_gradient_checkpointing)]:
        providers[name]=method_identity(method)
    save(RUN/'native-caller-providers.json',providers)
    from vlm.runtime.first_update_audit import fingerprint
    canonical=F/'vlm/runtime/first_update_audit.py'
    assert sha(canonical)=='816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49' and Path(inspect.getsourcefile(fingerprint)).resolve()==canonical.resolve()
    def identity(v):return {k:x for k,x in fingerprint(v).items() if k!='floating_l2'}
    def fields(v):return {k:identity(x) for k,x in sorted(v.items())}
    class CountSink:
     def __init__(self):self.count=0
     def write(self,data):
      if self.count+len(data)>2*1024**3-1:raise ValueError('complete collated artifact exceeds unchanged per-file limit')
      self.count+=len(data);return len(data)
     def flush(self):pass
     def tell(self):return self.count
    raw_rows=[json.loads(x) for x in dataset.read_bytes().splitlines()];assert len(raw_rows)==64
    phase='actual64-native-collator-and-logit-selection'
    for ordinal,raw_row in enumerate(raw_rows,1):
     check();encoded=template.encode(raw_row);encoded_fields=fields(encoded)
     assert encoded_fields==reference[ordinal-1]['fields'] and len(encoded['input_ids'])==reference[ordinal-1]['sequence_length']
     collated=template.data_collator([copy.deepcopy(encoded)])
     assert collated['input_ids'].ndim==collated['labels'].ndim==2 and collated['input_ids'].shape==collated['labels'].shape and collated['input_ids'].shape[0]==1
     length=collated['input_ids'].shape[1];assert length==len(encoded['input_ids']) and 0<length<=4096
     before=fields(collated);original_labels=collated['labels'].clone();loss_mask=(original_labels!=-100)[0];assert not bool(loss_mask[0])
     require(torch.equal(original_labels[0][loss_mask],collated['input_ids'][0][loss_mask]),'actual unmasked labels differ from native input IDs')
     eos_supervised=sum(int(((collated['input_ids'][0]==token)&loss_mask).sum()) for token in eos_ids)
     require(eos_supervised>0,'native supervised EOS absent; no literal token append')
     effective_tokens=int(((original_labels[0]!=-100)&(collated['attention_mask'][0]!=0))[1:].sum())
     require(effective_tokens>0,'actual shifted supervised tokens absent')
     image_tokens=int((collated['input_ids']==config.image_token_id).sum());assert image_tokens>0
     supervised=int(loss_mask.sum());assert supervised>0
     SwiftMixin.prepare_logits_to_keep(holder,collated)
     selected=collated['logits_to_keep'];assert selected.dtype==torch.bool and list(selected.shape)==[length]
     assert torch.equal(selected,torch.nn.functional.pad(loss_mask[1:],(0,1),value=True))
     assert torch.equal(collated['labels'],torch.nn.functional.pad(original_labels[:,loss_mask],(1,0),value=-100))
     selected_rows=int(selected.sum());assert selected_rows==supervised+1==collated['labels'].shape[1]
     after=fields(collated)
     unchanged=set(before)-{'labels'};assert all(before[k]==after[k] for k in unchanged) and set(after)==set(before)|{'logits_to_keep'}
     sink=CountSink();torch.save(dict(args=(),kwargs=collated),sink)
     rows.append(dict(ordinal=ordinal,sequence_length=length,selected_logit_rows=selected_rows,supervised_tokens=supervised,image_tokens=image_tokens,
      effective_shifted_supervised_tokens=effective_tokens,supervised_EOS_count=eos_supervised,native_EOS_ids=eos_ids,
      collated_serialized_bytes=sink.count,original_encoded_fields_exact=True,complete_input_sequence_unchanged=True,non_label_fields_unchanged=True,
      label_and_logit_mask_native_semantics_checked=True,before_fields=before,after_fields=after))
     del encoded,collated,original_labels,loss_mask,selected
     progress()

    save(RUN/'native-caller-rows.json',rows)
    require(len(rows)==64 and torch.cuda.__dict__.get('_initialized') is False and not attempts
            and all(parameter.device.type=='meta' for parameter in model.parameters()),'full64/no-device/meta-only final boundary')
    return dict(passed=True,planned_rows=64,completed_rows=64,missing_rows=0,pre_model64_reference_exact=True,
                actual_trainable_tensors=504,actual_trainable_parameters=43646976,native_supervised_EOS_rows=64,
                actual_shifted_supervised_tokens=sum(r['effective_shifted_supervised_tokens'] for r in rows),
                complete_sequence_length_range=[min(r['sequence_length'] for r in rows),max(r['sequence_length'] for r in rows)],
                native_logits_policy=use_logits,providers=providers,weight_file_open_attempts=attempts,
                actual_model_loaded=False,meta_model_only=True,image_embedding_forward=False,full_Trainer_constructed=False,
                training_forward_backward=False,canonical_post_model64_equality_qualified=False,RNG_parity_qualified=False,
                GPU=0,GPU_GO=False,Model_GO=False,training_numerical_GO=False)


def main(epoch,deadline,work_deadline):
    global WORK_DEADLINE,MONO_END,phase
    require(deadline==epoch+300 and time.time()<work_deadline<deadline,'original worker300 epoch')
    WORK_DEADLINE=work_deadline;MONO_END=time.monotonic()+work_deadline-time.time()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original worker deadline')))
    signal.setitimer(signal.ITIMER_REAL,work_deadline-time.time())
    # This executes before Torch/numpy/Swift, their loaders, provider reads or input payloads.
    pre=admission()
    verify();require(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),'noGPU visibility')
    from bounded import identity
    save(RUN/'worker-owner.json',dict(owner=identity(),CPU_affinity=sorted(os.sched_getaffinity(0)),epoch=epoch,deadline=deadline,work_deadline=work_deadline))
    sys.addaudithook(guard_open)
    started=time.monotonic();progress();after_fds=None
    try:
        require(own_device_fds()==[],'worker owns device FDs before import')
        phase='original_input_and_source_closure';verify_input_files();prepare_metadata_view()
        phase='native_processor_and_complete64_caller';result=run_native(pre)
        after_fds=own_device_fds();require(after_fds==[],'worker owns device FDs after native caller')
    except BaseException as error:
        import traceback
        save(RUN/'failure-private.json',dict(phase=phase,error_type=type(error).__name__,message=str(error)[:4096],trace=traceback.format_exc()[-4096:]))
        progress(True)
        result=dict(passed=False,phase=phase,error_type=type(error).__name__,planned_rows=64,completed_rows=len(rows),
                    missing_rows=64-len(rows),GPU=0,GPU_GO=False,Model_GO=False,training_numerical_GO=False)
    result.update(epoch=epoch,deadline=deadline,work_deadline=work_deadline,wall_seconds=time.monotonic()-started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  implicit_library_device_queries='UNKNOWN',raw_encoded_values_saved=False,own_device_FDs_after=after_fds,
                  actual_trainable_values_loaded=False,meta_PEFT_structure_only=True,provider_qualification_inherited=False)
    save(RUN/'worker-result.json',result)
    print(json.dumps({k:result[k] for k in ('passed','completed_rows','planned_rows','missing_rows','GPU')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=float,required=True);parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--work-deadline',type=float,required=True)
    a=parser.parse_args();raise SystemExit(main(a.epoch,a.deadline,a.work_deadline))
