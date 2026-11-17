"""Target-only native64 CPU worker. Never import Torch while prerequisites are pending."""
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
from scope import D,P,RUN,read,pin,save,require,admission,verify

W=RUN
S=D
rows=[]
phase='admission'
attempts=[]
ADMISSION_CLOSED=False


def close_admission(claim,prerequisites):
    global ADMISSION_CLOSED
    # A root/CLI boolean alone cannot close the pending prerequisite chain.
    qualified=admission(claim,prerequisites)
    ADMISSION_CLOSED=True
    return qualified


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    require(time.time()<WORK_DEADLINE and time.monotonic()<MONO_END,'ORIGINAL_WORK_DEADLINE_EXHAUSTED')


def progress(failed=False):
    save(RUN/'caller-progress.json',dict(planned_rows=64,completed_rows=len(rows),missing_rows=64-len(rows),failed=failed),replace=True)


def guard_open(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if path.endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
            attempts.append(hashlib.sha256(os.fsencode(path)).hexdigest())
            raise RuntimeError('weight or capture file read forbidden in this target CPU scope')
        if '/torch/version.py' in path and not ADMISSION_CLOSED:
            raise RuntimeError('pending Torch version file cannot be returned through this new scope')


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


def run_native(pre):
    global phase,F
    F=Path(pre['framework_root']);root=Path(P['model_metadata_root'])
    prior=read(S/'INPUT_REFERENCE.json');reference=prior['identities'];require(len(reference)==64,'original premodel64 reference')
    fm=read(D/'FRAMEWORK_FILES.json')
    for name,expected in fm['files'].items():
        check();require(pin(F/name)=={k:expected[k] for k in ('bytes','SHA256')},'actual framework source closure differs')
    sys.path.insert(0,str(F))
    binding=json.loads((S/'NATIVE_BINDING.json').read_bytes());assert binding['spec']['case']=='T03.sft' and len(binding['trainable_manifest'])==504
    parameters=sum(p['numel'] for p in binding['trainable_manifest']);assert parameters==43646976
    from tools.hygon_vlm_recipe import build_launch
    from vlm.runtime.template_inputs import settings_from_launch
    dataset=Path(P['dataset'])
    launch=build_launch('T03.sft',(S/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath(str(dataset)),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
    settings=settings_from_launch(launch);assert settings==prior['capture']['settings'] and settings['template_type'] is None
    phase='actual-native-caller-setup'
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    def no_device(*args,**kwargs):raise RuntimeError('device initialization forbidden in CPU caller scope')
    torch.cuda._lazy_init=no_device
    random.seed(settings['seed']);np.random.seed(settings['seed']);torch.random.default_generator.manual_seed(settings['seed'])
    from swift.model import get_model_processor
    from swift.template import get_template
    empty,processor=get_model_processor(str(root),model_type='internvl',load_model=False,download_model=False,device_map='cpu',attn_impl='sdpa',torch_dtype=torch.bfloat16)
    assert empty is None and processor.model_meta.template=='internvl_hf'
    template=get_template(processor,template_type=None,max_length=4096,max_pixels=settings['max_pixels'],padding_side='right',padding_free=False);template.set_mode('train')
    assert template.sequence_parallel_size==1 and not template.padding_free
    from transformers import InternVLConfig,InternVLForConditionalGeneration
    config=InternVLConfig.from_pretrained(root,local_files_only=True)
    config._attn_implementation='sdpa';config.text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
    with torch.device('meta'):core=InternVLForConditionalGeneration(config)
    core.to(dtype=torch.bfloat16);core.requires_grad_(False);core.model_meta=processor.model_meta;core.model_info=processor.model_info
    from swift.trainers import Seq2SeqTrainingArguments
    from swift.trainers.mixin import SwiftMixin
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-output'),use_cpu=True,report_to='none',vit_gradient_checkpointing=False,**binding['spec']['training'])
    SwiftMixin._prepare_gradient_checkpointing(SimpleNamespace(args=args),core)
    holder=SimpleNamespace(args=args,template=template,model=core)
    assert args.use_logits_to_keep is None
    use_logits=SwiftMixin.get_use_logits_to_keep(holder,template.sequence_parallel_size==1)
    assert use_logits is True and args.use_logits_to_keep is True
    from vlm.runtime.native_caller_budget import plan_capture,validate_capture_conditions
    conditions=dict(case='T03.sft',hidden_size=4096,vocab_size=151936,language_cache=False,hidden_state_stacks=False,attention_maps=False)
    guard=validate_capture_conditions(core,conditions)
    from wrapper_binding import bind_encode
    providers={'encode':bind_encode(template,pre['hygon_encode_wrapper_policy'],torch.inference_mode)}
    for name,method in [('collator',template.data_collator),('base_collator',template._data_collator),('logit_policy',SwiftMixin.get_use_logits_to_keep),('logit_selection',SwiftMixin.prepare_logits_to_keep),('native_prepare_inputs',Seq2SeqTrainer._prepare_inputs),('post_encode',template._post_encode)]:
        pin=pre['native_method_providers'][name]
        path=Path(inspect.getsourcefile(method)).resolve()
        assert str(path)==pin['path'] and sha(path)==pin['SHA256'] and path.stat().st_size==pin['bytes']
        providers[name]=dict(module=method.__module__,name=method.__qualname__,path=str(path),bytes=pin['bytes'],SHA256=pin['SHA256'])
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
      collated_serialized_bytes=sink.count,original_encoded_fields_exact=True,complete_input_sequence_unchanged=True,non_label_fields_unchanged=True,
      label_and_logit_mask_native_semantics_checked=True,before_fields=before,after_fields=after))
     del encoded,collated,original_labels,loss_mask,selected
     progress()

    save(RUN/'native-caller-rows.json',rows)
    save(RUN/'native-caller-providers.json',providers)
    require(len(rows)==64 and not torch.cuda.is_initialized() and not attempts
            and all(parameter.device.type=='meta' for parameter in core.parameters()),'full64/no-device/meta-only final boundary')
    return dict(passed=True,planned_rows=64,completed_rows=64,missing_rows=0,pre_model64_reference_exact=True,
                native_logits_policy=use_logits,providers=providers,weight_file_open_attempts=attempts,
                actual_model_loaded=False,meta_model_only=True,image_embedding_forward=False,full_Trainer_constructed=False,
                training_forward_backward=False,canonical_post_model64_equality_qualified=False,RNG_parity_qualified=False,
                GPU=0,GPU_GO=False,Model_GO=False,training_numerical_GO=False)


def main(epoch,deadline,work_deadline):
    global WORK_DEADLINE,MONO_END,phase
    require(deadline==epoch+900 and time.time()<work_deadline<deadline,'original worker900 epoch')
    WORK_DEADLINE=work_deadline;MONO_END=time.monotonic()+work_deadline-time.time()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original worker deadline')))
    signal.setitimer(signal.ITIMER_REAL,work_deadline-time.time())
    # This executes before Torch/numpy/Swift, their loaders, provider reads or input payloads.
    pre=close_admission(read(D/'CLAIM.json'),read(D/'PREREQUISITES.json'))
    verify();require(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),'noGPU visibility')
    sys.addaudithook(guard_open)
    started=time.monotonic();progress()
    try:
        phase='original_input_and_source_closure';verify_input_files()
        phase='native_processor_and_complete64_caller';result=run_native(pre)
    except BaseException as error:
        progress(True)
        result=dict(passed=False,phase=phase,error_type=type(error).__name__,planned_rows=64,completed_rows=len(rows),
                    missing_rows=64-len(rows),GPU=0,GPU_GO=False,Model_GO=False,training_numerical_GO=False)
    result.update(epoch=epoch,deadline=deadline,work_deadline=work_deadline,wall_seconds=time.monotonic()-started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  implicit_library_device_queries='UNKNOWN',raw_encoded_values_saved=False)
    save(RUN/'worker-result.json',result)
    print(json.dumps({k:result[k] for k in ('passed','completed_rows','planned_rows','missing_rows','GPU')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=float,required=True);parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--work-deadline',type=float,required=True)
    a=parser.parse_args();raise SystemExit(main(a.epoch,a.deadline,a.work_deadline))
