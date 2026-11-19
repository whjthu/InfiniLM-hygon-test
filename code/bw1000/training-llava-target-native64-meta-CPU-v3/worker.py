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
    metadata=read(D/'MODEL_METADATA.json');require(len(metadata['files'])==14,'exact14 metadata')
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
        require(Path(name).name==name and name.endswith(('.json','.txt','.jinja','.py')),'metadata member format')
        source=Path(P['model_metadata_root'])/name
        require(pin(source)=={k:expected[k] for k in ('bytes','SHA256')},'original14 metadata differs')
        raw=source.read_bytes()
        require(dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())=={k:expected[k] for k in ('bytes','SHA256')},'metadata copy changed')
        with (root/name).open('xb') as stream:stream.write(raw)
    require({p.name for p in root.iterdir()}==set(metadata['files']),'metadata only14 view')


def run_native(pre):
    global phase,F
    F=Path(P['framework_root']);root=RUN/P['model_view_basename'];prior=read(S/'INPUT_REFERENCE.json');reference=prior['identities'];caller=read(S/'COLLATOR_REFERENCE.json')['rows'];require(len(reference)==len(caller)==64,'A100_PREMODEL64_REFERENCE')
    for name,expected in read(D/'FRAMEWORK_FILES.json')['files'].items():check();require(pin(F/name)=={k:expected[k] for k in ('bytes','SHA256')},'actual common393 closure')
    sys.path.insert(0,str(F));binding=read(S/'NATIVE_BINDING.json');require(binding['spec']['case']=='T04.sft' and len(binding['trainable_manifest'])==504,'T04_NATIVE_BINDING')
    from tools.hygon_vlm_recipe import build_launch
    from vlm.runtime.template_inputs import settings_from_launch
    from vlm.runtime.native_recipe_contract import parameter_manifest,validate_native_recipe
    dataset=Path(P['dataset']);launch=build_launch('T04.sft',(S/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(root.parent)),PurePosixPath(str(dataset)),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501);settings=settings_from_launch(launch)
    require(settings==prior['capture']['settings'] and settings['template_type'] is None and settings['attention']=='eager','ORIGINAL_T04_FULL4096_SETTINGS')
    phase='actual-native-T04-processor'
    import numpy as np
    import torch
    require(torch.cuda.__dict__.get('_initialized') is False,'CPU_NO_CUDA_INIT');torch.set_num_threads(2)
    def no_device(*args,**kw):raise RuntimeError('explicit GPU init forbidden in CPU scope')
    torch.cuda._lazy_init=no_device;random.seed(42);np.random.seed(42);torch.random.default_generator.manual_seed(42)
    from vlm.plugins import llava_onevision as compatibility
    from swift.model import get_model_processor
    from swift.template import get_template
    empty,processor=get_model_processor(str(root),model_type='llava_onevision1_5',load_model=False,download_model=False,device_map='cpu',attn_impl='eager',torch_dtype=torch.bfloat16)
    require(empty is None and processor.model_meta.template=='llava_onevision1_5','T04_PROCESSOR_NO_MODEL')
    template=get_template(processor,template_type=None,max_length=4096,max_pixels=262144,padding_side='right',padding_free=False);template.set_mode('train');require(template.template_meta.template_type=='llava_onevision1_5' and template.sequence_parallel_size==1 and not template.padding_free,'NATIVE_TEMPLATE')
    eos=processor.tokenizer.eos_token_id;eos_ids=eos if isinstance(eos,list) else [eos];require(eos_ids and all(type(x)is int for x in eos_ids),'ACTUAL_EOS')
    phase='actual-custom-meta-model-504'
    import transformers
    from transformers import AutoConfig, cache_utils
    from cache_alias import scoped_sliding_cache_alias
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    config=AutoConfig.from_pretrained(root,trust_remote_code=True,local_files_only=True)
    for part in (config,config.text_config,config.vision_config):part._attn_implementation='eager'
    contract = P['cache_alias_contract']
    require(transformers.__version__ == contract['transformers_version'], 'EXACT_TARGET_TRANSFORMERS_VERSION')
    provider = pin(Path(cache_utils.__file__).resolve())
    require(binding['model_source_SHA256'] == contract['model_sha'] and
            sha(root/'modeling_llavaonevision1_5.py') == contract['model_sha'],
            'EXACT_ONEVISION_CONSUMER_BEFORE_ALIAS')
    compatibility._prepare_load_config(config)
    phase = 'actual-custom-meta-import-cache-alias'
    with scoped_sliding_cache_alias(cache_utils, provider_pin=provider, expected_provider=contract['provider']) as static_cache:
        compatibility._preimport_verified_eager_class(str(root),config,{'attn_implementation':'eager'})
        cls = get_class_from_dynamic_module('modeling_llavaonevision1_5.LLaVAOneVision1_5_ForConditionalGeneration',
                                            str(root), local_files_only=True)
        require(vars(sys.modules[cls.__module__]).get('SlidingWindowCache') is static_cache,
                'NATIVE_CONSUMER_BINDS_THE_UPSTREAM_ALIAS')
    require('SlidingWindowCache' not in vars(cache_utils), 'TRANSFORMERS_CACHE_EXPORT_RESTORED')
    save(RUN/'cache-alias-observation.json', dict(provider=provider, model_SHA256=contract['model_sha'],
         transformers_version=transformers.__version__, static_cache_bound=True, temporary_export_restored=True))
    phase = 'actual-custom-meta-model-504'
    require(sha(Path(inspect.getsourcefile(cls)))==binding['model_source_SHA256'],'EXACT_CUSTOM_MODEL_SOURCE')
    with torch.device('meta'):core=cls(config)
    core.to(dtype=torch.bfloat16);core.requires_grad_(False);core.model_meta=processor.model_meta;core.model_info=processor.model_info;compatibility._verify_loaded_native_eager_classes(core,cls.__module__)
    require(not getattr(core,'_vlm_onevision_rope_initialized',False),'MATERIALIZED_ROPE_NOT_QUALIFIED')
    from dataclasses import dataclass
    from swift.arguments.tuner_args import TunerArguments
    from swift.pipelines.train.tuner import get_target_modules,prepare_adapter
    @dataclass
    class NativeTunerArguments(TunerArguments):
        model_meta:object=None
        tuner_type:str='lora'
    tuner=NativeTunerArguments(lora_rank=16,lora_alpha=32,lora_dropout=.05,freeze_vit=True,model_meta=processor.model_meta);require(tuner.freeze_aligner is True and tuner.freeze_llm is False,'ORIGINAL_FREEZE_BOUNDARY');tuner.task_type='causal_lm';tuner.tuner_type='lora';tuner.tuner_backend='peft';tuner.use_swift_lora=False
    target=get_target_modules(tuner,core)
    with torch.device('meta'):model=prepare_adapter(tuner,core,task_type='CAUSAL_LM')
    from swift.trainers import Seq2SeqTrainingArguments
    from swift.trainers.mixin import SwiftMixin
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    args=Seq2SeqTrainingArguments(output_dir=str(W/'unused-output'),use_cpu=True,report_to='none',vit_gradient_checkpointing=False,**binding['spec']['training']);method=SwiftMixin._prepare_gradient_checkpointing;method(SimpleNamespace(args=args),core)
    measured=copy.deepcopy(binding);measured['target_modules']=target;measured['trainable_manifest']=parameter_manifest(sorted((n,p) for n,p in model.named_parameters() if p.requires_grad));require(measured['trainable_manifest']==binding['trainable_manifest'],'ORIGINAL504_EXACT_NAME_SHAPE_DTYPE')
    verified,frozen=validate_native_recipe(model,args,measured);require(len(verified)==504 and sum(p.numel() for _,p in verified)==43646976 and all(p.device.type=='meta' for p in model.parameters()),'COMPLETE_META504_ONLY')
    language=core.get_submodule('model.language_model');vision=core.get_submodule('model.visual');require(language.gradient_checkpointing is False and vision.gradient_checkpointing is False and args.gradient_checkpointing is False,'T04_GC_FALSE')
    from providers import method_identity,bind_encode,observe_t04_sources
    policy=read(S/'TARGET_MATH_SOURCES.json');observed=observe_t04_sources(model,policy,lambda x:save(RUN/'model-source-observation.json',x));lifecycle=dict(original_requested=False,language_checkpointing=False,vision_checkpointing=False,native_Trainer_flag=False,GC_method=method_identity(method))
    save(RUN/'native-PEFT-structure.json',dict(trainable_manifest=measured['trainable_manifest'],target_modules=target,frozen=frozen,lifecycle=lifecycle,actual_model_sources=observed,actual_trainable_tensors=504,actual_trainable_parameters=43646976,initial_values_qualified=False,platform_source_contract_SHA256=pin(S/'TARGET_MATH_SOURCES.json')['SHA256'],reference_model_source_SHA256=binding['model_source_SHA256'],measured_target_model_source_SHA256=binding['model_source_SHA256'],numerical_equivalence_proven=False))
    holder=SimpleNamespace(args=args,template=template,model=core);require(args.use_logits_to_keep is None,'NATIVE_POLICY_UNSET');use_logits=SwiftMixin.get_use_logits_to_keep(holder,True);require(use_logits is False and args.use_logits_to_keep is False,'ORIGINAL_T04_FULL_LOGITS_POLICY')
    phase='actual-native-provider-chain';providers={'encode':bind_encode(template,torch.inference_mode)}
    for name,fn in [('collator',template.data_collator),('base_collator',template._data_collator),('logit_policy',SwiftMixin.get_use_logits_to_keep),('logit_selection',SwiftMixin.prepare_logits_to_keep),('native_prepare_inputs',Seq2SeqTrainer._prepare_inputs),('post_encode',template._post_encode),('prepare_adapter',prepare_adapter),('GC',method)]:providers[name]=method_identity(fn)
    save(RUN/'native-caller-providers.json',providers)
    from vlm.runtime.first_update_audit import fingerprint
    canonical=F/'vlm/runtime/first_update_audit.py';require(sha(canonical)=='816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49','CANONICAL_FIELD_DIGEST_SOURCE')
    def fields(v):return {k:{n:x for n,x in fingerprint(x).items() if n!='floating_l2'} for k,x in sorted(v.items())}
    raw_rows=[json.loads(x) for x in dataset.read_bytes().splitlines()];require(len(raw_rows)==64,'ALL64_ORIGINAL_HIP_INPUTS');phase='actual64-native-encode-full-collator'
    for ordinal,raw_row in enumerate(raw_rows,1):
        check();encoded=template.encode(raw_row);encoded_fields=fields(encoded);length=len(encoded['input_ids']);require(0<length<=4096 and len(encoded['labels'])==length,'FULL_INPUT4096');collated=template.data_collator([copy.deepcopy(encoded)])
        ids=collated['input_ids'];labels=collated['labels'];mask=collated['attention_mask'];require(tuple(ids.shape)==tuple(labels.shape)==tuple(mask.shape)==(1,length) and torch.equal(ids[0],torch.as_tensor(encoded['input_ids'],dtype=ids.dtype)),'FULL_COLLATED_INPUT')
        active=labels[0]!=-100;require(bool(active.any()) and bool((~active).any()) and not bool(active[0]) and torch.equal(labels[0][active],ids[0][active]),'NATIVE_LOSS_MASK');eos_count=sum(int(((ids[0]==x)&active).sum()) for x in eos_ids);tokens=int(((labels[0]!=-100)&(mask[0]!=0))[1:].sum());require(eos_count>0 and tokens>0,'SUPERVISED_EOS_AND_SHIFTED_TOKENS')
        image_tokens=int((ids==config.image_token_id).sum());require(image_tokens>0 and collated['pixel_values'].numel()>0 and 'logits_to_keep' not in collated,'T04_VISION_FULL_LOGITS');before=fields(collated);require(all(v['finite'] is True for v in before.values()),'FINITE_NATIVE_FIELDS')
        after=fields(collated);require(after==before,'NATIVE_FULL_FIELDS_PRESERVED');ref=reference[ordinal-1];cref=caller[ordinal-1];encoded_exact=encoded_fields==ref['fields'] and length==ref['sequence_length'];collator_exact=before==cref['collated_fields'] and length==cref['sequence_length']
        rows.append(dict(ordinal=ordinal,sequence_length=length,selected_logit_rows=length,supervised_tokens=int(active.sum()),image_tokens=image_tokens,effective_shifted_supervised_tokens=tokens,supervised_EOS_count=eos_count,native_EOS_ids=eos_ids,encoded_fields=encoded_fields,original_encoded_fields_exact=encoded_exact,original_pre_model_collated_fields_exact=collator_exact,reference_phase='A100_CPU_pre_model_encode_collator',post_model200_compared=False,complete_input_sequence_unchanged=True,non_label_fields_unchanged=True,label_and_logit_mask_native_semantics_checked=True,before_fields=before,after_fields=after,native_logits_policy=False))
        del encoded,collated,ids,labels,mask,active;progress()
    save(RUN/'native-caller-rows.json',rows);require(len(rows)==64 and torch.cuda.__dict__.get('_initialized') is False and not attempts and all(p.device.type=='meta' for p in model.parameters()),'64_ROWS_NO_DEVICE_META_ONLY')
    return dict(passed=True,planned_rows=64,completed_rows=64,missing_rows=0,pre_model64_reference_exact=all(r['original_encoded_fields_exact'] for r in rows),pre_model64_collator_reference_exact=all(r['original_pre_model_collated_fields_exact'] for r in rows),actual_trainable_tensors=504,actual_trainable_parameters=43646976,native_supervised_EOS_rows=64,actual_shifted_supervised_tokens=sum(r['effective_shifted_supervised_tokens'] for r in rows),complete_sequence_length_range=[min(r['sequence_length'] for r in rows),max(r['sequence_length'] for r in rows)],native_logits_policy=False,logit_selection_called=False,providers=providers,weight_file_open_attempts=attempts,actual_model_loaded=False,meta_model_only=True,image_embedding_forward=False,full_Trainer_constructed=False,training_forward_backward=False,canonical_post_model64_equality_qualified=False,RNG_parity_qualified=False,GPU=0,GPU_GO=False,Model_GO=False,training_numerical_GO=False)


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
                  actual_trainable_values_loaded=False,meta_PEFT_structure_only=True,provider_qualification_inherited=False,numerical_equivalence_proven=False,platform_source_contract_only=True)
    save(RUN/'worker-result.json',result)
    print(json.dumps({k:result[k] for k in ('passed','completed_rows','planned_rows','missing_rows','GPU')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=float,required=True);parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--work-deadline',type=float,required=True)
    a=parser.parse_args();raise SystemExit(main(a.epoch,a.deadline,a.work_deadline))
