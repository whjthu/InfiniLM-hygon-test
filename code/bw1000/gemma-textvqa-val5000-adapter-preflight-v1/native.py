"""Owner-callable Swift arm adapter. This module has no model or scheduler CLI."""

from contextlib import contextmanager
import functools
from hashlib import sha256
import inspect
import os
from pathlib import Path
import stat

import time

from contract import (ANN, ARMS, CAP, CP, IDENTITIES, MMF, SHA, STRUCTURAL, checkpoint_binding,
                      compare_layer_execution, complete_records, document, identities, input_plan, need, packed, sha, skipped)


BASE_MANIFEST='5c6875cfe190980cad4315703e8d73dbd1fb803d620ea1f707ba35fe5de30c63'


def file_stamp(path, expected, check, decoder=lambda file:file.read()):
    count,digest=expected
    path=Path(path)
    before=path.lstat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and before.st_size==count,
         'CANONICAL_EXPECTED_FILE_EXTENT_REQUIRED')
    h=sha256()
    with path.open('rb') as stream:
        need(identity(os.fstat(stream.fileno()))==identity(before),'SOURCE_FD_IDENTITY_CHANGED')
        until=count
        while until:
            check()
            block=stream.read(min(until,1<<20))
            need(bool(block),'SOURCE_BYTES_TRUNCATED')
            h.update(block);until-=len(block)
        need(not stream.read(1) and h.hexdigest()==digest,'EXACT_SAME_FD_WEIGHT_OR_MEDIA_DIGEST')
        stream.seek(0)
        result=decoder(stream)
        need(identity(os.fstat(stream.fileno()))==identity(before)==identity(path.lstat()),'EXACT_FILE_CHANGED_ON_LOAD')
    return result


def normalize(name):
    return name.replace('.lora_A.default.','.lora_A.').replace('.lora_B.default.','.lora_B.')


def load_arm(arm, source, output, checkpoint_raw, structural_raw, torch, fingerprint, check):
    """Verify the historical pins, then create an exclusive adapter dir for AdapterRequest."""
    from safetensors.torch import load, save_file

    need(sha(structural_raw)==STRUCTURAL,'EXACT_ORIGINAL_GEMMA476_LAYOUT')
    layout=document(structural_raw)
    descriptor=checkpoint_binding(arm,checkpoint_raw)
    source=Path(source)
    output=Path(output)
    conf=file_stamp(source/'adapter_config.json',descriptor['config'],check)
    cfg=document(conf)
    need(cfg['r']==16 and cfg['lora_alpha']==32 and cfg['lora_dropout']==.05 and cfg['bias']=='none',
         'ORIGINAL_GEMMA_LORA_RECIPE')
    if descriptor['format']=='target_initial_pt':
        payload=file_stamp(source/'target-fresh-initial.pt',descriptor['weight'],check,
                           lambda stream:torch.load(stream,map_location='cpu',weights_only=True))
        need(set(payload)=={'schema','seed','named_SHA256','parameters'} and payload['schema']=='GEMMA_TARGET_FRESH_INITIAL_V1'
             and payload['seed']==42 and payload['named_SHA256']==fingerprint(payload['parameters'])['SHA256']==CP[arm],
             'ACTUAL_851_NAMED_INITIAL_REQUIRED_NO_BASE_SUBSTITUTION')
        raw=payload['parameters']
    else:
        raw=file_stamp(source/'adapter_model.safetensors',descriptor['weight'],check,lambda stream:load(stream.read()))
    state={normalize(k):v for k,v in raw.items()}
    expected={normalize(r['name']):r for r in layout['trainable_manifest']}
    need(len(raw)==len(state)==len(expected)==476 and set(state)==set(expected)
         and sum(meta['numel'] for meta in expected.values())==29802496,'ORIGINAL_ADAPTER_476_COHORT_REQUIRED')
    for key,value in state.items():
        ref=expected[key]
        need(value.device.type=='cpu' and list(value.shape)==ref['shape'] and str(value.dtype)==ref['dtype']
             and bool(torch.isfinite(value).all()),'ADAPTER_LAYOUT_OR_FINITE_MISMATCH')
    output.mkdir(mode=0o700)
    with (output/'adapter_config.json').open('xb') as stream:stream.write(conf)
    save_file({k:v.contiguous() for k,v in state.items()},str(output/'adapter_model.safetensors'))
    return state


def live_parameter_name(name, adapter):
    result=name.replace('.lora_A.','.lora_A.'+adapter+'.').replace('.lora_B.','.lora_B.'+adapter+'.')
    need(result!=name,'ORIGINAL_LORA_PARAMETER_NAMESPACE')
    return result


@contextmanager
def observe_lora(engine_model,base, name, state, tensor_equal, value_witness):
    need(hasattr(engine_model,'get_base_model') and engine_model.get_base_model() is base
         and getattr(engine_model,'training',None) is False and name in getattr(engine_model,'peft_config',{}),
         'ACTUAL_PEFT_WRAPPER_ON_EXACT_BASE_REQUIRED')
    actual=dict(engine_model.named_parameters())
    mapping={k:live_parameter_name(k,name) for k in state}
    need(len(mapping)==476 and all(v in actual for v in mapping.values()),'ACTUAL_LIVE_ADAPTER476_REQUIRED')
    identity={k:(id(actual[v]),actual[v].data_ptr(),actual[v]._version) for k,v in mapping.items()}
    if not value_witness:
        need(all(tensor_equal(actual[bound].detach().cpu(),state[key]) for key,bound in mapping.items()),
             'ACTUAL_LIVE_ADAPTER476_VALUES_MISMATCH')
        value_witness.update(identity)
    else:
        need(value_witness==identity,'ACTUAL_LIVE_ADAPTER_IDENTITY_OR_VERSION_CHANGED')
    layer_names={v.split('.lora_',1)[0] for v in mapping.values()}
    modules=dict(engine_model.named_modules())
    need(layer_names<=modules.keys(),'ACTUAL_NATIVE_LORA_LAYER_NAMESPACE')
    called=set()
    restore=[]
    try:
        for key in sorted(layer_names):
            layer=modules[key]
            original=layer.forward
            def wrap(layer,key,original):
                @functools.wraps(original)
                def guarded(*args,**kwargs):
                    # PEFT's per-batch dispatch hooks have already run at this method boundary.
                    compare_layer_execution(layer,name,kwargs)
                    called.add(key)
                    return original(*args,**kwargs)
                return guarded
            had='forward' in vars(layer)
            restore.append((layer,had,original))
            layer.forward=wrap(layer,key,original)
        yield called
    finally:
        for layer,had,original in restore:
            if had:layer.forward=original
            else:del layer.forward


def owner_preflight(authority, arm, runtime, identity_sha, question_sha, media_sha, now):
    vendor='A100_CUDA' if arm=='A100' else 'HYGON_HIP'
    need(arm in ARMS and all(SHA.fullmatch(value) for value in (identity_sha,question_sha,media_sha)) and identity_sha==IDENTITIES
         and len(authority['request'])==32 and all(c in '0123456789abcdef' for c in authority['request'])
         and authority['arm']==arm and authority['checkpoint_SHA256']==CP[arm] and authority['vendor']==vendor
         and authority['runtime']==runtime and authority['identity_SHA256']==identity_sha
         and authority['question_projection_SHA256']==question_sha and authority['media_readback_SHA256']==media_sha
         and authority['root_single_native_arm'] is authority['independent_current_runtime'] is True
         and authority['gold_allowed'] is False and authority['authorized_start_epoch']<=now<authority['deadline_epoch']
         and authority['media_bytes_and_64_train_disjoint_verified'] is True,'FRESH_ROOT_AUTHORITY_BOUND_TO_SEPARATE_INPUT_AND_ARM')


def execute_native(engine,template,base,swift,torch,arm,adapter_dir,state,rows,media_root,check,summary):
    """Run one admitted arm with .3682 dynamic Swift routing and the formal TextVQA prompt."""
    need(arm in ARMS and len(rows)==5000 and template.template_meta.template_type=='gemma3_vision',
         'NATIVE_PREPARED_VAL5000_ONLY')
    adapter='gemma_textvqa_val5000_'+arm.lower()
    request=swift.AdapterRequest(name=adapter,path=str(adapter_dir))
    config=swift.RequestConfig(max_tokens=CAP,temperature=0,num_beams=1,n=1,stream=False,return_details=True)
    records={}
    delegate=template.generate
    activation=[]
    value_witness={}
    exhausted_reason='EXECUTION_NOT_REACHED'

    @functools.wraps(delegate)
    def audited(*args,**kwargs):
        options=kwargs['generation_config']
        need(kwargs.get('adapter_names')==[adapter] and kwargs.get('past_key_values') is None and
             not options.do_sample and options.num_beams==1 and options.max_new_tokens==CAP,
             'NATIVE_SWIFT_REQUEST_ROUTE_AND_GREEDY32_REQUIRED')
        with observe_lora(engine.model,base,adapter,state,torch.equal,value_witness) as called:
            answer=delegate(*args,**kwargs)
        need(bool(called),'NATIVE_LORA_LAYER_FORWARD_NOT_OBSERVED')
        activation.append(len(called))
        return answer

    template.generate=audited
    try:
        for qid,question,image in rows:
            try:
                check()
            except TimeoutError:
                exhausted_reason='DEADLINE'
                break
            path=Path(media_root)/image['relative_path']
            try:
                file_stamp(path,(image['bytes'],image['SHA256']),check,lambda _stream:None)
            except (OSError,ValueError):
                records[qid]=skipped(qid,'INPUT_UNAVAILABLE')
                continue
            text='<image>\n'+question+'\nAnswer the question using a short phrase.'
            row=swift.InferRequest(messages=[{'role':'user','content':text}],images=[str(path)])
            activation.clear()
            try:
                answer,=engine.infer([row],config,use_tqdm=False,adapter_request=request)
                if isinstance(answer,Exception):raise answer
                need(len(activation)==1,'EXACT_ONE_ACTIVATED_NATIVE_REQUEST_REQUIRED')
                choice,=answer.choices
                if choice.finish_reason=='length':
                    records[qid]=skipped(qid,'GENERATION_CAP')
                elif choice.finish_reason!='stop':
                    records[qid]=skipped(qid,'NATIVE_ERROR')
                elif not isinstance(choice.message.content,str) or not choice.message.content.strip():
                    records[qid]=skipped(qid,'EMPTY_RESPONSE')
                else:
                    need(len(choice.message.content)<=4096,'ACTUAL_RESPONSE_BOUNDED')
                    records[qid]=dict(question_id=qid,status='scored',answer=choice.message.content)
                summary['activated_requests']+=1
                layers=activation[0]
                summary['minimum_observed_LoRA_layers']=min(layers,summary.get('minimum_observed_LoRA_layers',layers))
            except Exception as error:
                if isinstance(error,ValueError) and str(error).startswith(('ACTUAL_','NATIVE_','EXACT_ONE_')):
                    exhausted_reason='ACTIVATION_NOT_PROVED'
                    records[qid]=skipped(qid,exhausted_reason)
                    break
                records[qid]=skipped(qid,'NATIVE_ERROR')
    finally:
        template.generate=delegate
    return records,exhausted_reason


def seal_arm(output,identity,records,fallback,arm,summary):
    rows=complete_records(identity,records,fallback)
    raw=b''.join(packed(row) for row in rows)
    with Path(output).open('xb') as file:
        file.write(raw);file.flush();os.fsync(file.fileno())
    return dict(arm=arm,checkpoint_SHA256=CP[arm],identity_SHA256=IDENTITIES,annotations_SHA256=ANN,
                scorer_MMF_SHA256=MMF,planned=5000,scored=sum(r['status']=='scored' for r in rows),
                failed=sum(r['status']=='failed' for r in rows),predictions_SHA256=sha(raw),native_activation=summary)


def enter_arm(arm, *, authority, runtime, identity_raw, questions_raw, media_raw, media_root,
              source, checkpoint_raw, structural_raw, model_manifest_raw, model_root, output, swift, torch, fingerprint):
    """Normal owner callback. Authority and inputs must be current on the nominated CUDA/HIP worker."""
    plan=identities(identity_raw)
    owner_preflight(authority,arm,runtime,sha(identity_raw),sha(questions_raw),sha(media_raw),time.time())
    rows=input_plan(plan,document(questions_raw),document(media_raw))
    need(sha(model_manifest_raw)==BASE_MANIFEST,'EXACT_ORIGINAL_SHARED_GEMMA_BASE16_MANIFEST')
    artifacts=document(model_manifest_raw)['files']
    need(len(artifacts)==16 and all(Path(name).name==name for name in artifacts),'ORIGINAL_GEMMA_BASE16_EXPECTED')
    deadline=authority['deadline_epoch']
    def check():
        if time.time()>=deadline:raise TimeoutError('NATIVE_ARM_SINGLE_EPOCH_EXPIRED')
    need(getattr(torch.version,'hip',None) if arm!='A100' else getattr(torch.version,'cuda',None),
         'ACTUAL_CUDA_OR_HIP_PROVIDER_REQUIRED')
    need(not (torch.version.cuda if arm!='A100' else torch.version.hip) and torch.cuda.device_count()==1,
         'EXACT_SINGLE_NATIVE_DEVICE_AND_VENDOR')
    expected_providers=runtime['effective_provider_SHA256']
    symbols={'get_model_processor':swift.get_model_processor,'get_template':swift.get_template,
             'TransformersEngine':swift.TransformersEngine,'AdapterRequest':swift.AdapterRequest}
    need(set(expected_providers)==set(symbols) and
         all(sha(Path(inspect.getsourcefile(fn)).read_bytes())==expected_providers[name] for name,fn in symbols.items()),
         'EXACT_AUTHORIZED_RUNTIME_SOURCE_PROVIDER')
    model_root=Path(model_root)
    for name,loc in artifacts.items():
        file_stamp(model_root/name,(loc['bytes'],loc['SHA256']),check,lambda _file:None)
    output=Path(output)
    output.mkdir(mode=0o700)
    records={};fallback='EXECUTION_NOT_REACHED';summary={'activated_requests':0,'scope':authority['request']}
    try:
        state=load_arm(arm,source,output/'adapter',checkpoint_raw,structural_raw,torch,fingerprint,check)
        model,processor=swift.get_model_processor(str(model_root),model_type='gemma3_vision',load_model=True,
                  download_model=False,device_map={'':0},torch_dtype=torch.bfloat16,attn_impl='sdpa')
        need(not hasattr(model,'peft_config') and {str(v.device) for v in model.parameters()}=={'cuda:0'},
             'EXACT_ORIGINAL_BASE_BEFORE_DYNAMIC_ADAPTER')
        model.eval()
        for v in model.parameters():v.requires_grad_(False)
        template=swift.get_template(processor,template_type='gemma3_vision',max_length=512,max_pixels=262144,
                                   padding_side='right',padding_free=False,truncation_strategy='raise')
        engine=swift.TransformersEngine(model,template=template,max_batch_size=1)
        records,fallback=execute_native(engine,template,model,swift,torch,arm,output/'adapter',state,rows,media_root,check,summary)
    except TimeoutError:
        fallback='DEADLINE'
    except (OSError,ValueError,RuntimeError,TypeError,KeyError):
        fallback='ACTIVATION_NOT_PROVED' if not records else 'NATIVE_ERROR'
    receipt=seal_arm(output/'predictions-private.jsonl',plan,records,fallback,arm,summary)
    with (output/'terminal-metadata.json').open('xb') as stream:
        stream.write(packed(receipt));stream.flush();os.fsync(stream.fileno())
    return receipt
