"""Same original HF/Swift base, three LoRA states, frozen dev10; never reads gold.

Run only under root's qualified bounded GPU supervisor. This is an inference worker,
not a launcher, training loop, new CPU qualification or Slurm submission.
"""
import argparse
import copy
import functools
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time

from effect_contract import ARMS, CAP, canonical, dump, eos_observation, load_dev, need, pinned, require_ready, sha

D = Path(__file__).resolve().parent


def main(a):
    plan = json.loads((D / 'PLAN.json').read_bytes())
    require_ready(plan)  # Unknown generic25 stops before import, encoding, load or device access.
    for name, pin in plan['public_prerequisites'].items():
        pinned(D / 'prerequisites' / name, pin)
    need(a.deadline > time.time() and a.deadline - a.epoch <= 900, 'ONE_ORIGINAL_EPOCH_REQUIRED')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('EVAL_DEADLINE')))
    signal.setitimer(signal.ITIMER_REAL, max(.001, a.deadline - time.time()))
    def check():
        need(time.time() < a.deadline, 'EVAL_DEADLINE_EXPIRED')
    admission = json.loads(pinned(a.admission, dict(bytes=a.admission_bytes, SHA256=a.admission_sha)))
    need(admission['passed'] is True and admission['epoch'] == a.epoch and admission['deadline'] == a.deadline
         and admission['physical_GPU_count'] == 1 and admission['unique_owned_PCI_KFD_PID_port_checked'] is True,
         'FRESH_ROOT_OWNED_GPU_ADMISSION_REQUIRED')
    own_manifest = (D / 'SOURCE_FILES.json').read_bytes()
    need(sha(own_manifest) == admission['source_manifest_SHA256'], 'PUBLISHED_OWN_SOURCE_MANIFEST_REQUIRED')
    for name, pin in json.loads(own_manifest)['files'].items():
        check(); pinned(D / name, pin)
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    need(claim['root_GO'] is claim['GPU_GO'] is True and claim['Model_GO'] is False,
         'ROOT_SOURCE_ADOPTION_FLAGS_MUST_MATCH')
    need(str(a.framework) == plan['framework_root'], 'ACTUAL_ORIGINAL400_REQUIRED')
    framework = json.loads((D / 'FRAMEWORK_BINDING.json').read_bytes())
    need(len(framework['files']) == 400, 'FRAMEWORK400_DENOMINATOR')
    for name, pin in framework['files'].items():
        check(); pinned(a.framework / name, pin)
    need({k: md.version(k) for k in plan['versions']} == plan['versions'], 'ACTUAL_A100_VERSIONS_CHANGED')
    for pin in plan['runtime_sources']:
        check(); pinned(pin['path'], pin)
    for pin in plan['qualified_loader_providers']:
        check(); pinned(pin['canonical_path'], pin)
    for name, pin in plan['Swift_sources'].items():
        pinned(md.distribution('ms-swift').locate_file(name), pin)
    for name, pin in plan['model_metadata'].items():
        check(); pinned(Path(plan['model_root']) / name, pin, 16 * 1024**2)
    rows, refs = load_dev(plan, a.data, a.identities)
    need(str(a.data) == plan['data_root'], 'ORIGINAL_DEV_ABSOLUTE_INPUTS_REQUIRED')
    # A new exclusive private output is mandatory. No gold path is accepted by this CLI.
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    planned = dict(arms=list(ARMS), dev_denominator_each=10, input_SHA256=plan['dev_inputs']['SHA256'],
                   generation_cap=CAP, epoch=a.epoch, deadline=a.deadline,
                   generic25=plan['generic25'], domain_original_training_accepted=False,
                   dev_gold_read=False, blind_inputs_gold_or_media_read=False,
                   actual_updates=0, Model_GO=False, formal17=False)
    dump(a.output / 'generation-plan.json', planned)
    sys.path.insert(0, str(a.framework))
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from safetensors.torch import save_file
    from vlm.runtime.cuda_resume_witness import trusted_load, tree_fingerprint
    from vlm.runtime.initial_trainables import named_parameter_sha256
    from vlm.runtime.first_update_audit import fingerprint
    from vlm.plugins import llava_onevision  # register the same common-framework model loader
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.infer_engine import AdapterRequest, InferRequest, RequestConfig, TransformersEngine
    from wrapper_binding import bind_encode
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'ONE_VISIBLE_A100_REQUIRED')
    torch.cuda.set_device(0)
    need('A100' in torch.cuda.get_device_properties(0).name, 'ACTUAL_A100_HARDWARE_REQUIRED')
    torch.manual_seed(42)
    manifest = json.loads(pinned(D / 'T04-native-binding.json', plan['native_binding']))['trainable_manifest']
    expected = {r['name']: r for r in manifest}
    need(len(expected) == 504 and sum(r['numel'] for r in manifest) == 43646976, 'ORIGINAL504_PARAMETER_SCOPE')
    normalized = {name.replace('.default.', '.'): row for name, row in expected.items()}
    need(len(normalized) == 504, 'UNAMBIGUOUS_STATE_KEY_NORMALIZATION')
    init = plan['initial']
    payload = trusted_load(str(Path(init['path']).parent), 'initial.pt', init, check)
    need(set(payload) == {'parameters', 'RNG', 'manifest', 'global_step'} and payload['global_step'] == 0
         and payload['RNG']['restored'] is False, 'QUALIFIED_ORIGINAL_INITIAL_PAYLOAD')
    initial = payload['parameters']
    need(set(initial) == set(expected) and named_parameter_sha256(list(initial.items())) == plan['initial_named_SHA256'],
         'SAME_ORIGINAL_INITIAL504_REQUIRED')
    states = {'initial': {k.replace('.default.', '.'): v for k, v in initial.items()}}
    adapter_dirs = {}
    for arm in ARMS:
        check()
        binding = plan['domain25'] if arm == 'domain25' else plan['generic25'] if arm == 'generic25' else None
        if binding:
            states[arm] = trusted_load(binding['root'], 'adapter_model.safetensors', binding['files']['adapter_model.safetensors'], check)
            config = pinned(Path(binding['root']) / 'adapter_config.json', binding['files']['adapter_config.json'])
            adapter_dirs[arm] = Path(binding['root'])
        else:
            config = pinned(Path(plan['domain25']['root']) / 'adapter_config.json', plan['domain25']['files']['adapter_config.json'])
            root = a.output / 'derived-initial504'; root.mkdir(mode=0o700)
            (root / 'adapter_config.json').write_bytes(config)
            save_file({k: v.contiguous() for k, v in states[arm].items()}, str(root / 'adapter_model.safetensors'))
            adapter_dirs[arm] = root
        need(set(states[arm]) == set(normalized), 'EXACT504_ADAPTER_KEYS')
        for name, value in states[arm].items():
            pin = normalized[name]
            need(value.device.type == 'cpu' and list(value.shape) == pin['shape'] and str(value.dtype) == pin['dtype']
                 and bool(torch.isfinite(value).all()), 'FINITE_CANONICAL_STATE_SHAPE_DTYPE')
        if binding:
            need(tree_fingerprint(states[arm]) == binding['adapter_tree_SHA256'], 'TYPED_SAVED_ADAPTER_STATE_CHANGED')
    load_start = time.monotonic()
    model, processor = get_model_processor(plan['model_root'], model_type='llava_onevision1_5', load_model=True,
        download_model=False, device_map={'': 0}, attn_impl='eager', torch_dtype=torch.bfloat16)
    need(not hasattr(model, 'peft_config') and {str(p.device) for p in model.parameters()} == {'cuda:0'}, 'SINGLE_BASE_GPU_LOAD')
    template = get_template(processor, template_type=None, max_length=4096, max_pixels=262144,
        padding_side='right', padding_free=False, truncation_strategy='raise')
    need(template.template_meta.template_type == 'llava_onevision1_5', 'ORIGINAL_NATIVE_TEMPLATE')
    bind_encode(template, json.loads((D / 'WRAPPER_SOURCES.json').read_bytes()), torch.inference_mode)
    need(sha(Path(inspect.getsourcefile(template.data_collator)).read_bytes()) == plan['template_base_SHA256'], 'ORIGINAL_COLLATOR_SOURCE')
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    config = RequestConfig(max_tokens=CAP, temperature=0, num_beams=1, n=1, stream=False, return_details=True, stop=[])
    effective = engine._prepare_generation_config(config)
    need(not effective.do_sample and effective.max_new_tokens == CAP and effective.num_beams == 1,
         'FROZEN_GREEDY_GENERATION_REQUIRED')
    model_load_seconds = time.monotonic() - load_start
    current = {}
    collate = template.data_collator; get_ids = template.get_generate_ids; generate = template.generate; encode = template.encode
    def fields(value):
        return {key: {k: v for k, v in fingerprint(item).items() if k != 'floating_l2'} for key, item in sorted(value.items())}
    @functools.wraps(encode)
    def observed_encode(*args, **kwargs):
        started = time.monotonic()
        try:
            return encode(*args, **kwargs)
        finally:
            current['encode_seconds'] += time.monotonic() - started
            current['encode_calls'] += 1
    @functools.wraps(collate)
    def observed_collator(*args, **kwargs):
        started = time.monotonic(); value = collate(*args, **kwargs)
        current['preprocess_collator_seconds'] += time.monotonic() - started
        current['collator_calls'] += 1
        need(fields(value) == current['ref']['collated_fields'] and value.get('labels') is None,
             'SAME_NATIVE_DEV_FULL_INPUT_IDENTITY_REQUIRED')
        return value
    @functools.wraps(get_ids)
    def observed_ids(*args, **kwargs):
        value = get_ids(*args, **kwargs)
        need(current['raw_ids'] is None and value.ndim == 2 and value.shape[0] == 1, 'ONE_NATIVE_GENERATION_RESULT')
        current['raw_ids'] = value[0].detach().cpu().tolist()
        return value
    @functools.wraps(generate)
    def observed_generate(*args, **kwargs):
        generation = kwargs['generation_config']
        need(kwargs.get('adapter_names') == [current['adapter_name']] and not generation.do_sample
             and generation.max_new_tokens == CAP and generation.num_beams == 1, 'NATIVE_ADAPTER_ROUTE_OR_GENERATION_CHANGED')
        eos = generation.eos_token_id
        current['eos_ids'] = eos if isinstance(eos, list) else [eos]
        current['generate_calls'] += 1
        return generate(*args, **kwargs)
    template.encode = observed_encode; template.data_collator = observed_collator
    template.get_generate_ids = observed_ids; template.generate = observed_generate
    generation_receipt = dict(**planned, passed=False, model_loads=1, model_load_seconds=model_load_seconds,
        arm_results={arm: dict(planned=10, attempted=0, completed=0, missing=10, state='NOT_ATTEMPTED') for arm in ARMS},
        all_predictions_finalized_before_gold=False)
    try:
        for arm in ARMS:
            check(); count = 0; adapter_name = 'usgs_' + arm
            adapter = AdapterRequest(name=adapter_name, path=str(adapter_dirs[arm]))
            path = a.output / (arm + '-predictions-private.jsonl')
            with path.open('x', encoding='utf8', newline='\n') as stream:
                for ordinal, (row, ref) in enumerate(zip(rows, refs)):
                    check(); current.clear(); current.update(ref=ref, raw_ids=None, eos_ids=None, adapter_name=adapter_name,
                        collator_calls=0, generate_calls=0, preprocess_collator_seconds=0., encode_calls=0, encode_seconds=0.)
                    generation_receipt['arm_results'][arm].update(attempted=ordinal+1, state='INFER_ATTEMPTED')
                    started = time.monotonic()
                    result, = engine.infer([InferRequest(messages=copy.deepcopy(row['messages']), images=list(row['images']))],
                                           config, use_tqdm=False, adapter_request=adapter)
                    choice, = result.choices
                    # Preserve a real returned call before any post-inference admission assertion.
                    dump(a.output / (arm + '-native-return-%02d.json' % ordinal), dict(
                        id=row['id'], arm=arm, ordinal=ordinal, raw_text=choice.message.content,
                        raw_generated_token_ids=current['raw_ids'], effective_EOS_ids=current['eos_ids'],
                        returned_token_ids=choice.token_ids, finish_reason=choice.finish_reason,
                        collator_calls=current['collator_calls'], generate_calls=current['generate_calls'],
                        actual_native_infer_returned=True, qualified_prediction=False))
                    need(current['encode_calls'] == current['collator_calls'] == current['generate_calls'] == 1, 'ONE_OBSERVED_FULL_NATIVE_REQUEST')
                    observation = eos_observation(current['raw_ids'], current['eos_ids'])
                    need(choice.token_ids == [x for x in current['raw_ids'] if x != engine.tokenizer.pad_token_id],
                         'NATIVE_RETURN_DETAILS_TOKEN_PROVENANCE')
                    need(engine.model.get_base_model() is model, 'THREE_ADAPTERS_MUST_SHARE_ONE_BASE_OBJECT')
                    if ordinal == 0:
                        actual = dict(engine.model.named_parameters())
                        for name, value in states[arm].items():
                            native_name = name.replace('.lora_A.', '.lora_A.' + adapter_name + '.').replace('.lora_B.', '.lora_B.' + adapter_name + '.')
                            need(native_name in actual and torch.equal(actual[native_name].detach().cpu(), value),
                                 'ACTUAL_SELECTED504_STATE_NOT_ORIGINAL_SAVED_TENSORS')
                    record = dict(id=row['id'], status='COMPLETED', text=choice.message.content,
                        ordinal=ordinal, arm=arm, finish_reason=choice.finish_reason, **observation,
                        prompt_tokens=result.usage.prompt_tokens, raw_generated_token_ids=current['raw_ids'],
                        raw_EOS_ids=current['eos_ids'], native_field_identity_verified=True,
                        request_seconds=time.monotonic() - started,
                        encode_seconds=current['encode_seconds'], collator_seconds=current['preprocess_collator_seconds'])
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush(); count += 1
                    generation_receipt['arm_results'][arm].update(completed=count, missing=10-count)
                    dump(a.output / (arm + '-progress-%02d.json' % count), dict(planned=10, completed=count, missing=10-count))
            generation_receipt['arm_results'][arm].update(state='COMPLETED', rows=count, file=path.name,
                bytes=path.stat().st_size, SHA256=sha(path.read_bytes()))
        check(); again_rows, again_refs = load_dev(plan, a.data, a.identities)
        need(again_rows == rows and again_refs == refs, 'ORIGINAL_USER_IMAGE_AND_NATIVE_IDENTITIES_CHANGED')
        generation_receipt.update(passed=True, all_predictions_finalized_before_gold=True)
    except BaseException as error:
        generation_receipt.update(error_type=type(error).__name__, phase='native_generation', error=str(error)[:500])
        raise
    finally:
        template.encode = encode; template.data_collator = collate
        template.get_generate_ids = get_ids; template.generate = generate
        for arm in ARMS:
            path = a.output / (arm + '-predictions-private.jsonl')
            if path.exists():
                generation_receipt['arm_results'][arm].update(file=path.name, bytes=path.stat().st_size, SHA256=sha(path.read_bytes()))
        generation_receipt['all_prediction_files_closed_before_gold'] = True
        dump(a.output / 'generation-finalized.json', generation_receipt)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('framework', 'data', 'identities', 'admission', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--admission-bytes', type=int, required=True); p.add_argument('--admission-sha', required=True)
    p.add_argument('--epoch', required=True, type=float); p.add_argument('--deadline', required=True, type=float)
    raise SystemExit(main(p.parse_args()))
