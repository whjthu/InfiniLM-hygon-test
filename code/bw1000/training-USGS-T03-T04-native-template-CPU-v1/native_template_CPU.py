"""One shared processor-only USGS train/dev qualifier in the existing396 framework."""
import argparse
import copy
import hashlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import random
import resource
import signal
import sys
import time

from prepare_inputs import D, canonical, digest, need, pinned, verify_source


def main(a):
    verify_source()
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    need(claim['root_GO'] is True and claim['CPU_GO'] is True, 'root publication/CPU admission required')
    need(a.case in {'T03.sft', 'T04.sft'} and a.epoch <= time.time() < a.deadline <= a.epoch + 260,
         'one original bounded CPU epoch required')
    start = time.monotonic(); remaining = a.deadline - time.time()
    def check():
        if min(a.deadline - time.time(), remaining - (time.monotonic() - start)) <= 0:
            raise TimeoutError('original CPU work epoch exhausted')
    def expired(*_):
        raise TimeoutError('original CPU work epoch exhausted')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, remaining)
    need(len(os.sched_getaffinity(0)) == 2 and not {0, 1} & os.sched_getaffinity(0), 'qualified2CPU excluding0/1 required')
    need(all(os.environ.get(k) == '-1' for k in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']),
         'all GPU devices must be hidden')
    need(not a.output.exists(), 'existing CPU namespace; never replay')
    a.output.mkdir(mode=0o700); records = []; phase = 'admission'; denied = []
    def save(name, value):
        raw = canonical(value); need(len(raw) <= 4 * 1024**2, 'private metadata output cap exceeded')
        temp = a.output / (name + '.tmp'); temp.write_bytes(raw); temp.replace(a.output / name)
    def progress(failed=False):
        save('progress.json', dict(planned_train=45, planned_dev=10, completed_train=sum(r['split'] == 'train' for r in records),
             completed_dev=sum(r['split'] == 'dev' for r in records), phase=phase, failed=failed))
    def fds():
        found = []
        for p in Path('/proc/self/fd').iterdir():
            try: target = os.readlink(p)
            except OSError: continue
            if target == '/dev/kfd' or target.startswith(('/dev/nvidia', '/dev/dri/renderD')):
                found.append(int(p.name))
        return found
    need(not fds(), 'CPU stage already holds device FDs')
    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo'}:
            denied.append('network'); raise PermissionError('offline CPU scope')
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if path.endswith(('.safetensors', '.pt', '.pth', '.bin', '.ckpt')) or path == '/dev/kfd' or path.startswith(('/dev/nvidia', '/dev/dri/renderD')):
                denied.append('weight-or-device'); raise PermissionError('weight/device access forbidden in CPU scope')
    sys.addaudithook(audit)
    result = dict(passed=False, case=a.case, planned_train=45, planned_dev=10, actual_updates=0,
                  Dataset_GO=False, training_GO=False, GPU_GO=False, Model_GO=False)
    progress()
    try:
        plan = json.loads((D / 'PLAN.json').read_bytes()); binding = plan['cases'][a.case]
        fm = json.loads((D / 'FRAMEWORK_BINDING.json').read_bytes())
        need(len(fm['files']) == 396, 'same complete396 source required')
        for name, pin in fm['files'].items():
            check(); pinned(a.framework / name, pin, 2 * 1024**2)
        original = pinned(D / binding['launch_file'], binding['launch_pin'])
        native = json.loads(pinned(a.native_binding, binding['native_binding_pin']))
        need(native['spec'] == binding['native_spec'] and native['resolved_template_type'] == binding['resolved_template'],
             'frozen original LoRA/freeze recipe differs')
        metadata = binding['model_metadata']; need(len(metadata['files']) == binding['metadata_files'], 'processor file denominator differs')
        need({p.name for p in a.model_view.iterdir()} == set(metadata['files']) and not a.model_view.is_symlink(),
             'metadata-only model view must contain exact approved13/14 files')
        for name, pin in metadata['files'].items():
            check(); pinned(a.model_view / name, pin, 16 * 1024**2)
        data = json.loads(pinned(a.data / 'data-binding-private.json', dict(bytes=a.data_binding_bytes, SHA256=a.data_binding_sha)))
        need(data['passed'] is True and data['dataset_manifest_SHA256'] == plan['dataset_manifest']['SHA256']
             and data['dataset_audit_SHA256'] == plan['dataset_audit']['SHA256']
             and data['dev_gold_read'] is False and data['blind_inputs_gold_or_media_read'] is False,
             'new dataset adapter not qualified')
        need(Path(data['target_data']) == a.data, 'staged absolute media root differs')
        for name, pin in data['media'].items():
            check(); need(PurePosixPath(name).parts[0] in {'train', 'dev'}, 'blind/foreign media forbidden')
            pinned(a.data / name, pin, 1024**2)
        split_rows = {}
        for split, count in [('train', 45), ('dev', 10)]:
            raw = pinned(a.data / (split + '-swift.jsonl'), data['files'][split + '-swift.jsonl'])
            split_rows[split] = [json.loads(x) for x in raw.splitlines()]
            need(len(split_rows[split]) == count, 'all55 input denominator required')
        sys.path.insert(0, str(a.framework))
        from tools.hygon_vlm_recipe import build_launch
        from vlm.runtime.template_inputs import settings_from_launch
        paper = build_launch(a.case, original, PurePosixPath('/unused/swift'),
            PurePosixPath(metadata['model_root']).parent, PurePosixPath(str(a.data / 'train-swift.jsonl')),
            PurePosixPath('/unused/plugins'), PurePosixPath('/unused/output'), 29921)
        settings = settings_from_launch(paper)
        need(settings['max_length'] == 4096 and settings['max_pixels'] == 262144
             and settings['model_type'] == binding['model_type'] and settings['attention'] == binding['attention'],
             'common frozen processor arguments differ')
        need({k: md.version(k) for k in plan['versions']} == plan['versions'], 'original qualified A100 runtime changed')
        for name, pin in binding['Swift_sources'].items():
            pinned(Path(md.distribution('ms-swift').locate_file(name)), pin)
        phase = 'native-import'; progress()
        import numpy as np
        import torch
        need(not torch.cuda.is_initialized(), 'CPU stage initialized CUDA')
        torch.set_num_threads(2)
        def forbid(*args, **kwargs):
            denied.append('model-load-or-device-init'); raise PermissionError('no model weights/device allowed')
        torch.cuda._lazy_init = forbid; torch.load = forbid; torch.serialization.load = forbid
        import safetensors
        import safetensors.torch
        safetensors.safe_open = forbid; safetensors.torch.safe_open = forbid; safetensors.torch.load_file = forbid
        random.seed(42); np.random.seed(42); torch.random.default_generator.manual_seed(42)
        if a.case == 'T04.sft':
            from vlm.plugins import llava_onevision
        from swift.model import get_model_processor
        from swift.template import get_template
        empty, processor = get_model_processor(str(a.model_view), model_type=binding['model_type'], load_model=False,
            download_model=False, device_map='cpu', attn_impl=binding['attention'], torch_dtype=torch.bfloat16)
        need(empty is None and processor.model_meta.template == binding['resolved_template'], 'native processor/template differs')
        template = get_template(processor, template_type=None, max_length=4096, max_pixels=262144,
            padding_side='right', padding_free=False, truncation_strategy='raise')
        from wrapper_binding import bind_encode
        policy = json.loads((D / 'WRAPPER_SOURCES.json').read_bytes())
        encoded_provider = bind_encode(template, policy, torch.inference_mode)
        leaf_path = Path(inspect.getsourcefile(template.data_collator)).resolve()
        pinned(leaf_path, binding['Swift_sources']['swift/template/base.py'])
        from vlm.runtime.first_update_audit import fingerprint
        def identities(value):
            result = {}
            for key, v in sorted(value.items()):
                pin = fingerprint(v); need(pin['finite'] is True, 'nonfinite native input')
                if isinstance(v, torch.Tensor): need(v.device.type == 'cpu', 'nonCPU preprocessing tensor')
                result[key] = {k: x for k, x in pin.items() if k != 'floating_l2'}
            return result
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        eos = tokenizer.eos_token_id; eos_ids = eos if isinstance(eos, list) else [eos]
        need(eos_ids and all(type(x) is int for x in eos_ids), 'native EOS identity missing')
        for split in ['train', 'dev']:
            template.set_mode('train' if split == 'train' else 'transformers')
            need(template.mode == ('train' if split == 'train' else 'transformers'), 'native mode differs')
            for ordinal, row in enumerate(split_rows[split]):
                check(); phase = split + ':native-encode'; progress()
                need(set(row) == {'id', 'messages', 'images'} and len(row['messages']) == (2 if split == 'train' else 1), 'supervision split differs')
                need(len(row['images']) == 1 and row['images'][0] in {str(a.data / name) for name in data['media']
                     if PurePosixPath(name).parts[:2] == (split, 'media')}, 'staged image escaped its declared split')
                encoded = template.encode(dict(messages=row['messages'], images=row['images']))
                ids = encoded['input_ids']; need(0 < len(ids) <= 4096, 'full native input exceeds4096; retain failed slot')
                encoded_ids = identities(encoded); collated = template.data_collator([copy.deepcopy(encoded)])
                actual_ids = collated['input_ids']
                need(actual_ids.device.type == 'cpu' and tuple(actual_ids.shape) == (1, len(ids))
                     and torch.equal(actual_ids[0], torch.as_tensor(ids, dtype=actual_ids.dtype)), 'native collator changed/shortened IDs')
                labels = collated.get('labels'); eos_supervised = None; first_supervised = None
                if split == 'train':
                    need(isinstance(labels, torch.Tensor) and labels.shape == actual_ids.shape, 'training labels absent or truncated')
                    mask = labels[0] != -100; need(bool(mask.any()) and bool((~mask).any()), 'training supervision/user mask absent')
                    need(torch.equal(labels[0][mask], actual_ids[0][mask]), 'native nonmasked labels differ from IDs')
                    first_supervised = int(mask.nonzero()[0]); need(first_supervised > 0, 'native prefix not masked')
                    eos_supervised = sum(int(((actual_ids[0] == token) & mask).sum()) for token in eos_ids)
                    need(eos_supervised > 0, 'native supervised EOS not established; do not append tokens')
                else:
                    need(encoded.get('labels') is None and labels is None, 'dev received training gold/labels')
                need('pixel_values' in collated and collated['pixel_values'].numel() > 0, 'native vision input absent')
                records.append(dict(split=split, ordinal=ordinal, source_id_SHA256=digest(row['id'].encode()),
                    full_length=len(ids), encoded_fields=encoded_ids, collated_fields=identities(collated),
                    native_EOS_ids=eos_ids, supervised_EOS_count=eos_supervised, first_supervised_index=first_supervised,
                    labels_None=labels is None, full_input_retained=True))
                del encoded, collated; progress()
        need(len(records) == 55 and not denied and not fds() and not torch.cuda.is_initialized(), 'CPU scope/device/denominator failed')
        save('native-identities-private.json', records)
        save('processor-provider-private.json', encoded_provider)
        result.update(passed=True, state='TRAIN45_LABEL_MASK_EOS_AND_DEV10_NATIVE_CPU_ONLY', completed_train=45, completed_dev=10,
            identities_SHA256=digest((a.output / 'native-identities-private.json').read_bytes()),
            lengths={s: [min(r['full_length'] for r in records if r['split'] == s), max(r['full_length'] for r in records if r['split'] == s)] for s in ['train', 'dev']},
            original_native_spec= binding['native_spec'], original_freeze_LoRA_binding_static_only=True,
            actual_model_freeze_or_trainable_initialization_qualified=False, model_constructed=False,
            dev_gold_read=False, blind_data_read=False, raw_processed_payload_saved=False, framework_files=396)
    except BaseException as error:
        progress(True); result.update(error_type=type(error).__name__, error=str(error)[:200], phase=phase,
                                      completed_train=sum(r['split'] == 'train' for r in records), completed_dev=sum(r['split'] == 'dev' for r in records))
    result.update(epoch=a.epoch, deadline=a.deadline, CPU_wall_seconds=time.monotonic() - start,
                  peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, denied_attempts=denied)
    save('result.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'case', 'planned_train', 'planned_dev', 'actual_updates', 'GPU_GO']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=['T03.sft', 'T04.sft'], required=True)
    for name in ['framework', 'model-view', 'native-binding', 'data', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--data-binding-sha', required=True); p.add_argument('--data-binding-bytes', type=int, required=True)
    p.add_argument('--epoch', type=float, required=True); p.add_argument('--deadline', type=float, required=True)
    raise SystemExit(main(p.parse_args()))
