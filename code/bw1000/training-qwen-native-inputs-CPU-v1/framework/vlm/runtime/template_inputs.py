"""Native CPU template identities before model loading or post-encode hooks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time


def settings_from_launch(launch):
    command = launch['command']
    def option(name, optional=False):
        if optional and name not in command:
            return None
        if command.count(name) != 1:
            raise ValueError('ambiguous frozen template option: ' + name)
        return command[command.index(name) + 1]
    result = dict(case=launch['case'], original_launch_SHA256=launch['a100_launch_sha256'],
                  model_type=option('--model_type'), template_type=option('--template', True),
                  max_length=int(option('--max_length')), max_pixels=int(option('--max_pixels')),
                  padding_side=option('--padding_side'), padding_free=option('--padding_free'),
                  seed=int(option('--seed')), data_seed=int(option('--data_seed')),
                  precision=option('--torch_dtype'), attention=option('--attn_impl'))
    if (result['precision'] != 'bfloat16' or result['padding_free'] != 'false'
            or result['padding_side'] != 'right' or result['max_length'] <= 0
            or result['max_pixels'] <= 0):
        raise ValueError('unsupported original CPU template configuration')
    return result


def compare_input_identities(original, actual):
    """Compare all 64 occurrences; never report a first-update qualification."""
    if len(original) != 64 or len(actual) != 64:
        raise ValueError('input comparison requires all 64 original rows')
    rows = []
    for ordinal, (a, b) in enumerate(zip(original, actual), 1):
        if a['row_ordinal'] != ordinal or b['row_ordinal'] != ordinal:
            raise ValueError('input row order differs')
        for record in (a, b):
            if not {'input_ids', 'labels'}.issubset(record['fields']):
                raise ValueError('missing complete native input or label identity')
            for value in record['fields'].values():
                digest = value['SHA256']
                if (value.get('finite') is not True or len(digest) != 64
                        or any(c not in '0123456789abcdef' for c in digest)):
                    raise ValueError('invalid native encoded field identity')
        names = set(a['fields']) | set(b['fields'])
        changed = sorted(name for name in names if a['fields'].get(name) != b['fields'].get(name))
        rows.append(dict(row_ordinal=ordinal, all_fields_exact=not changed,
                         different_fields=changed, field_denominator=len(names)))
    return dict(planned_rows=64, compared_rows=64, exact_rows=sum(r['all_fields_exact'] for r in rows),
                differing_rows=sum(not r['all_fields_exact'] for r in rows), occurrences=rows,
                all_fields_exact=all(r['all_fields_exact'] for r in rows),
                model_loaded=False, first_update_numerics_qualified=False, formal_performance=False)


def capture_native_inputs(settings, model, dataset, output, model_pins, dataset_sha, *, epoch, deadline):
    """Hash each native encoded field, then discard its private CPU value."""
    if (not all(math.isfinite(x) for x in (epoch, deadline)) or epoch <= 0
            or not 0 < deadline - epoch <= 600):
        raise ValueError('invalid input-capture epoch')
    if any(os.environ.get(k) != '-1' for k in
           ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('native CPU input capture requires hidden devices')
    initial = deadline - time.time()
    if not 0 < initial <= deadline - epoch:
        raise TimeoutError('input-capture epoch exhausted or reset')
    anchor = time.monotonic()
    def check():
        if min(deadline - time.time(), initial - (time.monotonic() - anchor)) <= 0:
            raise TimeoutError('input-capture original budget exhausted')
    model, dataset, output = Path(model), Path(dataset), Path(output)
    if (output.resolve() == model.resolve() or model.resolve() in output.resolve().parents
            or output.resolve() in model.resolve().parents):
        raise ValueError('input output overlaps model assets')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    completed = []
    def write(name, value):
        with (output / name).open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2)
    def progress(failed=False):
        p = output / 'progress.tmp'
        p.write_text(json.dumps(dict(planned_rows=64, completed_rows=len(completed), failed=failed)), encoding='utf8')
        p.replace(output / 'progress.json')
    progress()
    torch = None
    native_init = None
    try:
        if (not 0 < len(model_pins) <= 32 or sum(x['bytes'] for x in model_pins.values()) > 64 * 1024**2
                or any(name.endswith('.safetensors') or type(pin['bytes']) is not int
                       or not 0 < pin['bytes'] <= 64 * 1024**2 for name, pin in model_pins.items())):
            raise ValueError('CPU input capture accepts bounded processor assets only')
        for name, pin in model_pins.items():
            check()
            path = model / name
            if (Path(name).name != name or path.is_symlink() or path.stat().st_size != pin['bytes']
                    or hashlib.sha256(path.read_bytes()).hexdigest() != pin['SHA256']):
                raise ValueError('original processor asset identity differs')
        if dataset.is_symlink() or not 0 < dataset.stat().st_size <= 2 * 1024**2:
            raise ValueError('original dataset path or size differs')
        raw = dataset.read_bytes()
        if len(raw) > 2 * 1024**2 or hashlib.sha256(raw).hexdigest() != dataset_sha:
            raise ValueError('original dataset identity differs')
        rows = [json.loads(x) for x in raw.splitlines()]
        if len(rows) != 64:
            raise ValueError('native input capture requires all original 64 rows')
        import numpy as np
        import torch
        from .first_update_audit import fingerprint
        if torch.cuda.is_initialized():
            raise ValueError('device context was initialized before CPU capture')
        native_init = torch.cuda._lazy_init
        def forbid_device(*args, **kwargs):
            raise RuntimeError('device initialization forbidden during CPU input capture')
        torch.cuda._lazy_init = forbid_device
        torch.set_num_threads(2)
        random.seed(settings['seed'])
        np.random.seed(settings['seed'])
        torch.random.default_generator.manual_seed(settings['seed'])
        from swift.model import get_model_processor
        from swift.template import get_template
        value, processor = get_model_processor(str(model), model_type=settings['model_type'],
            load_model=False, download_model=False, device_map='cpu',
            attn_impl=settings['attention'], torch_dtype=torch.bfloat16)
        if value is not None:
            raise ValueError('CPU input capture unexpectedly constructed a model')
        template_type = settings['template_type']
        if template_type is None:
            template_type = processor.model_meta.template
        template = get_template(processor, template_type=template_type,
            max_length=settings['max_length'], max_pixels=settings['max_pixels'],
            padding_side=settings['padding_side'], padding_free=False)
        template.set_mode('train')
        for ordinal, row in enumerate(rows, 1):
            check()
            encoded = template.encode(row)
            if (not isinstance(encoded, dict) or not {'input_ids', 'labels'}.issubset(encoded)
                    or not 0 < len(encoded['input_ids']) == len(encoded['labels']) <= settings['max_length']):
                raise ValueError('native template returned incomplete or oversized inputs')
            fields = {}
            for key, item in sorted(encoded.items()):
                digest = fingerprint(item)
                # CPU reduction-order norms are not part of exact input identity.
                fields[key] = {k: v for k, v in digest.items() if k != 'floating_l2'}
            completed.append(dict(row_ordinal=ordinal, sequence_length=len(encoded['input_ids']),
                                  fields=fields))
            del encoded
            progress()
        check()
        if torch.cuda.is_initialized():
            raise ValueError('CPU input capture initialized a device context')
        write('input-identities.json', completed)
        result = dict(passed=True, rows=64, settings=settings, actual_template_type=template_type,
                      model_files_verified=len(model_pins), dataset_SHA256=dataset_sha,
                      identities_SHA256=hashlib.sha256((output / 'input-identities.json').read_bytes()).hexdigest(),
                      versions={k: importlib.metadata.version(k) for k in
                                ('torch', 'transformers', 'ms-swift', 'tokenizers', 'Pillow')},
                      epoch=epoch, deadline=deadline, wall_seconds=time.monotonic() - anchor,
                      model_loaded=False, CUDA_initialized=False, GPU_cards=0,
                      raw_encoded_payload_saved=False, first_update_numerics_qualified=False,
                      formal_performance=False)
        write('result.json', result)
        return result
    except BaseException as error:
        progress(True)
        write('failure.json', dict(error_type=type(error).__name__, completed_rows=len(completed),
              planned_rows=64, epoch=epoch, deadline=deadline, model_qualified=False))
        raise
    finally:
        if torch is not None and native_init is not None:
            torch.cuda._lazy_init = native_init
