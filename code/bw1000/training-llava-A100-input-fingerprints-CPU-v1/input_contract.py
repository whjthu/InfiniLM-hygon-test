"""T04-only small schema gates, resolved before any Torch or payload read."""


def require(ok, why):
    if not ok:
        raise ValueError(why)


def select_affinity(allowed):
    require(all(type(core) is int and core >= 0 for core in allowed), 'CPU_IDS')
    selected=sorted(set(allowed)-{0,1})[:2]
    require(len(selected)==2, 'TWO_NON_STREAM_CORES_REQUIRED')
    return selected


def check_affinity(actual, expected):
    require(len(actual)==len(set(actual))==len(expected)==2 and sorted(actual)==sorted(expected)
            and not set(actual)&{0,1}, 'ACTUAL_CPU_AFFINITY_OVERLAP_OR_MISMATCH')


def receipt_files(receipt, rank, plan):
    require(type(rank) is int and 0 <= rank < 8, 'RANK')
    require(receipt['state'] == 'TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED'
            and type(receipt['rank']) is int and receipt['rank'] == rank and receipt['world_size'] == 8
            and receipt['completed_global_step'] == receipt['optimizer_updates'] == receipt['clip_calls'] == 2
            and receipt['micro_forwards'] == 8, 'T04_RECEIPT_BOUNDARY')
    records = receipt['files']
    expected = {'initial.pt'} | {'forward-%02d-%s.pt' % (n,kind) for n in range(1,9) for kind in ('input','output')} \
        | {'update-%02d-%s.pt' % (n,kind) for n in (1,2) for kind in ('preclip','postclip','before-optimizer','after-optimizer')}
    require(type(records) is list and len(records) == 25, 'RECEIPT_25')
    files = {row['name']:row for row in records}
    require(len(files) == 25 and set(files) == expected, 'EXACT_25_NAMES')
    require(all(type(row['bytes']) is int and row['bytes'] > 0 and type(row['SHA256']) is str
                and len(row['SHA256']) == 64 and not set(row['SHA256'])-set('0123456789abcdef') for row in records),
            'RECEIPT_TYPED_PINS')
    require(sum(row['bytes'] for row in records) == plan['original_capture_bytes_per_rank'][rank], 'ORIGINAL_CAPTURE_BYTES')
    return files


def native_binding(value):
    require(value['spec']['case'] == 'T04.sft'
            and value['spec']['model_class'] == 'LLaVAOneVision1_5_ForConditionalGeneration'
            and value['spec']['attention'] == 'eager' and value['spec']['training']['gradient_checkpointing'] is False
            and value['spec']['frozen_paths'] == ['model.visual'] and value['resolved_template_type'] == 'llava_onevision1_5'
            and len(value['trainable_manifest']) == 504
            and sum(row['numel'] for row in value['trainable_manifest']) == 43646976, 'ORIGINAL_T04_BINDING')


def input_schema(value, row):
    require(type(value) is dict and set(value) == {'global_step','micro_ordinal','args','kwargs','RNG_before_forward'}
            and type(value['global_step']) is int and value['global_step'] == row['global_step']
            and type(value['micro_ordinal']) is int and value['micro_ordinal'] == row['micro_ordinal']
            and isinstance(value['args'], (tuple,list)) and type(value['kwargs']) is dict, 'T04_INPUT_SCHEMA')
