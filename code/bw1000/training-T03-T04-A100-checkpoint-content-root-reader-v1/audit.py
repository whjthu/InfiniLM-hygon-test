"""Audit one saved JSON. Never access checkpoint files, Torch or pickle."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_value(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, 'SHA256_ENCODING')
    return value


def stable(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def load_original(scope):
    require(time.time() < READ_DEADLINE, 'READ_DEADLINE')
    path = Path(scope['original_outcome'])
    require(path.is_absolute() and not any(p.is_symlink() for p in [path, *path.parents]), 'ORIGINAL_JSON_PATH')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                and 0 < before.st_size <= 4 * 1024**2, 'ORIGINAL_JSON_SIZE_UID')
        with os.fdopen(fd, 'rb', buffering=0, closefd=False) as stream:
            raw = stream.read(before.st_size + 1)
        require(len(raw) == before.st_size and stable(before) == stable(os.fstat(fd)) == stable(path.stat()),
                'ORIGINAL_JSON_CHANGED')
    finally:
        os.close(fd)
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == scope['original_outcome_SHA256'], 'ORIGINAL_OUTCOME_SHA256')
    return json.loads(raw), dict(bytes=len(raw), SHA256=digest)


def header(record, expected):
    require(type(record['header_bytes']) is int and 0 < record['header_bytes'] <= 4 * 1024**2, 'HEADER_SIZE_RECORD')
    sha_value(record['header_SHA256'])
    rows = record['tensors']
    require(isinstance(rows, list) and len(rows) == expected['tensors'], 'ADAPTER_TENSOR_COUNT')
    require(len({t['name'] for t in rows}) == len(rows), 'ADAPTER_DUPLICATE_NAME')
    total = 0; dtypes = collections.Counter()
    allowed = {'F64', 'F32', 'F16', 'BF16', 'I64', 'I32', 'I16', 'I8', 'U8', 'BOOL'}
    for row in rows:
        require(isinstance(row['name'], str) and 0 < len(row['name']) <= 512, 'ADAPTER_NAME_RECORD')
        require(row['dtype'] in allowed and isinstance(row['shape'], list) and len(row['shape']) <= 8, 'ADAPTER_SCHEMA_RECORD')
        require(all(type(x) is int and x > 0 for x in row['shape']), 'ADAPTER_DIMENSION_RECORD')
        require(row['elements'] == math.prod(row['shape']), 'ADAPTER_ELEMENT_RECORD')
        total += row['elements']; dtypes[row['dtype']] += 1
    require(total == expected['elements'] == record['parameter_elements'] and record['tensor_count'] == len(rows),
            'ADAPTER_SAME_CASE_ELEMENTS')
    require(dict(dtypes) == record['dtype_counts'] and record['counts_vs_same_case_binding_match'] is True,
            'ADAPTER_RECORDED_DTYPE_COUNTS')
    require(record['floating_values_or_finiteness_checked'] is False
            and record['exact_names_and_order_vs_model_qualified'] is False, 'ADAPTER_LIMITATION_PRESERVED')
    return dict(recorded_tensors=len(rows), recorded_elements=total, dtype_counts=dict(dtypes),
                header_SHA256=record['header_SHA256'], original_header_bytes_rehashed=False,
                names_or_floating_values_independently_qualified=False)


def archive(record, file_bytes):
    require(type(record['member_count']) is int and 0 < record['member_count'] <= 20000, 'ARCHIVE_MEMBER_COUNT')
    require(type(record['storage_member_count']) is int and 0 <= record['storage_member_count'] <= record['member_count'],
            'ARCHIVE_STORAGE_COUNT')
    require(type(record['declared_storage_bytes']) is int and 0 <= record['declared_storage_bytes'] <= file_bytes,
            'ARCHIVE_DECLARED_STORAGE_BYTES')
    metadata = record['pickle_metadata']
    sha_value(metadata['SHA256'])
    require(type(metadata['bytes']) is int and 0 < metadata['bytes'] <= min(4 * 1024**2, file_bytes), 'PICKLE_METADATA_SIZE')
    histogram = metadata['opcode_counts']
    require(isinstance(histogram, dict) and all(isinstance(k, str) and type(v) is int and v > 0 for k, v in histogram.items()),
            'PICKLE_OPCODE_HISTOGRAM')
    require(sum(histogram.values()) <= 500000 and histogram.get('STOP') == 1, 'PICKLE_OPCODE_BOUNDS')
    names = metadata['GLOBAL_names']
    require(isinstance(names, list) and len(names) <= 64
            and all(isinstance(n, str) and re.fullmatch(r'[A-Za-z0-9_\.]+ [A-Za-z0-9_]+', n) for n in names),
            'PICKLE_GLOBAL_NAME_RECORD')
    require(record['archive_storage_members_extracted'] is False and record['arbitrary_pickle_execution'] is False
            and metadata['GLOBAL_or_REDUCE_executed'] is False
            and metadata['complete_optimizer_schema_interpreted'] is False
            and metadata['tensor_shapes_or_values_decoded'] is False
            and metadata['RNG_restoration_qualified'] is False, 'ARCHIVE_LIMITATION_PRESERVED')
    markers = metadata['structural_keyword_markers']
    allowed = {'state', 'param_groups', 'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq',
               'python', 'numpy', 'cpu', 'cuda', 'last_epoch', '_step_count', 'base_lrs', '_last_lr'}
    require(isinstance(markers, list) and set(markers) <= allowed, 'PICKLE_MARKER_RECORD')
    return dict(member_count=record['member_count'], storage_member_count=record['storage_member_count'],
                pickle_metadata_SHA256=metadata['SHA256'], GLOBAL_names=names, keyword_markers=markers,
                independent_unpickle_or_optimizer_schema_qualification=False)


def tail(rows):
    require(isinstance(rows, list) and len(rows) <= 256, 'TAIL_ROW_COUNT')
    clean = []
    for row in rows:
        require(type(row['global_step']) is int and row['global_step'] in range(21, 26), 'TAIL_STEP')
        keys = row['field_names']; references = row['reference_fields']; losses = row['finite_loss_scalars']
        require(isinstance(keys, list) and len(keys) <= 64
                and all(isinstance(k, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', k) for k in keys), 'TAIL_KEYS')
        require(not any(any(w in k.lower() for w in ('password', 'secret', 'token', 'credential', 'environment')) for k in keys),
                'TAIL_SECRET_KEY_REJECTED')
        require(isinstance(losses, dict) and all(k in keys and 'loss' in k.lower()
                and type(v) in (int, float) and math.isfinite(v) for k, v in losses.items()), 'TAIL_LOSS_FIELDS')
        require(isinstance(references, list) and len(references) <= 32, 'TAIL_REFERENCE_FIELDS')
        for reference in references:
            require(set(reference) == {'field', 'value_type', 'nonempty'} and reference['field'] in keys
                    and reference['value_type'] in {'str', 'dict', 'list', 'tuple', 'int', 'float', 'bool', 'NoneType'}
                    and type(reference['nonempty']) is bool, 'TAIL_REFERENCE_SCHEMA')
        clean.append(dict(global_step=row['global_step'], field_names=keys,
                          finite_loss_scalars=losses, reference_fields=references))
    return clean


def audit(data, scope, expected_bindings):
    for key in ('request_id', 'epoch', 'deadline', 'publication', 'source_manifest_SHA256', 'claim_SHA256'):
        require(data[key] == scope[key], 'ORIGINAL_CONTEXT_' + key)
    require(data['deadline'] == data['epoch'] + 600 and data['epoch'] <= data['finished_epoch'] <= data['deadline'],
            'ORIGINAL_EPOCH_WINDOW')
    require(data['passed'] is True and data['GPU_GO'] is False and data['recovery_qualified'] is False, 'ORIGINAL_LIMITS')
    require(data['planned_files'] == 56 and data['planned_checkpoints'] == 4
            and data['planned_whole_file_bytes'] == scope['whole_file_bytes']
            and data['hashed_file_bytes'] == scope['whole_file_bytes'], 'ORIGINAL_BYTE_DENOMINATOR')
    require(data['explicit_read_bytes'] == scope['actual_explicit_read_bytes']
            and data['explicit_read_bytes'] <= scope['maximum_explicit_read_bytes'], 'ORIGINAL_READ_BUDGET')
    require(data['actual_affinity'] == [2, 3] and data['owner_end'] == data['owner'], 'ORIGINAL_OWNER_OR_AFFINITY')
    for key in ('wall_seconds', 'self_cpu_user_seconds', 'self_cpu_system_seconds'):
        require(type(data[key]) in (int, float) and math.isfinite(data[key]) and 0 <= data[key] <= 1200, 'ORIGINAL_COST_RECORD')
    require(type(data['self_maxrss_bytes']) is int and 0 < data['self_maxrss_bytes'] <= 512 * 1024**2, 'ORIGINAL_RSS_RECORD')
    require(data['arbitrary_pickle_executed'] is False and data['Torch_or_SDK_imported'] is False
            and data['optimizer_tensor_numeric_or_full_structure_qualified'] is False, 'ORIGINAL_QUALIFICATION_BOUNDARY')
    require(data['prior_metadata_outcome']['SHA256'] == scope['prior_metadata_outcome_SHA256'], 'PRIOR_METADATA_DEPENDENCY')
    require(len(data['cases']) == 2 and {c['case'] for c in data['cases']} == set(expected_bindings), 'CASE_DENOMINATOR')
    result = []; file_count = 0; file_bytes = 0; formats = collections.Counter()
    for planned in scope['cases']:
        case = next(c for c in data['cases'] if c['case'] == planned['case']); expected = expected_bindings[planned['case']]
        require(case['state'] == 'CONTENT_IDENTITY_COMPLETED_SCHEMA_LIMITATIONS_RETAINED', 'CASE_NOT_COMPLETED')
        authority = case['native_binding_authority']
        require(authority['SHA256'] == expected['SHA256'] and authority['bytes'] == expected['bytes']
                and authority['trainable_tensors'] == expected['tensors']
                and authority['trainable_elements'] == expected['elements']
                and authority['initial_tensor_values_reverified'] is False, 'ORIGINAL_SAME_CASE_BINDING')
        require(len(case['checkpoints']) == 2, 'CHECKPOINT_DENOMINATOR')
        cp_results = []
        for plan in planned['checkpoints']:
            cp = next(c for c in case['checkpoints'] if c['global_step'] == plan['global_step'])
            require(cp['path'] == plan['path'] and cp['full_14_byte_identities_verified'] is True
                    and set(cp['members']) == set(plan['members']) and len(cp['members']) == 14, 'FOURTEEN_MEMBERS')
            entries = {}; header_result = None; archive_results = {}
            for name, pin in plan['members'].items():
                member = cp['members'][name]; require(member['bytes'] == pin['bytes'], 'FILE_BYTE_RECORD')
                sha_value(member['SHA256']); entries[name] = dict(bytes=member['bytes'], SHA256=member['SHA256'])
                file_count += 1; file_bytes += member['bytes']
                if name.endswith('.safetensors'):
                    formats['safetensors'] += 1; header_result = header(member['format'], expected)
                elif name.endswith(('.pt', '.pth')):
                    formats['archives'] += 1; archive_results[name] = archive(member['format'], member['bytes'])
                elif name.endswith('.json'):
                    formats['JSON'] += 1
                    require(set(member) == {'bytes', 'SHA256'}, 'JSON_RECORD_NOT_OPAQUE')
                else:
                    raise ValueError('UNEXPECTED_CHECKPOINT_MEMBER')
            cp_results.append(dict(path=cp['path'], global_step=cp['global_step'], files=entries,
                                   adapter_header=header_result, archive_metadata=archive_results,
                                   actual_checkpoint_files_rehashed_by_reader=False))
        require(len(case['tail_metadata']) == 8 and {r['rank'] for r in case['tail_metadata']} == set(range(8)), 'TAIL_RANK_DENOMINATOR')
        tails = []
        for row in sorted(case['tail_metadata'], key=lambda r: r['rank']):
            sha_value(row['SHA256']); require(0 < row['bytes'] <= 524288, 'TAIL_METADATA_BYTES')
            tails.append(dict(rank=row['rank'], original_JSON_SHA256=row['SHA256'], tail_rows=tail(row['tail_rows'])))
        result.append(dict(case=planned['case'], native_binding=expected, checkpoints=cp_results,
                           original_continuous_tail_metadata=tails, original_input_rng_exact_replay_qualified=False))
    require(file_count == 56 and file_bytes == scope['whole_file_bytes']
            and dict(formats) == {'JSON': 12, 'safetensors': 4, 'archives': 40}, 'GLOBAL_RECORD_DENOMINATOR')
    return dict(saved_JSON_record_audit_passed=True, recorded_files=file_count, recorded_whole_file_bytes=file_bytes,
                recorded_formats=dict(formats), cases=result, actual_checkpoint_files_rehashed=False,
                recorded_system_costs={k: data[k] for k in ('wall_seconds', 'self_cpu_user_seconds',
                                                           'self_cpu_system_seconds', 'self_maxrss_bytes', 'explicit_read_bytes')},
                optimizer_numeric_or_full_schema_qualified=False, recovery_qualified=False, GPU_GO=False,
                authority_boundary='Independent validation of authenticated saved JSON only; original producer recorded the file hashes.')


def main():
    global READ_DEADLINE
    READ_DEADLINE = READER_CONTEXT['deadline']
    require(READER_CONTEXT['epoch'] <= time.time() < READ_DEADLINE, 'READER_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('READ_DEADLINE')))
    signal.setitimer(signal.ITIMER_REAL, READ_DEADLINE - time.time())
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    require({2, 3}.issubset(os.sched_getaffinity(0)), 'READER_CPU2_3_NOT_ALLOWED')
    os.sched_setaffinity(0, {2, 3})
    outcome, pin = load_original(SCOPE)
    result = audit(outcome, SCOPE, EXPECTED_BINDINGS)
    result.update(original_outcome_pin=pin, reader_context=READER_CONTEXT)
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
