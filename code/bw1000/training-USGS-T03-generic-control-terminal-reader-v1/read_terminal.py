"""Independent A100 metadata audit. Inject READ_CLAIM/READ_ORIGIN/READ_REQUEST on stdin."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import time


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def run(claim, origin, read_request):
    need(claim['root_GO'] is True and claim['READ_GO'] is True and claim['CPU_GO'] is True,
         'ROOT_READER_ADOPTION_REQUIRED')
    need(len(read_request) == 32 and not set(read_request) - set('0123456789abcdef'), 'READER_REQUEST')
    need(type(origin) in (int, float) and math.isfinite(origin) and 0 < origin <= time.time() < origin + 45,
         'READ_ORIGIN_EXHAUSTED')
    end = time.monotonic() + origin + 45 - time.time()
    def check():
        need(time.time() < origin + 45 and time.monotonic() < end, 'READ_ORIGIN_EXHAUSTED')
    def expired(*_):
        raise TimeoutError('READ_ORIGIN_EXHAUSTED')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, origin + 45 - time.time())
    available = sorted(os.sched_getaffinity(0) - {0, 1})
    need(len(available) >= 2, 'CPU_AFFINITY_UNAVAILABLE')
    os.sched_setaffinity(0, set(available[:2]))
    need(os.sched_getaffinity(0) == set(available[:2]), 'ACTUAL_TWO_CPU_REQUIRED')
    ctx = claim['original']; scope = Path(ctx['scope']); pins = {}; total = 0
    def load(name):
        nonlocal total
        check(); p = scope / name; before = p.lstat()
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 262144, 'BOUNDED_REGULAR_METADATA')
        with p.open('rb') as f:
            raw = f.read(262145); after = os.fstat(f.fileno())
        key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(key(before) == key(after) == key(p.lstat()) and len(raw) == before.st_size, 'METADATA_CHANGED')
        total += len(raw); need(total <= 2 * 1024**2, 'TOTAL_METADATA_BOUND')
        pins[name] = dict(bytes=len(raw), SHA256=sha(raw))
        return json.loads(raw)
    # Original package is checked, never executed/imported. Its members are code and small identity metadata only.
    mf = load('src/SOURCE_FILES.json')
    need(pins['src/SOURCE_FILES.json']['SHA256'] == ctx['source_MF_SHA256'], 'ORIGINAL_SOURCE_MF')
    need(len(mf['files']) == 21, 'EXACT_ORIGINAL_22_SOURCE_FILES')
    source = {}
    for name, pin in mf['files'].items():
        need(Path(name).name == name and name not in ('.', '..'), 'SOURCE_PATH')
        # Python/Markdown members are hashed as bytes without JSON parsing.
        check(); p = scope / 'src' / name; before = p.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes'] <= 262144, 'SOURCE_BYTES_BOUND')
        with p.open('rb') as f:
            raw = f.read(262145); after = os.fstat(f.fileno())
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'SOURCE_CHANGED')
        need(len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'ORIGINAL_SOURCE_SHA')
        total += len(raw); need(total <= 2 * 1024**2, 'TOTAL_METADATA_BOUND')
        pins['src/' + name] = dict(bytes=len(raw), SHA256=sha(raw))
        if name in ('CLAIM.json', 'PLAN.json', 'WRAPPER_SOURCES.json'):
            source[name] = json.loads(raw)
    need(pins['src/CLAIM.json']['SHA256'] == ctx['claim_SHA256'], 'ORIGINAL_CLAIM_SHA')
    original_claim = source['CLAIM.json']; plan = source['PLAN.json']
    need(original_claim['root_GO'] is True and original_claim['CPU_GO'] is True and
         original_claim['GPU_GO'] is False and original_claim['training_GO'] is False, 'CPU_ONLY_SCOPE')
    need(set(plan['cases']) == {'T03.sft'} and plan['roster_SHA256'] == claim['expected_roster_SHA256'],
         'T03_FIXED45_SCOPE')
    values = {name: load(name) for name in claim['metadata_files']}
    need(len(values) == 16, 'EXACT16_METADATA')
    install = values['install-intent.json']; intent = values['run/controller/intent.json']
    out = values['run/controller/outcome.json']; post = values['run/controller/post.json']
    for row in (install, intent, out):
        need(row['case'] == 'T03.sft' and row['request_id'] == ctx['request'] and
             row['publication'] == ctx['publication'] and row['epoch'] == ctx['epoch'] and
             row['deadline'] == ctx['deadline'] == ctx['epoch'] + 300, 'ORIGINAL_CONTEXT')
    need(install['source_manifest_SHA256'] == intent['source_manifest_SHA256'] == ctx['source_MF_SHA256']
         and install['claim_SHA256'] == ctx['claim_SHA256'], 'ORIGINAL_INSTALL_SOURCE_BINDING')
    owner = out['owner']
    need(owner == intent['owner'] and owner['uid'] == install['UID'] == os.getuid() and
         owner['host'] == install['host'] == os.uname().nodename and
         owner['boot_id'] == install['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'ORIGINAL_OWNER_SAME_HOST_BOOT')
    def absent(saved):
        need(type(saved['PID']) is int and type(saved['startticks']) is int and saved['PID'] > 0
             and saved['startticks'] > 0, 'STRONG_PROCESS_IDENTITY')
        try:
            fields = (Path('/proc') / str(saved['PID']) / 'stat').read_text().rsplit(') ', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            return 'ORIGINAL_PID_ABSENT'
        need(int(fields[19]) != saved['startticks'], 'ORIGINAL_IDENTITY_STILL_ALIVE')
        return 'ORIGINAL_IDENTITY_ABSENT_PID_REUSED'
    owner_state = absent(owner)
    child = values['run/controller/child-owner.json']; anchor = values['run/controller/held-child-anchor.json']
    need(anchor['supervisor'] == owner and anchor['request_id'] == ctx['request'] and
         anchor['PID'] == child['PID'] == anchor['PGID'] == child['PGID'] and
         child['PPID'] == owner['PID'] and anchor['start_new_session'] is True, 'HELD_WORKER_OWNERSHIP')
    child_state = absent(child)
    registry = values['run/controller/owned-registry.json']
    need(str(child['PID']) in registry and registry[str(child['PID'])]['startticks'] == child['startticks'],
         'ORIGINAL_WORKER_REGISTRY_BINDING')
    registry_states = [absent(saved) for saved in registry.values()]
    need(out['passed'] is True and out['worker_returncode'] == 0 and out['failure'] is None and
         out['stopped'] is False and out['cleanup_errors'] == {} and ctx['epoch'] <= out['finished'] <= ctx['deadline'],
         'KNOWN_NORMAL_CPU_TERMINAL')
    need(out['held_cleanup']['reaped'] is True and out['held_cleanup']['termination_needed'] is False and
         out['registry_cleanup']['owned_processes_remaining'] == 0 and
         out['registry_cleanup']['termination_needed'] is False and post == out['post'] and
         post['same_host'] is True and post['same_boot'] is True and post['controller_device_FDs'] == [] and
         post['owned_descendants_remaining'] == 0 and post['owned_device_FD_holders'] == {}, 'OWNED_POST_CLEANUP')
    need(len(intent['CPU_affinity']) == 2 and not ({0, 1} & set(intent['CPU_affinity'])), 'ORIGINAL_CPU_MASK')
    resources = out['resources']
    need(out['RSS_is_hard_limit'] is False and out['RSS_unknown_not_zero'] is True and
         resources['soft_limit_bytes'] == 8 * 1024**3 and resources['soft_limit_exceeded'] is False and
         resources['samples'] > 0 and resources['sampled_group_peak_RSS_bytes'] <= 8 * 1024**3,
         'SAMPLED_SOFT_RSS_WITH_UNKNOWN_SCHEMA')
    native = values['run/native/result.json']; progress = values['run/native/progress.json']
    need(pins['run/native/result.json'] == ctx['native_result'] and
         all(out['native_result'][k] == ctx['native_result'][k] for k in ('bytes', 'SHA256')), 'ACTUAL_NATIVE_PIN')
    need(native['passed'] is True and native['case'] == 'T03.sft' and native['planned_train'] ==
         native['completed_train'] == 45 and native['planned_dev'] == native['completed_dev'] == 0 and
         native['shared_dev_rows_reused'] == 10 and native['shared_dev_new_encodes'] == 0 and
         native['epoch'] == ctx['epoch'] and native['deadline'] == ctx['epoch'] + 260 and
         native['actual_updates'] == 0 and native['model_constructed'] is False and native['denied_attempts'] == []
         and native['dev_gold_read'] is False and native['blind_data_read'] is False and
         native['raw_processed_payload_saved'] is False, 'NEW45_ZERO_DEV_ZERO_MODEL')
    need(progress['planned_train'] == progress['completed_train'] == 45 and
         progress['planned_dev'] == progress['completed_dev'] == 0 and progress['failed'] is False, 'PROGRESS_45')
    rows = values['run/native/native-identities-private.json']; data = values['data/data-binding-private.json']
    prov = values['data/control-provenance-private.json']; prep = values['data/prepare-progress.json']
    need(pins['run/native/native-identities-private.json']['SHA256'] == native['identities_SHA256'] and
         pins['data/data-binding-private.json']['SHA256'] == native['data_binding_SHA256'] and
         pins['data/control-provenance-private.json']['SHA256'] == data['provenance_SHA256'], 'ACTUAL_OUTPUT_CHAIN')
    need(data['passed'] is True and data['target_data'] == str(scope / 'data') and
         data['files']['train-swift.jsonl']['rows'] == 45 and
         len(data['media']) == 45 and data['dev_gold_read'] is False and data['blind_inputs_gold_or_media_read'] is False
         and data['shared_dev_new_encodes'] == 0 and data['shared_dev_rows_reused'] == 10 and
         data['generic_roster_SHA256'] == prov['roster_SHA256'] == claim['expected_roster_SHA256'] and
         data['generic_original_dataset_SHA256'] == prov['original_JSONL_SHA256'] == plan['generic_original_dataset']['SHA256'],
         'FROZEN_DATA_PROVENANCE')
    need(prep['completed_images'] == prep['planned_images'] == 45 and prep['actual_image_bytes'] == 7321484 and
         prep['shared_dev_new_encodes'] == 0 and prep['dev_gold_read'] is False and prep['blind_data_read'] is False,
         'FULL45_STAGING_PROGRESS')
    need(len(rows) == len(prov['selected']) == len(claim['expected_selected_metadata']) == 45, 'ALL45_FULL_DENOMINATOR')
    totals = []; lengths = []; eos_count = 0; record_hashes = []
    for i, (row, provenance, expected) in enumerate(zip(rows, prov['selected'], claim['expected_selected_metadata'])):
        need(provenance['source_ordinal'] == expected['source_ordinal'] and
             provenance['source_id_SHA256'] == row['source_id_SHA256'] == expected['source_id_SHA256'] and
             provenance['image_SHA256'] == expected['image_SHA256'], 'SOURCE45_IDENTITY_SELECTION')
        need(row['ordinal'] == i and row['split'] == 'train' and row['labels_None'] is False and
             row['full_input_retained'] is True and type(row['full_length']) is int and 0 < row['full_length'] <= 4096 and
             type(row['first_supervised_index']) is int and 0 < row['first_supervised_index'] < row['full_length'] and
             type(row['supervised_EOS_count']) is int and row['supervised_EOS_count'] > 0 and
             type(row['effective_supervised_tokens']) is int and 0 < row['effective_supervised_tokens'] < row['full_length'],
             'FULL_NATIVE_MASK_EOS_TOKEN_WITNESS')
        for fields in (row['encoded_fields'], row['collated_fields']):
            need({'input_ids', 'labels', 'pixel_values'} <= set(fields) and
                 all(v['finite'] is True for v in fields.values()), 'FINITE_NATIVE_COLLATOR_FIELDS')
        need(row['native_EOS_ids'] and all(type(v) is int for v in row['native_EOS_ids']), 'NATIVE_EOS_SCHEMA')
        totals.append(row['effective_supervised_tokens']); lengths.append(row['full_length'])
        eos_count += row['supervised_EOS_count']; record_hashes.append(sha(canonical(row)))
    need(sum(totals) == native['effective_supervised_token_total'] and
         [min(totals), max(totals)] == native['effective_supervised_token_range'] and
         [min(lengths), max(lengths)] == native['lengths']['train'], 'ACTUAL_TOKEN_COUNTER_SUM')
    declaration = values['run/native/processor-declaration-private.json']; resolved = values['run/native/resolved-template-private.json']
    need(declaration['empty_is_None'] is True and declaration['raw_template'] == resolved['raw_template'] ==
         resolved['resolved_template'] == native['native_raw_template'] == native['actual_resolved_template'] == 'internvl_hf'
         and Path(declaration['model_view']).name == native['metadata_view_basename'] == 'InternVL3_5-8B-HF',
         'NATIVE_T03_TEMPLATE_BASENAME')
    provider = values['run/native/processor-provider-private.json']; policy = source['WRAPPER_SOURCES.json']
    need(provider['binding_kind'] == 'EXACT_DECORATED_NATIVE_ENCODE_CHAIN' and
         provider['closure_targets_match_wrapped'] is True and provider['invocation_preserves_decorators'] is True and
         provider['unwrapped_leaf_called'] is False and provider['retry_count'] == 3 and len(provider['wrapper_chain']) == 3,
         'NATIVE_WRAPPER_CHAIN')
    for actual, expected in zip(provider['wrapper_chain'], policy['chain']):
        expected_path = Path(expected['path']).resolve()
        need(all(Path(actual[k]).resolve() == expected_path for k in ('path', 'code_filename', 'inspect_sourcefile')) and
             actual['bytes'] == expected['bytes'] and actual['file_SHA256'] == expected['SHA256'] and
             actual['code_qualname'] == expected['code_qualname'] and actual['role'] == expected['role'], 'CLOSED_PROVIDER')
    check()
    result = dict(state='T03_GENERIC45_NATIVE_CPU_INDEPENDENT_METADATA_PASS', independent_passed=True,
        original=ctx, read_request=read_request, read_origin=origin, read_finished=time.time(), CPU_affinity=sorted(os.sched_getaffinity(0)),
        original_source_files=22, original_metadata_files=16, total_read_bytes=total, pins=pins,
        planned_train=45, completed_train=45, missing_train=0, failed_train=0, new_dev_encodes=0, shared_dev_qualified_reuse=10,
        all45_identity_provenance=True, all45_full_mask_EOS_finite_witness=True,
        effective_supervised_token_total=sum(totals), effective_supervised_token_range=[min(totals), max(totals)],
        full_length_range=[min(lengths), max(lengths)], supervised_EOS_count=eos_count,
        EOS_schema_SHA256=sha(canonical([r['native_EOS_ids'] for r in rows])), native45_records_SHA256=sha(canonical(record_hashes)),
        adapter_files=data['files'], data_binding_SHA256=native['data_binding_SHA256'], identities_SHA256=native['identities_SHA256'],
        original_controller_terminal=owner_state, original_worker_terminal=child_state, owned_postclean=True,
        original_registry_identity_absent_count=len(registry_states),
        sampled_group_RSS_peak_bytes=resources['sampled_group_peak_RSS_bytes'], RSS_samples=resources['samples'],
        RSS_unknown_coverage_samples=resources.get('RSS_unknown_coverage_samples', 0), RSS_is_hard_limit=False, RSS_unknown_not_zero=True,
        CPU_wall_seconds=native['CPU_wall_seconds'], peak_self_RSS_bytes=native['peak_self_RSS_bytes'],
        native_retry_policy_count=3, actual_native_retry_attempts=None,
        raw_caption_tokenIDs_tensor_or_media_exported=False, new_processor_calls=0, new_GPU_queries=0,
        dev_gold_read=False, blind_data_read=False, actual_updates=0, training_GO=False, GPU_GO=False, Model_GO=False,
        qualification_boundary=claim['verification_boundary'])
    encoded = canonical(result); need(len(encoded) <= 16384, 'OUTPUT_BOUND')
    signal.setitimer(signal.ITIMER_REAL, 0)
    return encoded


if __name__ == '__main__':
    print(run(READ_CLAIM, READ_ORIGIN, READ_REQUEST).decode(), end='')
