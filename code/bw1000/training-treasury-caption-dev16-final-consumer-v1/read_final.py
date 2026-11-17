"""One local read-only terminal audit. No HTTP, subprocess, imports of producer code or selection rerun."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import time
from urllib.parse import urlsplit
import zlib

G = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
RUN = G/'local-only/training-treasury-caption-dev16-final-tail-v1-original'
SOURCE = G/'code/bw1000/training-treasury-caption-dev16-final-tail-v1'
REQUEST = '011de10da36149aa9934ded287fdba27'
PUBLICATION = '3973f95839fa84278b756467995c4eb9956e8cc7'
CLAIM_SHA = '6d7b008736f49e5ad856a7782748ce2519f59645b2cc8b7cd61dd4766884d64f'
MF_SHA = '1446740736b90eb305155e1d11615674edca574036151c9a25950109c39c4f3a'
EPOCH, DEADLINE = 1794907315.2376022, 1794907915.2376022
OLD_REQUEST = 'c4c40fce350442f18365b1c1f96cc05c'
FIRST_REQUEST = '6b6f69f382b24a129a997aac0e0e4d40'
FAILURE_SHA = 'a43366712f2ecacef2669ea263248c29f3939c4e6e05fbdce6c7f245f31dbe0e'
OLD_AUDIT_SHA = 'cbc5783ea02787ff4cdc2cde784f0bab3eedfb7a8f330d3e3a4225a107fad058'
PREFIX_BYTES, TAIL_BYTES, FULL_BYTES = 17039360, 1776788, 18816148
PREFIX_SHA = 'bbb745d869992ddf398648195e143abd3df582364dfb2a183a6551a669d560b5'
ETAG = '"f4bbac642086de4f52a3fdda2de5fa2c"'
SEED = hashlib.sha256(b'Treasury-generic-caption-dev16-v1-20261117').hexdigest()
BINDING_SHA = '5d9e6132e937da172dd714313ba1dfdc7775e86b27fb623131fe9bbe2c1d1822'
PROMPT = '<image>\nDescribe this image in one short sentence.'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(s):
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def small(path, cap=32768, expected=None, parse=True):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size <= cap, 'Unsafe metadata')
    raw = path.read_bytes()
    require(identity(before) == identity(path.lstat()), 'Metadata changed while reading')
    pin = dict(bytes=len(raw), SHA256=sha(raw))
    require(expected is None or pin['SHA256'] == expected, 'Metadata SHA differs')
    return (json.loads(raw) if parse else raw), pin


def old_chain():
    first, fp = small(G/'results/bw1000/2026-11-17/training-caption-dev16-original-public-Range-timeout-and-partial.json', expected='6df8b3908e28812aba84d38d4202dd811dcd114e135cdf0f74a5d4c6881d618b')
    first_audit, ap = small(G/'local-only/training-treasury-caption-dev16-v1/known-timeout-partial-independent-result.json', expected='1c8a5d27cbb85a35dba1496c05621f64ae55b4536922a06902e5fa782f53502d')
    second, sp = small(G/'results/bw1000/2026-11-17/training-caption-dev16-first-suffix-timeout-and-verified-17MB-prefix.json', expected=FAILURE_SHA)
    second_audit, sap = small(G/'local-only/training-treasury-caption-dev16-v1/tail-original-independent-result.json', 65536, OLD_AUDIT_SHA)
    first_sha = 'fa39234196d0f6d93f22502596a237ee82461d2a01cf6b42f2a11a790c912917'
    require(first['request'] == FIRST_REQUEST and first['original_acquire_passed'] is False
            and first['original_closed'] is True and first['received_partial_SHA256'] == first_sha
            and first['received_partial_bytes'] == 8650752 and first['full_original_request_replay_allowed'] is False,
            'First published failure chain differs')
    require(first_audit['passed'] is True and first_audit['state'] == 'KNOWN_TIMEOUT_FAILURE_PARTIAL_FULL_SHA_VERIFIED'
            and first_audit['original_request'] == FIRST_REQUEST and first_audit['partial']['SHA256'] == first_sha
            and first_audit['partial']['bytes'] == 8650752 and first_audit['original_Job_returncode'] == 124,
            'First independent failure differs')
    require(isinstance(second, dict) and second_audit['passed'] is True and second_audit['acquisition_passed'] is False
            and second_audit['state'] == 'KNOWN_TERMINAL_SUFFIX_FAILURE_RETAINED' and second_audit['request'] == OLD_REQUEST
            and second_audit['actual_guard']['returncode'] == 124 and second_audit['actual_guard']['timed_out'] is True
            and second_audit['failed_assembled']['bytes'] == PREFIX_BYTES
            and second_audit['failed_assembled']['SHA256'] == PREFIX_SHA
            and second_audit['failed_assembled']['original_prefix_SHA256'] == first_sha
            and second_audit['failed_assembled']['stable_stat'] is True
            and second_audit['missing_slots'] == second_audit['planned_images'] == 16,
            'Second independent failure/cumulative prefix differs')
    return dict(first_publication='db776bc0475f5276e65da780522d6d30a0d73df7', first_published_failure=fp,
        first_independent_failure=ap, second_publication='9119889c14125372019c44112f4990444ea8642c',
        second_published_failure=sp, second_independent_failure=sap, old_partials_opened=False)


def source_pins():
    mf, mp = small(SOURCE/'MF.json', expected=MF_SHA)
    _, cp = small(SOURCE/'CLAIM.json', expected=CLAIM_SHA)
    require({p.name for p in SOURCE.iterdir() if p.is_file()} == set(mf) | {'MF.json'}, 'Published source inventory differs')
    for name, pin in mf.items():
        require(Path(name).name == name, 'Source path escapes')
        _, actual = small(SOURCE/name, cap=200000, parse=False)
        require(pin == actual, 'Published source bytes differ')
    return dict(MF=mp, claim=cp, files=len(mf)+1, publication=PUBLICATION,
                publication_identity='Root-provided published commit plus exact source pins; no Git operation in reader')


def photo(url):
    part = urlsplit(url)
    host = re.fullmatch(r'(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com|farm[0-9]+\.static\.flickr\.com', part.hostname or '')
    match = re.fullmatch(r'/(?:[0-9]+/){1,3}([0-9]+)_[0-9a-f]+(?:_[a-z0-9]+)?\.(?:jpg|jpeg|png|gif)', part.path, re.I)
    return match[1] if part.scheme in ('http', 'https') and host and match else None


def protected():
    binding, _ = small(G/'code/bw1000/training-vlm-original-assets-v1/original-bindings.json', 1024**2, BINDING_SHA)
    cache = G/'local-only/training-qwen-generic-public-holdout-v1'
    _, _ = small(cache/'coco-caption-val-member-header.bin', 30, '80e5e7f1592d23905a4d51ed6e179177e6120c303d6747840b368f9b2fede6fd', False)
    body, _ = small(cache/'coco-caption-val-member.bin', 805766, 'ba2348b66158aea7083493f3152b8a6c8f8fc5ee1e46df0a48741526941ce7f3', False)
    require(body[:33] == b'annotations/captions_val2017.json', 'Val metadata filename differs')
    decoder = zlib.decompressobj(-15)
    raw = decoder.decompress(body[61:], 3872474)
    require(len(raw) == 3872473 and decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
            and zlib.crc32(raw) == 1836330781
            and sha(raw) == 'afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9', 'Val metadata identity differs')
    images = {r['id']: r for r in json.loads(raw)['images']}
    rows = binding['common']['image_identity']; ids = set(); photos = set(); hashes = set()
    require(len(rows) == 64 and len(images) == 5000, 'Original64 denominator differs')
    for i, row in enumerate(rows):
        match = re.fullmatch(r'coco-val2017/images/([0-9]{12})\.jpg', row['relative_role'])
        require(match and row['ordered_index'] == i, 'Original binding differs')
        identifier = int(match[1]); group = photo(images[identifier]['flickr_url'])
        require(group is not None, 'Original source group unavailable')
        ids.add(identifier); photos.add(group); hashes.add(row['sha256'])
    require(len(ids) == len(hashes) == 64, 'Original64 duplicates')
    return ids, photos


def stream_member(path, progress, receipt, until):
    """A single pass over the NEW complete file, output chunks <=1MiB; no decoded JSON parse."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size == FULL_BYTES, 'New member incomplete')
    whole, prefix, tail, decoded = [hashlib.sha256() for _ in range(4)]
    offset = size = crc = 0; decoder = zlib.decompressobj(-15)
    with path.open('rb') as stream:
        header = stream.read(93)
        fields = struct.unpack('<4s5H3I2H', header[:30])
        require(fields[0] == b'PK\x03\x04' and (fields[2], fields[3], fields[6], fields[7], fields[8], fields[9], fields[10])
                == (0, 8, 3427477015, 18816055, 91865115, 35, 28)
                and header[30:65] == b'annotations/captions_train2017.json', 'Exact ZIP header differs')
        whole.update(header); prefix.update(header); offset = len(header)
        while offset < FULL_BYTES:
            require(time.monotonic() < until, 'Reader local wall budget exceeded')
            chunk = stream.read(min(256*1024, FULL_BYTES-offset)); require(chunk, 'New member truncated')
            whole.update(chunk); boundary = max(0, min(len(chunk), PREFIX_BYTES-offset))
            prefix.update(chunk[:boundary]); tail.update(chunk[boundary:]); offset += len(chunk)
            pending = chunk
            while pending:
                require(time.monotonic() < until, 'Reader local wall budget exceeded')
                output = decoder.decompress(pending, 1024**2)
                size += len(output); require(size <= 91865115, 'Decoded size exceeds bound')
                decoded.update(output); crc = zlib.crc32(output, crc)
                pending = decoder.unconsumed_tail
                require(not decoder.unused_data, 'Unexpected trailing deflate bytes')
        require(stream.read(1) == b'', 'Unexpected member suffix')
    require(identity(before) == identity(path.lstat()), 'New member changed while auditing')
    require(size == 91865115 and decoder.eof and not decoder.unconsumed_tail and crc == 3427477015, 'Deflate completeness/CRC differs')
    require(whole.hexdigest() == progress['assembled_SHA256'] == receipt['annotation_record_SHA256']
            and prefix.hexdigest() == PREFIX_SHA and tail.hexdigest() == progress['suffix_SHA256']
            and decoded.hexdigest() == receipt['annotation_decoded_SHA256'], 'Member/segment/decoded SHA differs')
    return dict(bytes=offset, SHA256=whole.hexdigest(), prefix_SHA256=prefix.hexdigest(), suffix_SHA256=tail.hexdigest(),
                decoded_bytes=size, decoded_SHA256=decoded.hexdigest(), CRC32=crc, complete_member_passes=1,
                old_prefix_opened=False, decoded_JSON_parsed=False, decoded_bytes_written=0)



def selection(worker):
    directory = RUN/'selection'; receipt, rp = small(directory/'receipt.json')
    roster, rop = small(directory/'roster.json'); refs, refp = small(directory/'references.json', 262144)
    inputs, ip = small(directory/'inputs.json')
    require(receipt['roster'] == rop and receipt['references'] == refp and receipt['inputs'] == ip
            and worker['selection'] == {k:v for k,v in receipt.items() if k != 'reference_counts'}, 'Selection pins/worker differ')
    require(len(roster) == len(inputs) == 16 and receipt['selected'] == 16 and receipt['missing_slots'] == 0
            and receipt['planned'] == 16 and receipt['missing_reference_groups'] == 0
            and receipt['selection_seed_SHA256'] == SEED and receipt['original64_binding_SHA256'] == BINDING_SHA,
            'Frozen metadata denominator differs')
    ids, photos = protected(); selected_ids = set(); groups = set(); counts = Counter(); ref_counts = []; annotations = set(); ordering = []
    for row, request in zip(roster, inputs):
        identifier = row['coco_image_id']; name = f'{identifier:012d}.jpg'; group = photo(row['flickr_static_url'])
        require(type(identifier) is int and identifier > 0 and identifier not in ids | selected_ids
                and group is not None and group == row['flickr_photo_id'] and group not in photos | groups,
                'Selected ID/source group duplicated or protected')
        selected_ids.add(identifier); groups.add(group)
        url = urlsplit(row['coco_url'])
        require(row['file_name'] == name and url.hostname == 'images.cocodataset.org' and url.path == '/train2017/'+name
                and row['license_id'] == 4 and row['license_url'] == 'https://creativecommons.org/licenses/by/2.0/', 'Selected membership/license differs')
        width, height = row['width'], row['height']
        require(type(width) is int and type(height) is int and min(width, height) > 0, 'Invalid dimensions')
        stratum = 'landscape' if width > height else 'portrait' if width < height else 'square'
        require(stratum == row['stratum'], 'Stratum differs'); counts[stratum] += 1
        ordering.append((sha(f'{SEED}:{identifier:012d}'.encode()), identifier))
        require(request == dict(id='coco:'+str(identifier), messages=[dict(role='user', content=PROMPT)], images=['media/'+name]), 'Input leaks references or differs')
        references = refs[str(identifier)]
        require(1 <= len(references) <= 10, 'Missing/excess reference group')
        ref_counts.append(len(references))
        for reference in references:
            require(set(reference) == {'annotation_id', 'caption'} and type(reference['annotation_id']) is int
                    and reference['annotation_id'] not in annotations and isinstance(reference['caption'], str)
                    and reference['caption'].strip() and len(reference['caption']) <= 4096, 'Invalid selected reference')
            annotations.add(reference['annotation_id'])
    require(set(refs) == {str(i) for i in selected_ids} and ordering == sorted(ordering)
            and dict(counts) == receipt['quotas'] == receipt['selected_by_stratum'] == dict(landscape=8, portrait=4, square=4)
            and receipt['reference_counts'] == ref_counts, 'Quota/order/reference denominator differs')
    require(receipt['images_downloaded'] == receipt['training_updates'] == 0 and receipt['native_processor'] == 'NOT_RUN'
            and receipt['CIDEr_status'] == 'NOT_RUN_ACTUAL_FROZEN_SCORER_MISSING'
            and receipt['image_content_nonoverlap_with_original64'] == 'PENDING_ACTUAL_STAGING'
            and receipt['photographer_attribution_receipts'] == 'MISSING'
            and receipt['actual_candidate_content_SHA_overlap'] == receipt['near_duplicate_content_overlap'] == 'NOT_MEASURED'
            and all(receipt[k] is False for k in ('quality_metric_emitted', 'evaluation_ready', 'scorer_qualified',
                'formal_or_blind_CIDEr', 'selection_uses_captions_or_model_outputs', 'caption_text_used_to_choose_roster'))
            and receipt['no_backfill_after_media_or_scoring_failure'] is True, 'Qualification limitations differ')
    require(not (directory/'media').exists(), 'Unexpected staged media')
    return receipt, dict(files=dict(receipt=rp, roster=rop, references=refp, inputs=ip), selected=16,
        quotas=dict(counts), reference_counts=ref_counts, references_total=sum(ref_counts),
        unique_source_groups=16, original64_ID_and_Flickr_overlap=0, complete_pool_selection_rerun=False,
        fixed_seed_and_selected_order_checked=True, original64_actual_new_image_content_overlap='NOT_MEASURED',
        raw_references_remain_private=True, repository_history_raw_payload_audit='NOT_PERFORMED')


def audit():
    until = time.monotonic()+120
    out = dict(request=REQUEST, planned_images=16, passed=False, evaluation_ready=False, training_GO=False, GPU_GO=False,
               actual_images_downloaded=0, scorer_calls=0, training_updates=0, original_scope_still_failed=True)
    out['source'] = source_pins(); out['original_failure_chain'] = old_chain()
    intent, pin = small(RUN/'intent.json'); out['intent'] = pin
    require(intent['request'] == REQUEST and intent['publication'] == PUBLICATION and intent['claim_SHA256'] == CLAIM_SHA
            and intent['epoch'] == EPOCH and intent['deadline'] == DEADLINE and DEADLINE-EPOCH == 600
            and intent['original_failed_request'] == OLD_REQUEST and intent['maximum_HTTP_open_attempts'] == 1
            and (intent['HTTP_range_start'], intent['HTTP_range_end'], intent['new_HTTP_body_budget']) == (176326406,178103193,TAIL_BYTES)
            and intent['original_verified_prefix_bytes'] == PREFIX_BYTES and intent['original_prefix_SHA256'] == PREFIX_SHA
            and intent['replay_allowed'] is False and intent['old_epoch_not_extended'] is True, 'New original intent differs')
    outcome, op = small(RUN/'outcome.json'); out['outcome'] = op
    require(outcome['request'] == REQUEST and outcome['epoch'] == EPOCH and outcome['deadline'] == DEADLINE
            and outcome['original_failed_request'] == OLD_REQUEST and outcome['original_scope_still_failed'] is True
            and outcome['HTTP_prefix_bytes_requested'] == 0 and outcome['replay_allowed'] is False
            and outcome['original_consumer_result_SHA256'] == OLD_AUDIT_SHA
            and outcome['published_failure_result_SHA256'] == FAILURE_SHA
            and outcome['old_epoch_not_extended'] is True and outcome['planned_images'] == 16
            and all(outcome[k] is False for k in ('evaluation_ready', 'training_GO', 'GPU_GO')), 'New outcome differs')
    guard = outcome['guarded_process']; out['actual_guard'] = guard
    require(guard['deadline'] == DEADLINE and guard['WindowsJobObject_limits_verified'] is True
            and guard['process_commit_limit_bytes'] == guard['job_commit_limit_bytes'] == 2147483648
            and guard['job_CPU_affinity_verified'] is True and guard['worker_CPU_affinity_verified_before_gate'] is True
            and guard['worker_CPU_count'] == 2 and int(guard['worker_affinity_mask']).bit_count() == 2
            and type(guard['returncode']) is int and type(guard['timed_out']) is bool, 'Actual terminal Job limits missing')
    gate, gp = small(RUN/'job-gate.json'); out['job_gate'] = gp
    require(gate['deadline'] == DEADLINE and gate['memory_limit_bytes'] == 2147483648 and gate['CPU_count'] == 2
            and gate['worker_affinity_mask'] == guard['worker_affinity_mask'] and gate['job_affinity_verified'] is True, 'Actual gate differs')
    out['terminal_basis'] = 'Original controller returned after known Job child wait and reap; no new PID or liveness probe'
    progress, pp = small(RUN/'tail-progress.json'); out['tail_progress'] = pp
    require(progress['deadline'] == DEADLINE and progress['range_start'] == 176326406 and progress['range_end'] == 178103193
            and progress['expected_suffix_bytes'] == TAIL_BYTES and progress['expected_ETag'] == ETAG
            and progress['preserved_original_prefix_bytes'] == PREFIX_BYTES and progress['HTTP_prefix_bytes_requested'] == 0
            and progress['HTTP_open_attempts'] == 1 and progress['normal_TLS'] is True
            and progress['retries'] is False and progress['redirects'] is False and progress['proxy_environment'] is False
            and progress['URL'] == 'https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip', 'Actual suffix progress differs')
    worker = outcome.get('worker')
    if worker is not None:
        actual, wp = small(RUN/'worker-result.json'); out['worker_result'] = wp
        require(actual == worker and worker['actual_worker_CPU_count'] == 2
                and worker['actual_worker_affinity_mask'] == guard['worker_affinity_mask']
                and worker['planned_images'] == 16 and worker['HTTP_prefix_bytes_requested'] == 0
                and all(worker[k] is False for k in ('evaluation_ready', 'training_GO', 'GPU_GO', 'quality_metric_emitted')), 'Actual worker differs')
    if outcome['passed'] is True:
        require(guard['returncode'] == 0 and guard['timed_out'] is False and worker is not None
                and worker['state'] == 'SUFFIX_ACQUIRED_COMPLETE_MEMBER_CRC_AND_METADATA_SELECTION_RECORDED'
                and worker['worker_affinity_unchanged_after_work'] is True and worker['tail'] == progress
                and progress['complete'] is True and progress['response_headers_qualified'] is True
                and progress['suffix_bytes_read'] == progress['suffix_bytes_saved'] == progress['actual_new_HTTP_body_bytes'] == TAIL_BYTES
                and progress['assembled_bytes'] == FULL_BYTES and progress['member_file'] == 'captions_train2017.local-record.bin', 'Successful producer boundaries differ')
        require(worker['local_prefix_copy'] == dict(bytes=PREFIX_BYTES, SHA256=PREFIX_SHA,
                original_not_modified=True, HTTP_prefix_bytes_requested=0)
                and worker['actual_metadata_selected'] == 16 and worker['missing_slots'] == outcome['missing_slots'] == 0,
                'Prefix copy/metadata denominator differs')
        receipt, selected = selection(worker)
        out['member'] = stream_member(RUN/progress['member_file'], progress, receipt, until)
        out.update(state='COMPLETE_MEMBER_AND_PRIVATE_METADATA_SELECTION_INDEPENDENTLY_VERIFIED', selection=selected,
                   passed=True, actual_metadata_selected=16, missing_slots=0)
    else:
        saved = progress['suffix_bytes_saved']; read = progress['suffix_bytes_read']
        require(type(saved) is int and type(read) is int and 0 <= saved <= read <= TAIL_BYTES, 'Failure suffix denominator invalid')
        require(outcome['local_network_route_state'] == 'STOP_LOCAL_NETWORK_ROUTE'
                and (guard['returncode'] != 0 or worker is None or worker['state'] == 'TAIL_WORKER_FAILED'),
                'Actual final failure STOP state differs')
        candidates = [RUN/'captions_train2017.assembled.partial', RUN/'captions_train2017.local-record.bin']
        files = [p for p in candidates if p.exists()]; require(len(files) <= 1, 'Multiple assembled artifacts')
        actual_bytes = 0
        if files:
            s = files[0].lstat(); require(stat.S_ISREG(s.st_mode) and not files[0].is_symlink(), 'Unsafe partial')
            actual_bytes = s.st_size
            require(PREFIX_BYTES <= actual_bytes <= FULL_BYTES, 'Failure file length outside original member')
        roster = RUN/'selection/roster.json'; count = 0
        if roster.exists():
            rows, rop = small(roster); require(isinstance(rows, list) and len(rows) <= 16, 'Partial roster invalid')
            count = len(rows); out['surviving_roster_pin'] = rop
        require(outcome['missing_slots'] == 16-count, 'Failure missing-slot denominator differs')
        out.update(state='STOP_LOCAL_NETWORK_ROUTE_ACTUAL_FINAL_FAILURE_RETAINED', passed=True, actual_metadata_selected=0, missing_slots=16,
                   surviving_unqualified_roster_images=count, producer_missing_slots=outcome['missing_slots'],
                   actual_suffix_bytes_read=read, actual_suffix_bytes_saved=saved, actual_assembled_file_bytes=actual_bytes,
                   decompression_performed=False, selection_rerun=False, partial_content_SHA_audited=False,
                   partial_size_minus_reported_saved=(actual_bytes-PREFIX_BYTES-saved) if files else None)
    require(outcome['final_missing_tail_scope'] is True and outcome['final_failure_stops_local_network_route'] is True
            and outcome['further_tail_continuation_allowed'] is False and outcome['full_range_replay_allowed'] is False
            and outcome['previous_two_failure_requests_retained'] == [FIRST_REQUEST, OLD_REQUEST],
            'Final STOP/no continuation contract differs')
    out['further_tail_continuation_allowed'] = False
    out['local_network_route_state'] = outcome['local_network_route_state']
    out['acquisition_passed'] = outcome['passed']
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    require(args.output.resolve().is_relative_to((G/'local-only').resolve()) and not args.output.exists(), 'Output must be new ignored local-only file')
    result = audit()
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('request','passed','state','acquisition_passed','planned_images','actual_metadata_selected','missing_slots','evaluation_ready','training_GO','GPU_GO')}))
