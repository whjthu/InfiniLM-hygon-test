"""Independent local media terminal reader: no producer imports, HTTP, Torch or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import time

G = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
RUN = G/'local-only/training-treasury-caption-dev16-media-stage-v1-original'
SOURCE = G/'code/bw1000/training-treasury-caption-dev16-media-stage-v1'
ROSTER = G/'local-only/training-treasury-caption-dev16-final-tail-v1-original/selection/roster.json'
ROSTER_SHA = '96eaeb4b4bd5f52bcf6fbd39370f61fc1ed24ea22842dc77e15c69a5c1885627'
OLD_BINDING_SHA = '5d9e6132e937da172dd714313ba1dfdc7775e86b27fb623131fe9bbe2c1d1822'
SUCCESS_SHA = '47bfd69ac07f08ffae859f0abff58f3abcf2bf74fb5a00d0a42fc6e131b2230a'
SUCCESS = G/'results/bw1000/2026-11-17/training-caption-dev16-complete-member-and-private-roster-independent-terminal.json'
MAX_IMAGE_BYTES = 2097152


def require(ok, why):
    if not ok:
        raise ValueError(why)


def ident(value):
    return tuple(getattr(value,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))


def small(path, cap=131072, expected=None, parse=True):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size <= cap, 'Unsafe small evidence')
    raw = path.read_bytes(); pin = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
    require(ident(before) == ident(path.lstat()) and (expected is None or pin['SHA256'] == expected), 'Evidence changed/SHA differs')
    return (json.loads(raw) if parse else raw), pin


def file_evidence(path, row, until):
    """One full body SHA pass, then independent passive header; never Image.load or Image.verify."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and 0 <= before.st_size <= MAX_IMAGE_BYTES, 'Invalid actual media file')
    digest = hashlib.sha256(); size = 0; head = b''; end = b''
    with path.open('rb') as stream:
        while size < before.st_size:
            require(time.monotonic() < until, 'Independent local reader budget exhausted')
            chunk = stream.read(min(65536,before.st_size-size)); require(chunk, 'Actual media truncated')
            if size == 0:
                head = chunk[:2]
            end = (end+chunk)[-2:]; size += len(chunk); digest.update(chunk)
        require(stream.read(1) == b'', 'Actual media grew')
    # Record SHA/size even when JPEG qualification fails, retaining actual failure evidence.
    result = dict(bytes=size,SHA256=digest.hexdigest(), SOI=head == b'\xff\xd8', EOI=end == b'\xff\xd9',
                  full_body_SHA_passes=1, full_pixel_decode=False, passive_header_passed=False)
    try:
        from PIL import Image
        with Image.open(path) as image:
            result.update(actual_format=image.format, actual_width=image.width, actual_height=image.height)
            result['passive_header_passed'] = bool(image.format == 'JPEG' and image.size == (row['width'],row['height'])
                and 0 < image.width <= 640 and 0 < image.height <= 640 and result['SOI'] and result['EOI'])
    except Exception as error:
        result['passive_header_error_type'] = type(error).__name__
    require(ident(before) == ident(path.lstat()), 'Actual media changed during hash/header audit')
    result['stable_stat'] = True
    return result


def header_record(record, url):
    """Independent validation of persisted producer headers; no live response is fetched."""
    header = record.get('response_headers',{})
    return bool(record.get('HTTP_open_attempts') == 1 and record.get('normal_TLS') is True
        and record.get('redirects') is False and record.get('proxy_environment') is False and record.get('URL') == url
        and record.get('headers_qualified_before_body') is True and type(header.get('Content_Length')) is int
        and 0 < header['Content_Length'] <= MAX_IMAGE_BYTES and header.get('Content_Type') == 'image/jpeg'
        and isinstance(header.get('ETag'),str) and 1 <= len(header['ETag']) <= 256)


def main(a):
    require(re.fullmatch('[0-9a-f]{40}',a.publication) and re.fullmatch('[0-9a-f]{32}',a.request)
            and all(re.fullmatch('[0-9a-f]{64}',v) for v in (a.claim_sha256,a.source_mf_sha256)), 'Expected invocation identity invalid')
    until = time.monotonic()+120
    _, success_pin = small(SUCCESS,expected=SUCCESS_SHA)
    roster, roster_pin = small(ROSTER,32768,ROSTER_SHA)
    old, old_pin = small(G/'code/bw1000/training-vlm-original-assets-v1/original-bindings.json',1024**2,OLD_BINDING_SHA)
    old_hashes = {r['sha256'] for r in old['common']['image_identity']}
    require(len(roster) == len({r['coco_image_id'] for r in roster}) == len({r['flickr_photo_id'] for r in roster}) == 16
            and len(old['common']['image_identity']) == len(old_hashes) == 64, 'Frozen16/old64 identities differ')
    mf, mf_pin = small(SOURCE/'MF.json',expected=a.source_mf_sha256)
    claim, claim_pin = small(SOURCE/'CLAIM.json',expected=a.claim_sha256)
    require({p.name for p in SOURCE.iterdir() if p.is_file()} == set(mf)|{'MF.json'}, 'Actual source inventory differs')
    for name,pin in mf.items():
        require(Path(name).name == name, 'Unsafe source name')
        _, actual = small(SOURCE/name,1048576,parse=False); require(actual == pin, 'Actual source SHA differs')
    import PIL
    require(PIL.__version__ == claim['Pillow_version'], 'Actual independent passive Pillow version differs')
    for pin in claim['Pillow_passive_module_pins']:
        _, actual = small(Path(pin['path']),cap=16777216,parse=False)
        require(actual['bytes'] == pin['bytes'] and actual['SHA256'] == pin['SHA256'], 'Independent passive Pillow provider SHA differs')
    intent, intent_pin = small(RUN/'intent.json'); outcome, outcome_pin = small(RUN/'outcome.json')
    require(intent['request'] == outcome['request'] == a.request and intent['publication'] == a.publication
            and intent['claim_SHA256'] == a.claim_sha256 and intent['epoch'] == outcome['epoch'] == a.epoch
            and intent['deadline'] == outcome['deadline'] == a.epoch+600
            and intent['planned_images'] == 16 and intent['maximum_HTTP_open_attempts'] == 16
            and intent['maximum_concurrent_requests'] == 4 and intent['maximum_total_body_bytes'] == 33554432
            and intent['maximum_bytes_per_image'] == MAX_IMAGE_BYTES and intent['replay_allowed'] is False,
            'Actual original controller context differs')
    guard = outcome.get('guarded_process'); gate_pin = None
    if guard is not None:
        require(guard['deadline'] == a.epoch+600 and guard['WindowsJobObject_limits_verified'] is True
            and guard['process_commit_limit_bytes'] == guard['job_commit_limit_bytes'] == 2147483648
            and guard['job_CPU_affinity_verified'] is True and guard['worker_CPU_affinity_verified_before_gate'] is True
            and guard['worker_CPU_count'] == 2 and int(guard['worker_affinity_mask']).bit_count() == 2
            and type(guard['returncode']) is int and type(guard['timed_out']) is bool
            and 0 <= guard['actual_peak_process_commit_bytes'] <= 2147483648
            and 0 <= guard['actual_peak_job_commit_bytes'] <= 2147483648
            and guard['wall_seconds'] <= 600, 'Actual bounded Job/affinity/peak/terminal evidence differs')
        gate,gate_pin = small(RUN/'job-gate.json')
        require(gate['deadline'] == a.epoch+600 and gate['CPU_count'] == 2 and gate['memory_limit_bytes'] == 2147483648
                and gate['worker_affinity_mask'] == guard['worker_affinity_mask'] and gate['job_affinity_verified'] is True, 'Actual worker gate differs')
    else:
        require(outcome['passed'] is False and outcome.get('error_type'), 'No actual terminal guard/prelaunch failure')
    worker = outcome.get('worker'); worker_pin = None; progress_pin = None; records = {}
    if worker is not None:
        actual,worker_pin = small(RUN/'worker-result.json'); require(actual == worker, 'Embedded worker differs')
        require(guard is not None and worker['actual_worker_CPU_count'] == 2
                and worker['actual_worker_affinity_mask'] == guard['worker_affinity_mask']
                and worker['planned_images'] == 16 and worker['maximum_concurrent_requests'] == 4
                and worker['maximum_bytes_per_image'] == MAX_IMAGE_BYTES
                and worker['evaluation_ready'] is False and worker['photographer_attribution_receipts'] == 'MISSING'
                and worker['native_processor'] == 'NOT_RUN' and worker['scorer_calls'] == worker['training_updates'] == 0,
                'Worker scope/limitations differ')
        if 'image_records' in worker:
            require(len(worker['image_records']) == 16, 'Producer16 record denominator differs')
            records = {i:r for i,r in enumerate(worker['image_records'])}
        else:
            records = {int(i):r for i,r in worker.get('surviving_attempt_records',{}).items()}
    progress_path = RUN/'media-progress.json'
    if progress_path.exists():
        progress,progress_pin = small(progress_path)
        require(progress['deadline'] == a.epoch+600 and progress['planned_images'] == 16
                and progress['maximum_concurrent_requests'] == 4 and progress['no_failure_backfill'] is True, 'Progress scope differs')
        if not records:
            records = {int(i):r for i,r in progress['completed_attempt_records'].items()}
    require(all(0 <= i < 16 for i in records), 'Unexpected attempt slot')
    images = []; seen = {}; file_names = set(); total_actual_bytes = attempts = recorded_read = 0
    for i,row in enumerate(roster):
        name = row['file_name']; require(name == f"{row['coco_image_id']:012d}.jpg", 'Unsafe frozen name')
        file_names |= {name,name+'.partial'}
        record = records.get(i,{}); attempts += record.get('HTTP_open_attempts',0); recorded_read += record.get('bytes_read',0)
        require(record.get('HTTP_open_attempts',0) in (0,1) and 0 <= record.get('bytes_saved',0) <= record.get('bytes_read',0) <= MAX_IMAGE_BYTES, 'Attempt/body budget differs')
        url = 'https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/'+name
        final = RUN/'media'/name; partial = RUN/'media'/(name+'.partial')
        require(not (final.exists() and partial.exists()), 'Both final and partial exist for slot')
        value = dict(slot=i, qualified=False, persisted_header_contract=header_record(record,url),
                     producer_state=record.get('state','NO_RECORD'), HTTP_open_attempts=record.get('HTTP_open_attempts',0),
                     recorded_bytes_read=record.get('bytes_read',0),recorded_bytes_saved=record.get('bytes_saved',0))
        actual_file = final if final.exists() else partial if partial.exists() else None
        if actual_file is not None:
            evidence = file_evidence(actual_file,row,until); total_actual_bytes += evidence['bytes']; value['actual_file'] = evidence
            seen.setdefault(evidence['SHA256'],[]).append(i)
            value['original64_content_overlap'] = evidence['SHA256'] in old_hashes
            value['qualified'] = bool(final.exists() and evidence['passive_header_passed'] and value['persisted_header_contract']
                and evidence['SHA256'] == record.get('SHA256') and evidence['bytes'] == record.get('bytes_read')
                == record.get('bytes_saved') == record.get('response_headers',{}).get('Content_Length')
                and not value['original64_content_overlap'])
        images.append(value)
    actual_directory = RUN/'media'
    if actual_directory.exists():
        require(all(p.is_file() and not p.is_symlink() and p.name in file_names for p in actual_directory.iterdir()), 'Unexpected media file/path')
    for slots in seen.values():
        if len(slots) > 1:
            for i in slots:
                images[i].update(qualified=False,candidate16_content_duplicate=True)
    qualified = sum(v['qualified'] for v in images)
    require(attempts <= 16 and recorded_read <= 33554432 and total_actual_bytes <= 33554432, 'Whole16 request/byte budget exceeded')
    if worker and 'image_records' in worker:
        require(qualified == worker['actual_qualified_images'] and 16-qualified == worker['missing_or_rejected']
                and attempts == worker['actual_HTTP_open_attempts'] and recorded_read == worker['actual_HTTP_body_bytes'], 'Independent16 denominator/body/count differs from producer')
    if outcome['passed'] is True:
        require(guard is not None and guard['returncode'] == 0 and guard['timed_out'] is False and worker is not None
                and worker['passed'] is True and qualified == 16 and worker['worker_affinity_unchanged_after_work'] is True,
                'Producer success lacks independent16 terminal evidence')
    require(outcome['planned_images'] == 16 and outcome['replay_allowed'] is False
            and outcome['no_failure_backfill'] is True and outcome['further_local_media_retry_allowed'] is False
            and all(outcome[k] is False for k in ('evaluation_ready','Dataset_GO','training_GO','GPU_GO')), 'Final no-repeat/no-qualification contract differs')
    result = dict(passed=True, state='INDEPENDENT_MEDIA_TERMINAL_AUDITED_RIGHTS_NATIVE_SCORER_PENDING', request=a.request,
        source_publication=a.publication, producer_passed=outcome['passed'], planned_images=16,
        independently_qualified_media_images=qualified, missing_or_rejected=16-qualified,
        actual_HTTP_open_attempts=attempts, recorded_HTTP_body_bytes=recorded_read, actual_file_bytes=total_actual_bytes,
        original64_content_SHA_overlap=sum(v.get('original64_content_overlap',False) for v in images),
        candidate16_duplicate_members=sum(v.get('candidate16_content_duplicate',False) for v in images),
        private_ordered_slot_evidence=images, actual_guard=guard, epoch=a.epoch, deadline=a.epoch+600,
        pins=dict(success=success_pin,roster=roster_pin,old64=old_pin,source_MF=mf_pin,claim=claim_pin,
                  intent=intent_pin,outcome=outcome_pin,worker=worker_pin,job_gate=gate_pin,progress=progress_pin),
        terminal_basis='Original controller outcome after Job child wait/reap or explicit prelaunch failure; no independent PID probe',
        live_response_headers_requeried=False, headers_evidence='Persisted producer pre-body gate; not independent network observation',
        decoded_pixels=False, near_duplicate_detection='NOT_RUN', photographer_attribution='PENDING',
        native_processor='NOT_RUN', scorer_calls=0, training_updates=0, new_HTTP_requests=0,
        old_partials_opened=False, reference_files_opened=False, selection_rerun=False,
        evaluation_ready=False,Dataset_GO=False,training_GO=False,GPU_GO=False,further_local_media_retry_allowed=False)
    require(a.output.resolve().is_relative_to((G/'local-only').resolve()) and not a.output.exists(),'Output must be exclusive local-only')
    with a.output.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('passed','producer_passed','planned_images','independently_qualified_media_images','missing_or_rejected','new_HTTP_requests','Dataset_GO','training_GO','GPU_GO')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--claim-sha256',required=True)
    p.add_argument('--source-mf-sha256',required=True);p.add_argument('--request',required=True);p.add_argument('--epoch',type=float,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
