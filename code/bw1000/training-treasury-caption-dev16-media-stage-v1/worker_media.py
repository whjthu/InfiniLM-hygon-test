"""Job-limited exact16 media staging, max4 threads, all16 failure slots retained."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

from media_engine import acquire, remaining, require, content_dedup

G = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
ROSTER = G/'local-only/training-treasury-caption-dev16-final-tail-v1-original/selection/roster.json'
ROSTER_SHA = '96eaeb4b4bd5f52bcf6fbd39370f61fc1ed24ea22842dc77e15c69a5c1885627'
BINDING = G/'code/bw1000/training-vlm-original-assets-v1/original-bindings.json'
BINDING_SHA = '5d9e6132e937da172dd714313ba1dfdc7775e86b27fb623131fe9bbe2c1d1822'


def pinned(path, expected, cap):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, 'Unsafe frozen identity file')
    raw = path.read_bytes(); require(hashlib.sha256(raw).hexdigest() == expected, 'Frozen identity SHA differs')
    return json.loads(raw)


def main(a):
    from job_guard import current_affinity
    while not a.gate.exists():
        remaining(a.deadline, a.monotonic_end); time.sleep(.02)
    gate = json.loads(a.gate.read_bytes()); affinity = current_affinity()
    require(gate['deadline'] == a.deadline and gate['memory_limit_bytes'] == 2147483648 and gate['CPU_count'] == 2
            and gate['job_affinity_verified'] is True and affinity['CPU_count'] == 2
            and affinity['process_mask'] == gate['worker_affinity_mask'], 'Actual Job2CPU/2GiB gate differs')
    result = dict(passed=False, planned_images=16, actual_qualified_images=0, missing_or_rejected=16,
        maximum_HTTP_open_attempts=16, maximum_concurrent_requests=4, maximum_bytes_per_image=2097152,
        actual_worker_CPU_count=2, actual_worker_affinity_mask=affinity['process_mask'],
        evaluation_ready=False, native_processor='NOT_RUN', scorer_calls=0, training_updates=0,
        photographer_attribution_receipts='MISSING', full_pixel_decode=False,
        no_failure_backfill=True, retries=False, further_local_media_retry_allowed=False,
        Dataset_GO=False, training_GO=False, GPU_GO=False)
    records = {}; lock = threading.Lock()
    def save(name, value):
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        temporary = a.directory/(name+'.tmp'); temporary.write_bytes(raw); temporary.replace(a.directory/name)
    try:
        roster = pinned(ROSTER, ROSTER_SHA, 32768)
        old_rows = pinned(BINDING, BINDING_SHA, 1024**2)['common']['image_identity']
        old_hashes = {r['sha256'] for r in old_rows}
        require(len(roster) == len({r['coco_image_id'] for r in roster}) == len({r['flickr_photo_id'] for r in roster}) == 16
                and len(old_rows) == len(old_hashes) == 64, 'Frozen image denominators differ')
        media = a.directory/'media'; media.mkdir()
        # Private credits evidence never invents author names from static Flickr URLs.
        credits = [dict(coco_image_id=r['coco_image_id'], license_id=r['license_id'], license_url=r['license_url'],
            metadata_license_evidence='Frozen COCO official annotations image-license4; author not provided by roster',
            publisher_image_URL=r['coco_url'], actual_stage_URL='https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/'+r['file_name'],
            Flickr_static_source=r['flickr_static_url'], Flickr_photo_id=r['flickr_photo_id'],
            author='UNKNOWN_NOT_IN_FROZEN_METADATA', attribution_qualified=False,
            next_primary_source='https://www.flickr.com/photo.gne?id='+r['flickr_photo_id'],
            next_source_fetched=False, next_action='Retrieve actual Flickr photo page/API via separately bounded scope; verify photographer, title, CC BY2 license and retain evidence; inaccessible/mismatch remains failed attribution') for r in roster]
        save('credits-private.json', credits)
        def callback(index):
            def progress(value):
                with lock:
                    records[index] = dict(value)
                    save('media-progress.json', dict(planned_images=16, completed_attempt_records=records,
                        deadline=a.deadline, maximum_concurrent_requests=4, no_failure_backfill=True))
            return progress
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(acquire, row, media, a.deadline, a.monotonic_end, callback(i)):i for i,row in enumerate(roster)}
            for future in as_completed(futures):
                remaining(a.deadline, a.monotonic_end)
                i = futures[future]; value = future.result()
                with lock:
                    records[i] = value
        prior_content_overlap, intra_content_overlap = content_dedup([records[i] for i in range(16)], old_hashes)
        qualified = sum(r['actual_image_qualified'] is True for r in records.values())
        after = current_affinity()
        require(after['CPU_count'] == 2 and after['process_mask'] == affinity['process_mask'], 'Worker CPU affinity changed')
        result.update(passed=qualified == 16, state='MEDIA16_HEADERS_SHA_CONTENT_DEDUP_ONLY' if qualified == 16 else 'MEDIA_STAGING_FAILED_NO_BACKFILL',
            actual_qualified_images=qualified, missing_or_rejected=16-qualified, image_records=[records[i] for i in range(16)],
            actual_HTTP_open_attempts=sum(r['HTTP_open_attempts'] for r in records.values()),
            actual_HTTP_body_bytes=sum(r['bytes_read'] for r in records.values()),
            original64_content_SHA_overlap=prior_content_overlap, candidate16_duplicate_content_members=intra_content_overlap,
            actual_new_media_SHA_compared_to_original64=True, near_duplicate_detection='NOT_RUN',
            roster_SHA256=ROSTER_SHA, original64_binding_SHA256=BINDING_SHA, worker_affinity_unchanged_after_work=True)
    except BaseException as error:
        result.update(state='MEDIA_WORKER_FAILED_NO_BACKFILL', error_type=type(error).__name__,
            surviving_attempt_records=records, actual_HTTP_open_attempts=sum(r.get('HTTP_open_attempts',0) for r in records.values()),
            actual_HTTP_body_bytes=sum(r.get('bytes_read',0) for r in records.values()))
    with (a.directory/'worker-result.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('state','passed','planned_images','actual_qualified_images','missing_or_rejected','Dataset_GO','training_GO','GPU_GO')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True); p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True); p.add_argument('--monotonic-end', type=float, required=True)
    raise SystemExit(main(p.parse_args()))
