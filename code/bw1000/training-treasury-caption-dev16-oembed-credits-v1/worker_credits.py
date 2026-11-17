"""Actual2CPU/2GiB gate, exactly16 slots, max4 keyless public JSON requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

from credits_engine import acquire,left,require

G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
ROSTER=G/'local-only/training-treasury-caption-dev16-final-tail-v1-original/selection/roster.json'
ROSTER_SHA='96eaeb4b4bd5f52bcf6fbd39370f61fc1ed24ea22842dc77e15c69a5c1885627'
MEDIA_AUDIT=G/'local-only/training-treasury-caption-dev16-v1/media-original-independent-result.json'
MEDIA_AUDIT_SHA='9becb5ef8a50f5545af1f4bd387e06bf46d5b6415bf7401e0b3aa570f195c94f'


def pinned(path,digest,cap):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'Unsafe small private evidence')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,'Frozen private evidence SHA differs')
    return json.loads(raw)


def main(a):
    from job_guard import current_affinity
    while not a.gate.exists():left(a.deadline,a.monotonic_end);time.sleep(.02)
    gate=json.loads(a.gate.read_bytes());affinity=current_affinity()
    require(gate['deadline']==a.deadline and gate['memory_limit_bytes']==2147483648 and gate['CPU_count']==2
        and gate['job_affinity_verified'] is True and affinity['CPU_count']==2
        and affinity['process_mask']==gate['worker_affinity_mask'],'Actual credits Job2CPU2GiB gate differs')
    records={};lock=threading.Lock()
    result=dict(passed=False,planned_images=16,author_title_qualified=0,missing_author_title=16,
        explicit_same_CC_BY2_license_sourceproof=0,credits_sourceproof_qualified=0,license_pending_or_conflicting=16,
        actual_worker_CPU_count=2,actual_worker_affinity_mask=affinity['process_mask'],
        maximum_concurrent_requests=4,maximum_HTTP_open_attempts=16,maximum_body_bytes_per_slot=65536,
        no_retry=True,no_backfill=True,final_failed_slots_stop=True,
        new_images_fetched=0,public_photo_pages_fetched=0,private_API_keys_read=False,
        native_processor='NOT_RUN',scorer_calls=0,training_updates=0,evaluation_ready=False,
        Dataset_GO=False,training_GO=False,GPU_GO=False)
    def progress(index):
        def update(value):
            with lock:
                records[index]=dict(value)
                temporary=a.directory/'progress.tmp'
                temporary.write_text(json.dumps(dict(planned_images=16,records=records,deadline=a.deadline),ensure_ascii=False),encoding='utf8')
                temporary.replace(a.directory/'progress.json')
        return update
    try:
        roster=pinned(ROSTER,ROSTER_SHA,32768)
        media=pinned(MEDIA_AUDIT,MEDIA_AUDIT_SHA,131072)
        require(len(roster)==len({r['flickr_photo_id'] for r in roster})==16
            and media['passed'] is True and media['producer_passed'] is True
            and media['independently_qualified_media_images']==16 and media['missing_or_rejected']==0,
            'Original16 metadata/media qualification differs')
        bodies=a.directory/'raw-json';bodies.mkdir()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(acquire,r['flickr_photo_id'],bodies/f'{i:02d}.json',a.deadline,a.monotonic_end,progress(i)):i for i,r in enumerate(roster)}
            for future in as_completed(futures):
                left(a.deadline,a.monotonic_end);value=future.result()
                with lock:records[futures[future]]=value
        author=sum(r['author_title_qualified'] is True for r in records.values())
        license_count=sum(r['license_sourceproof_qualified'] is True for r in records.values())
        credits=sum(r['credits_sourceproof_qualified'] is True for r in records.values())
        after=current_affinity()
        require(after['CPU_count']==2 and after['process_mask']==affinity['process_mask'],'Actual credits worker affinity changed')
        result.update(state='ACTUAL_OEMBED_AUTHOR_TITLE_LICENSE_SEPARATE' if author==16 else 'FIXED_SLOTS_MISSING_NO_RETRY',
            passed=author==16,author_title_qualified=author,missing_author_title=16-author,
            explicit_same_CC_BY2_license_sourceproof=license_count,credits_sourceproof_qualified=credits,
            license_pending_or_conflicting=16-license_count,
            image_records=[records[i] for i in range(16)],actual_HTTP_open_attempts=sum(r['HTTP_open_attempts'] for r in records.values()),
            actual_HTTP_body_bytes=sum(r['bytes_read'] for r in records.values()),
            roster_SHA256=ROSTER_SHA,original_media_independent_result_SHA256=MEDIA_AUDIT_SHA,
            actual_worker_affinity_unchanged=True)
    except BaseException as error:
        result.update(state='STOP_CREDITS_SCOPE_FAILURE_NO_RETRY',error_type=type(error).__name__,surviving_attempt_records=records)
    with (a.directory/'worker-result.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('state','passed','planned_images','author_title_qualified','missing_author_title','explicit_same_CC_BY2_license_sourceproof','credits_sourceproof_qualified','Dataset_GO','training_GO','GPU_GO')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--monotonic-end',type=float,required=True)
    raise SystemExit(main(p.parse_args()))
