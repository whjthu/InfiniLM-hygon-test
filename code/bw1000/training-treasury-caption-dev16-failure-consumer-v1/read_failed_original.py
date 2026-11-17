"""Minimal local known-failure consumer: three metadata files and streamed partial SHA."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import stat

G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
RUN=G/'local-only/training-treasury-caption-dev16-v1-acquire-select-original'
REQUEST='6b6f69f382b24a129a997aac0e0e4d40'
PUBLICATION='71ee6ebbc8116571968f037f0cb43094652b3fce'
CLAIM='2b3cc39829f17bd37c9922875500f1952a89f10ccf7d3016f1079a11aeefea0c'
EPOCH=1794904178.6605182
DEADLINE=1794904478.6605182
URL='https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
ETAG='"f4bbac642086de4f52a3fdda2de5fa2c"'
START,END,EXPECTED=159287046,178103193,18816148


def require(ok,why):
    if not ok:raise ValueError(why)


def small(path):
    st=path.lstat();require(stat.S_ISREG(st.st_mode) and 0<st.st_size<=32768,'unsafe original metadata')
    raw=path.read_bytes();require(len(raw)==st.st_size,'metadata size changed')
    value=json.loads(raw);require(isinstance(value,dict),'metadata object differs')
    return value,dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def consume():
    result=dict(state='KNOWN_FAILED_EXECUTION_PARTIAL_AUDIT_PENDING',known_terminal_failure=True,
        original_request=REQUEST,original_publication=PUBLICATION,original_claim_SHA256=CLAIM,
        planned_images=16,actual_roster_images=0,missing_images=16,actual_model_updates=0,
        Dataset_GO=False,training_GO=False,GPU_GO=False,quality_metric_emitted=False,
        original_scope_replay_allowed=False,network_calls=0,original_files_modified=False,
        decompression_or_selection_called=False,original_metadata_pins={},passed=False)
    try:
        intent,identity=small(RUN/'intent.json');result['original_metadata_pins']['intent.json']=identity
        outcome,identity=small(RUN/'outcome.json');result['original_metadata_pins']['outcome.json']=identity
        progress,identity=small(RUN/'fetch-progress.json');result['original_metadata_pins']['fetch-progress.json']=identity
        require(intent['request']==outcome['request']==REQUEST and intent['publication']==PUBLICATION
                and intent['claim_SHA256']==CLAIM,'original request/publication/claim differs')
        require(intent['epoch']==outcome['epoch']==EPOCH and intent['deadline']==outcome['deadline']==DEADLINE
                and DEADLINE==EPOCH+300,'original epoch differs')
        require(intent['maximum_invocations']==1 and intent['replay_allowed']==outcome['replay_allowed']==False,
                'original maxinvoke/no-replay differs')
        guard=outcome['guarded_process']
        require(outcome['passed'] is False and guard['returncode']==124 and guard['timed_out'] is True
                and guard['deadline']==DEADLINE and 0<guard['wall_seconds']<=300,
                'original known timeout terminal differs')
        require(guard['WindowsJobObject_limits_verified'] is True
                and guard['process_commit_limit_bytes']==guard['job_commit_limit_bytes']==2147483648
                and guard['job_CPU_affinity_verified'] is True and guard['worker_CPU_affinity_verified_before_gate'] is True
                and guard['worker_CPU_count']==2 and guard['worker_affinity_mask'].bit_count()==2,
                'actual original Job2CPU/2GiB evidence differs')
        require(outcome['worker'] is None and outcome['pending_denominator']==16
                and not (RUN/'worker-result.json').exists() and not (RUN/'selection/roster.json').exists(),
                'original worker/roster failure denominator differs')
        require(progress['URL']==URL and progress['range_start']==START and progress['range_end']==END
                and progress['expected_body_bytes']==EXPECTED and progress['deadline']==DEADLINE
                and progress['HTTP_open_attempts']==1 and progress['response_headers_qualified'] is True
                and progress['expected_ETag']==ETAG and progress['official_TLS_default_validation'] is True
                and progress['redirects_allowed']==progress['retries_allowed']==progress['proxy_environment_used']==False
                and progress['complete'] is False,'original public-fetch header/partial state differs')
        partial=RUN/'captions_train2017.local-record.partial';before=partial.lstat()
        require(stat.S_ISREG(before.st_mode) and 0<before.st_size<EXPECTED
                and not (RUN/'captions_train2017.local-record.bin').exists(),'unsafe partial or unexpected complete member')
        digest=hashlib.sha256();count=0
        with partial.open('rb') as stream:
            for chunk in iter(lambda:stream.read(256*1024),b''):
                count+=len(chunk);require(count<=EXPECTED,'partial overrun');digest.update(chunk)
        after=partial.lstat()
        require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),
                'partial changed while hashing')
        require(count==before.st_size==progress['body_bytes_read']==progress['body_bytes_saved'],
                'actual partial and producer byte counters differ')
        result.update(state='KNOWN_TIMEOUT_FAILURE_PARTIAL_FULL_SHA_VERIFIED',passed=True,
            original_epoch=EPOCH,original_deadline=DEADLINE,original_Job_returncode=124,
            original_Job_timed_out=True,original_Job_wall_seconds=guard['wall_seconds'],
            terminal_basis='Root original execute session ended; bound guarded_run returned only after owned Popen.wait124 and closed Job. No separatePID liveness probe.',
            worker_result_present=False,roster_present=False,actual_worker_CPU_count=2,
            actual_worker_affinity_mask=guard['worker_affinity_mask'],actual_worker_memory_limit_bytes=2147483648,
            actual_peak_process_commit_bytes=guard['actual_peak_process_commit_bytes'],
            observed_OS_PeakWorkingSetSize_bytes=guard['observed_OS_PeakWorkingSetSize_bytes'],
            partial=dict(path=partial.as_posix(),bytes=count,SHA256=digest.hexdigest(),
                st_dev=before.st_dev,st_ino=before.st_ino,st_mtime_ns=before.st_mtime_ns,st_ctime_ns=before.st_ctime_ns),
            preserved_prefix_bytes=count,expected_complete_localrecord_bytes=EXPECTED,
            missing_tail_bytes=EXPECTED-count,exact_new_tail_range_start=START+count,exact_new_tail_range_end=END,
            expected_ETag=ETAG,ETag_basis='Original fixed header gate/source and producer headersqualified; raw response ETag was not separately saved.',
            whole_member_CRC_or_decoded_SHA_qualified=False,selection_qualified=False,
            raw_private_location=RUN.as_posix(),raw_not_part_of_published12_source=True,
            full_Git_index_or_history_scan_performed=False,
            future_suffix_scope_eligible_only_after_new_publication_and_same_ETag=True,
            future_suffix_request_executed=False)
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as error:
        result.update(state='KNOWN_TIMEOUT_FAILURE_PARTIAL_AUDIT_FAILED',failure_type=type(error).__name__)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(a.output.resolve().as_posix().startswith((G/'local-only').as_posix()+'/'),'new private output must stay in ignoredL')
    result=consume()
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],passed=result['passed'],planned_images=16,actual_roster_images=0,
        missing_images=16,network_calls=0,partial=result.get('partial'),missing_tail_bytes=result.get('missing_tail_bytes'))))
