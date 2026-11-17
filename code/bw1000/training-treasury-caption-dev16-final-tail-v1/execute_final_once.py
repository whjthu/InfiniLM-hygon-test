"""One fresh600s scope for the final missing bytes; both previous requests stay failed."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys
import time
import uuid

from chain import (G,FAILURE_PUBLICATION,FAILURE_RESULT,FAILURE_SHA,CONSUMER_SHA,consumer_record,
    ORIGINAL_REQUEST,PREFIX_BYTES,PREFIX_SHA,TAIL_START,TAIL_END,TAIL_BYTES,require,ready,BINDING)

SOURCE=G/'code/bw1000/training-treasury-caption-dev16-final-tail-v1'
RUN=G/'local-only/training-treasury-caption-dev16-final-tail-v1-original'


def main(publication,claim_sha):
    ready()
    epoch,started=time.time(),time.monotonic();deadline,mono_end=epoch+600,started+600
    require(not RUN.exists(),'Original new-tail scope exists; reconcile, no retry or full-range restart')
    RUN.mkdir();request=uuid.uuid4().hex
    def save(name,value):
        with (RUN/name).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    save('intent.json',dict(request=request,publication=publication,claim_SHA256=claim_sha,epoch=epoch,deadline=deadline,
        original_failed_request=ORIGINAL_REQUEST,first_failed_request=BINDING['first_failed_request'],old_epoch_not_extended=True,planned_images=16,
        maximum_HTTP_open_attempts=1,HTTP_range_start=TAIL_START,HTTP_range_end=TAIL_END,new_HTTP_body_budget=TAIL_BYTES,
        original_verified_prefix_bytes=PREFIX_BYTES,original_prefix_SHA256=PREFIX_SHA,replay_allowed=False))
    try:
        require(re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',claim_sha),'Invalid publication or claim')
        claimraw=(SOURCE/'CLAIM.json').read_bytes();require(hashlib.sha256(claimraw).hexdigest()==claim_sha,'Newtail claim SHA differs')
        claim=json.loads(claimraw)
        require((claim['whole_epoch_seconds'],claim['CPU_cores'],claim['memory_limit_bytes'],claim['maximum_HTTP_open_attempts'])==
                (600,2,2147483648,1),'Newtail resource/request claim differs')
        mfraw=(SOURCE/'MF.json').read_bytes();manifest=json.loads(mfraw)
        actual={p.name for p in SOURCE.iterdir() if p.is_file()}
        require(actual==set(manifest)|{'MF.json'},'Newtail source inventory differs')
        for name,pin in manifest.items():
            path=PurePosixPath(name);require(len(path.parts)==1 and not path.is_absolute(),'Unsafe newtail source path')
            file=SOURCE/name;data=file.read_bytes()
            require(not file.is_symlink() and len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['SHA256'],
                    'Newtail source bytes differ')
        git=['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-c','core.fsmonitor=false','-C',str(G)]
        def show(args):
            available=min(deadline-time.time(),mono_end-time.monotonic())-10
            require(available>0,'Newtail original epoch exhausted before request')
            return subprocess.check_output(git+args,stderr=subprocess.DEVNULL,timeout=min(10,available))
        rel=SOURCE.relative_to(G).as_posix()
        require(show(['show',publication+':'+rel+'/MF.json'])==mfraw
                and show(['show',publication+':'+rel+'/CLAIM.json'])==claimraw,'Newtail publication differs')
        require(not show(['status','--porcelain','--untracked-files=all','--',rel]).strip(),'Own newtail source dirty/untracked')
        original=consumer_record()
        for commit in (FAILURE_PUBLICATION,publication):
            blob=show(['show',commit+':'+FAILURE_RESULT])
            require(hashlib.sha256(blob).hexdigest()==FAILURE_SHA,'Published originalfailure prerequisite absent/different')
            failure=json.loads(blob)
            # Exact published second-failure blob plus private independently verified
            # Job terminal/cumulative-prefix record are both mandatory. No schema guess.
            require(isinstance(failure,dict),'Published secondfailure record must be object')
        for commit in (BINDING['first_failure_publication'],publication):
            blob=show(['show',commit+':'+BINDING['first_failure_public_path']])
            require(hashlib.sha256(blob).hexdigest()==BINDING['first_failure_public_SHA256'],
                    'First failed request published evidence missing or changed')
        for name,expected in claim['original_published_copies'].items():
            blob=show(['show','71ee6ebbc8116571968f037f0cb43094652b3fce:code/bw1000/training-treasury-caption-dev16-v1/'+name])
            require(hashlib.sha256(blob).hexdigest()==expected and (SOURCE/name).read_bytes()==blob,'Original published parser/job copy differs')
        from job_guard import guarded_run
        gate=RUN/'job-gate.json';argv=[sys.executable,'-B','-u',str(SOURCE/'worker_tail.py'),'--directory',str(RUN),
            '--gate',str(gate),'--deadline',str(deadline),'--monotonic-end',str(mono_end)]
        guard=guarded_run(argv,gate,RUN,deadline,mono_end)
        worker=json.loads((RUN/'worker-result.json').read_bytes()) if (RUN/'worker-result.json').is_file() else None
        passed=bool(guard['returncode']==0 and not guard['timed_out'] and worker
            and worker['state']=='SUFFIX_ACQUIRED_COMPLETE_MEMBER_CRC_AND_METADATA_SELECTION_RECORDED'
            and worker['actual_metadata_selected']==16 and worker['missing_slots']==0
            and worker['selection']['missing_reference_groups']==0)
        result=dict(request=request,passed=passed,epoch=epoch,deadline=deadline,guarded_process=guard,worker=worker,
            original_failed_request=ORIGINAL_REQUEST,first_failed_request=BINDING['first_failed_request'],original_scope_still_failed=True,old_epoch_not_extended=True,
            original_consumer_result_SHA256=CONSUMER_SHA,published_failure_result_SHA256=FAILURE_SHA,
            HTTP_prefix_bytes_requested=0,replay_allowed=False,evaluation_ready=False,training_GO=False,GPU_GO=False,
            planned_images=16,missing_slots=16 if worker is None else worker['missing_slots'])
    except BaseException as error:
        result=dict(request=request,passed=False,epoch=epoch,deadline=deadline,error_type=type(error).__name__,
            original_failed_request=ORIGINAL_REQUEST,original_scope_still_failed=True,HTTP_prefix_bytes_requested=0,
            replay_allowed=False,planned_images=16,actual_metadata_selected=0,missing_slots=16,
            evaluation_ready=False,training_GO=False,GPU_GO=False)
    result.update(final_missing_tail_scope=True,
        final_failure_stops_local_network_route=True,
        local_network_route_state='METADATA_ONLY_MEDIA_SCORER_PENDING' if result['passed'] else 'STOP_LOCAL_NETWORK_ROUTE',
        further_tail_continuation_allowed=False,full_range_replay_allowed=False,
        previous_two_failure_requests_retained=[BINDING['first_failed_request'],ORIGINAL_REQUEST])
    save('outcome.json',result)
    print(json.dumps(dict(passed=result['passed'],request=request,HTTP_prefix_bytes_requested=0,evaluation_ready=False,training_GO=False,GPU_GO=False)))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--claim-sha256',required=True)
    a=p.parse_args();raise SystemExit(main(a.publication,a.claim_sha256))
