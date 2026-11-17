"""Unique600s public media-only scope, after exact source/success/roster gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from media_engine import require

G = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
SOURCE = G/'code/bw1000/training-treasury-caption-dev16-media-stage-v1'
RUN = G/'local-only/training-treasury-caption-dev16-media-stage-v1-original'
SUCCESS_PUB = '966516e0026939c9a45fd20701c5d1a292cde14d'
SUCCESS_PATH = 'results/bw1000/2026-11-17/training-caption-dev16-complete-member-and-private-roster-independent-terminal.json'
SUCCESS_SHA = '47bfd69ac07f08ffae859f0abff58f3abcf2bf74fb5a00d0a42fc6e131b2230a'


def main(publication, claim_sha):
    epoch, started = time.time(), time.monotonic(); deadline, mono_end = epoch+600, started+600
    require(not RUN.exists(), 'Original media scope exists: reconcile original request, never repeat or backfill')
    RUN.mkdir(); request = uuid.uuid4().hex
    def save(name, value):
        with (RUN/name).open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    save('intent.json', dict(request=request, publication=publication, claim_SHA256=claim_sha,
        epoch=epoch, deadline=deadline, planned_images=16, maximum_HTTP_open_attempts=16,
        maximum_concurrent_requests=4, maximum_total_body_bytes=33554432,
        maximum_bytes_per_image=2097152, previous_caption_requests_not_replayed=True, replay_allowed=False))
    result = dict(request=request, epoch=epoch, deadline=deadline, passed=False,
        planned_images=16, actual_qualified_images=0, missing_or_rejected=16, replay_allowed=False,
        no_failure_backfill=True, further_local_media_retry_allowed=False,
        evaluation_ready=False, Dataset_GO=False, training_GO=False, GPU_GO=False)
    try:
        require(re.fullmatch('[0-9a-f]{40}', publication) and re.fullmatch('[0-9a-f]{64}', claim_sha), 'Invalid source publication/claim')
        claim_raw = (SOURCE/'CLAIM.json').read_bytes(); claim = json.loads(claim_raw)
        require(hashlib.sha256(claim_raw).hexdigest() == claim_sha, 'Media claim SHA differs')
        require((claim['whole_epoch_seconds'], claim['CPU_cores'], claim['memory_limit_bytes'],
                 claim['maximum_concurrent_requests'], claim['maximum_HTTP_open_attempts']) == (600,2,2147483648,4,16), 'Media resource/request claim differs')
        mf_raw = (SOURCE/'MF.json').read_bytes(); mf = json.loads(mf_raw)
        require({p.name for p in SOURCE.iterdir() if p.is_file()} == set(mf)|{'MF.json'}, 'Media source inventory differs')
        for name, pin in mf.items():
            require(Path(name).name == name, 'Unsafe source name')
            file = SOURCE/name; raw = file.read_bytes()
            require(not file.is_symlink() and len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256'], 'Media source changed')
        git = ['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-c','core.fsmonitor=false','-C',str(G)]
        def show(args):
            left = min(deadline-time.time(), mono_end-time.monotonic())-15
            require(left > 0, 'Original media epoch exhausted before publication gate')
            return subprocess.check_output(git+args, stderr=subprocess.DEVNULL, timeout=min(10,left))
        relative = SOURCE.relative_to(G).as_posix()
        require(show(['show',publication+':'+relative+'/MF.json']) == mf_raw
                and show(['show',publication+':'+relative+'/CLAIM.json']) == claim_raw
                and not show(['status','--porcelain','--untracked-files=all','--',relative]).strip(), 'Media publication/own dirty state differs')
        for commit in (SUCCESS_PUB, publication):
            require(hashlib.sha256(show(['show',commit+':'+SUCCESS_PATH])).hexdigest() == SUCCESS_SHA, 'Published fixed16 successful metadata source missing/changed')
        original_job = show(['show','71ee6ebbc8116571968f037f0cb43094652b3fce:code/bw1000/training-treasury-caption-dev16-v1/job_guard.py'])
        require(original_job == (SOURCE/'job_guard.py').read_bytes()
                and hashlib.sha256(original_job).hexdigest() == 'c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71', 'Actual qualified Job copy differs')
        from worker_media import pinned, ROSTER, ROSTER_SHA, BINDING, BINDING_SHA
        roster = pinned(ROSTER, ROSTER_SHA, 32768); pinned(BINDING, BINDING_SHA, 1024**2)
        require(len(roster) == 16, 'Frozen16 roster differs before worker')
        import PIL
        require(PIL.__version__ == claim['Pillow_version'], 'Pillow version changed')
        for pin in claim['Pillow_passive_module_pins']:
            file = Path(pin['path']); raw = file.read_bytes()
            require(file.is_file() and not file.is_symlink() and len(raw) == pin['bytes']
                    and hashlib.sha256(raw).hexdigest() == pin['SHA256'], 'Pillow passive source provider changed')
        from job_guard import guarded_run
        gate = RUN/'job-gate.json'
        argv = [sys.executable,'-B','-u',str(SOURCE/'worker_media.py'),'--directory',str(RUN),
                '--gate',str(gate),'--deadline',str(deadline),'--monotonic-end',str(mono_end)]
        guard = guarded_run(argv,gate,RUN,deadline,mono_end)
        worker = json.loads((RUN/'worker-result.json').read_bytes()) if (RUN/'worker-result.json').is_file() else None
        passed = bool(guard['returncode'] == 0 and not guard['timed_out'] and worker and worker['passed'] is True
                      and worker['actual_qualified_images'] == 16 and worker['missing_or_rejected'] == 0)
        result.update(passed=passed, guarded_process=guard, worker=worker,
            actual_qualified_images=worker['actual_qualified_images'] if passed else 0,
            missing_or_rejected=0 if passed else 16,
            state='FIXED16_MEDIA_STAGE_ONLY_ATTRIBUTION_NATIVE_SCORER_PENDING' if passed else 'STOP_MEDIA_STAGE_NO_BACKFILL')
    except BaseException as error:
        result.update(state='STOP_MEDIA_STAGE_NO_BACKFILL',error_type=type(error).__name__)
    save('outcome.json',result)
    print(json.dumps({k:result[k] for k in ('request','passed','state','planned_images','actual_qualified_images','missing_or_rejected','Dataset_GO','training_GO','GPU_GO')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--publication',required=True); p.add_argument('--claim-sha256',required=True)
    a=p.parse_args(); raise SystemExit(main(a.publication,a.claim_sha256))
