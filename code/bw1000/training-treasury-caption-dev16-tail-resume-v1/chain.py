"""Immutable original failure/prefix identity; never rehash original prefix during preparation."""
import hashlib
import json
from pathlib import Path
import stat

G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
ORIGINAL_REQUEST='6b6f69f382b24a129a997aac0e0e4d40'
ORIGINAL_PUBLICATION='71ee6ebbc8116571968f037f0cb43094652b3fce'
FAILURE_PUBLICATION='db776bc0475f5276e65da780522d6d30a0d73df7'
FAILURE_RESULT='results/bw1000/2026-11-17/training-caption-dev16-original-public-Range-timeout-and-partial.json'
FAILURE_SHA='6df8b3908e28812aba84d38d4202dd811dcd114e135cdf0f74a5d4c6881d618b'
CONSUMER=G/'local-only/training-treasury-caption-dev16-v1/known-timeout-partial-independent-result.json'
CONSUMER_SHA='1c8a5d27cbb85a35dba1496c05621f64ae55b4536922a06902e5fa782f53502d'
PREFIX=G/'local-only/training-treasury-caption-dev16-v1-acquire-select-original/captions_train2017.local-record.partial'
PREFIX_BYTES=8650752
PREFIX_SHA='fa39234196d0f6d93f22502596a237ee82461d2a01cf6b42f2a11a790c912917'
TAIL_START,TAIL_END,TAIL_BYTES=167937798,178103193,10165396
FULL_BYTES=18816148
ETAG='"f4bbac642086de4f52a3fdda2de5fa2c"'


def require(ok,why):
    if not ok:raise ValueError(why)


def consumer_record():
    require(CONSUMER.is_file() and not CONSUMER.is_symlink() and CONSUMER.stat().st_size<=8192,'unsafe independent failure record')
    raw=CONSUMER.read_bytes();require(hashlib.sha256(raw).hexdigest()==CONSUMER_SHA,'independent failure record SHA differs')
    value=json.loads(raw)
    require(value['passed'] is True and value['state']=='KNOWN_TIMEOUT_FAILURE_PARTIAL_FULL_SHA_VERIFIED'
            and value['original_request']==ORIGINAL_REQUEST and value['original_Job_returncode']==124
            and value['original_Job_timed_out'] is True and value['original_scope_replay_allowed'] is False
            and value['planned_images']==value['missing_images']==16 and value['actual_roster_images']==0,
            'original known failure/16 denominator differs')
    require(value['partial']['path']==PREFIX.as_posix() and value['partial']['bytes']==PREFIX_BYTES
            and value['partial']['SHA256']==PREFIX_SHA and value['expected_ETag']==ETAG
            and value['exact_new_tail_range_start']==TAIL_START and value['exact_new_tail_range_end']==TAIL_END
            and value['missing_tail_bytes']==TAIL_BYTES,'original prefix/tail identity differs')
    return value


def copy_prefix(destination,original,remaining):
    """At future execution only: hash once while copying to exclusive new file, never modify original."""
    before=PREFIX.lstat();expected=original['partial']
    require(stat.S_ISREG(before.st_mode) and before.st_size==PREFIX_BYTES
            and all(getattr(before,key)==expected[key] for key in ('st_dev','st_ino','st_mtime_ns','st_ctime_ns')),
            'verified original prefix stat changed')
    digest=hashlib.sha256();count=0
    with PREFIX.open('rb') as src,destination.open('xb') as out:
        while count<PREFIX_BYTES:
            remaining();chunk=src.read(min(256*1024,PREFIX_BYTES-count));require(chunk,'original prefix truncated')
            count+=len(chunk);digest.update(chunk);out.write(chunk)
        out.flush()
    after=PREFIX.lstat()
    require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
            and count==PREFIX_BYTES and digest.hexdigest()==PREFIX_SHA,'original prefix changed or copy SHA mismatch')
    return digest,dict(bytes=count,SHA256=digest.hexdigest(),original_not_modified=True,HTTP_prefix_bytes_requested=0)
