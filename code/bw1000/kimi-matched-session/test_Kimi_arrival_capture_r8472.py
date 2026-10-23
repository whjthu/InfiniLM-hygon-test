"""Only changed coordinator capture contract; reuse8456 TCP fault evidence.

No new CPU throughput numbers or GPU work. Immutable job data must reach durable
capture before a reference CPU/native mismatch is rejected.
"""
import io,dataclasses,types,json,tempfile,os
from kimi_actual_arrival_r8472 import RecordingBroker,FiniteBroker,LiveBuffer
import pathlib
from online_service_core_r8455 import Job


def reject(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('bad oracle accepted')


with tempfile.TemporaryFile(mode='w+') as capture:
    broker=RecordingBroker.__new__(RecordingBroker)
    connection=object();broker.owner_connections={3:connection}
    broker.capture=capture;broker.full_reference=True;broker.records=[]
    broker.core=types.SimpleNamespace(cursor=[0,0,0,0],busy=[None,None,None,0],
                                     jobs={0:Job(0,'prefill',(0,),(1,),(0,),((10,11),))})
    called=[];old=FiniteBroker.handle
    FiniteBroker.handle=lambda *args:called.append('parentfinish') or {'acked':0}
    try:
        req=dict(role='owner',op='finish',seq=0,ids=[21],oracle=dict(finite=True,native_equals_CPU=True,cpu_argmax_ids=[21],native_ids=[21]))
        broker.handle(req,12.5,connection)
        capture.seek(0);saved=json.loads(capture.readline())
        assert saved['job']['inputs']==[[10,11]] and saved['actual_ids']==[21]
        assert saved['coordinator_feedback_t']==12.5 and called==['parentfinish']
        broker.records=[];called.clear();req['oracle']['native_equals_CPU']=False
        reject(lambda:broker.handle(req,12.6,connection))
        assert not called # no corecommit before validation.
        capture.seek(0);assert len(capture.readlines())==2 # rejected actual retained.
    finally:FiniteBroker.handle=old
with tempfile.TemporaryDirectory()as folder:
    p=pathlib.Path(folder)/'live.jsonl';b=LiveBuffer(p)
    original=os.fsync;calls=[]
    os.fsync=lambda fd:calls.append(fd)
    try:
        b([dict(kind='token',t=1.25)]);b.flush()
        assert not p.exists() and not calls
        b.close();assert len(calls)==1 and json.loads(p.read_text())['t']==1.25
    finally:os.fsync=original
print(json.dumps(dict(CPU_capture_contract_passed=True,GPU='NOT_RUN',unchanged_TCP_matrix_reused=True,
                     actual_job_normalized=True,reference_failure_actuals_persisted_before_assert=True,no_core_commit_on_oracle_failure=True,
                     live_no_disk_fsync_until_after_drain=True,actual_live_timestamp_unchanged=True)))
