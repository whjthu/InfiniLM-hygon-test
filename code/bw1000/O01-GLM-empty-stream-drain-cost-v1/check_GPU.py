"""Bounded CUDA-compat SDK empty-stream cost; no model, Torch, tensor or weight load."""
import ctypes as C
import hashlib, json, math, os, re, socket, statistics, time
from pathlib import Path

ROOT=Path('/probe')
PIN='ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e'
LIB=Path('/opt/dtk/cuda/cuda/lib64/libcudart.so.12')
NODE='f12r3n17';FLOOR=16*1024**3
ORDER=[1,2,2,1,1,2];ITERATIONS=512;WARM=5;MEASURE=20
epoch=float(os.environ['DIAGNOSTIC_EPOCH']);deadline=float(os.environ['DIAGNOSTIC_DEADLINE'])
assert math.isfinite(epoch) and deadline==epoch+360
assert socket.gethostname().split('.')[0]==NODE and os.environ['SLURM_JOB_ID']=='1087656'
assert hashlib.sha256(LIB.read_bytes()).hexdigest()==PIN, 'original906 SDK provider differs'
lib=C.CDLL(str(LIB))
def api(name,args):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=C.c_int;return fn
count=api('cudaGetDeviceCount',[C.POINTER(C.c_int)])
setdev=api('cudaSetDevice',[C.c_int])
pci=api('cudaDeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int])
create=api('cudaStreamCreateWithFlags',[C.POINTER(C.c_void_p),C.c_uint])
sync=api('cudaStreamSynchronize',[C.c_void_p])
destroy=api('cudaStreamDestroy',[C.c_void_p])
memory=api('cudaMemGetInfo',[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)])
def ck(rc):assert rc==0,('SDK returncode',rc)
def check_deadline():
    assert time.time()<min(epoch+140,deadline-90),'original single epoch budget exhausted'
def save(name,v):
    with (ROOT/name).open('x') as f:json.dump(v,f,indent=2)
def mem():
    free=C.c_size_t();total=C.c_size_t();ck(memory(C.byref(free),C.byref(total)))
    assert free.value>=FLOOR and total.value==68702699520
    return dict(free=free.value,total=total.value)
n=C.c_int();ck(count(C.byref(n)));assert n.value==1;ck(setdev(0))
bus=C.create_string_buffer(64);ck(pci(bus,len(bus),0));identity=bus.value.decode().lower()
if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',identity):identity='0000:'+identity
assert re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',identity)
save('assigned.json',dict(PCI=identity,visible_cards=1,step='1087656.'+os.environ['SLURM_STEP_ID'],
                         node=NODE,PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])))
stream=C.c_void_p();windows=[];arms=[];error=None;closed=False;created=False
start=time.monotonic();before=mem()
try:
    check_deadline();ck(create(C.byref(stream),1));assert stream.value;created=True;ck(sync(stream))
    # Each operation starts and ends with no queued work. This deliberately
    # measures the extra already-drained driver call, not a compute drain.
    for arm,multiple in enumerate(ORDER):
        arm_rows=[]
        for j in range(WARM+MEASURE):
            check_deadline();ck(sync(stream))
            begin=time.perf_counter_ns();cpu=time.thread_time_ns()
            for _ in range(ITERATIONS):
                for _ in range(multiple):ck(sync(stream))
            cpu=time.thread_time_ns()-cpu;wall=time.perf_counter_ns()-begin
            check_deadline()
            row=dict(arm=arm,syncs_per_unit=multiple,index=j,warm=j<WARM,
                     units=ITERATIONS,actual_calls=ITERATIONS*multiple,wall_ns=wall,thread_CPU_ns=cpu)
            assert wall>0 and cpu>=0;windows.append(row);arm_rows.append(row)
        samples=[x['wall_ns']/ITERATIONS for x in arm_rows if not x['warm']]
        arms.append(dict(arm=arm,syncs_per_unit=multiple,measured_windows=len(samples),
                         median_wall_ns_per_unit=statistics.median(samples),minimum=min(samples),maximum=max(samples)))
    after=mem();ck(sync(stream));ck(destroy(stream));closed=True;stream=C.c_void_p()
except BaseException as e:
    error=dict(type=type(e).__name__,message=str(e)[:300])
finally:
    if created and not closed:
        ck(sync(stream));ck(destroy(stream));closed=True
    if error:save('failure.json',dict(error=error,completed_windows=len(windows),planned_windows=150,
                                    stream_destroyed=closed,windows=windows))
if error:raise RuntimeError(error)
pairs=[]
for i in range(3):
    a,b=arms[2*i:2*i+2];single=next(x for x in (a,b) if x['syncs_per_unit']==1)
    double=next(x for x in (a,b) if x['syncs_per_unit']==2)
    pairs.append(dict(pair=i,single_wall_ns_per_unit=single['median_wall_ns_per_unit'],
                      double_wall_ns_per_unit=double['median_wall_ns_per_unit'],
                      extra_empty_call_ns=double['median_wall_ns_per_unit']-single['median_wall_ns_per_unit']))
assert len(windows)==150 and sum(x['actual_calls'] for x in windows)==115200
save('result.json',dict(completed=True,stream_destroyed=closed,provider_SHA256=PIN,
                       node=NODE,step='1087656.'+os.environ['SLURM_STEP_ID'],PCI=identity,
                       planned_windows=150,completed_windows=len(windows),measured_windows=120,
                       SDK_timed_calls=115200,arms=arms,pairs=pairs,windows=windows,
                       median_extra_empty_call_ns=statistics.median(x['extra_empty_call_ns'] for x in pairs),
                       wall_seconds=time.monotonic()-start,memory_before=before,memory_after=after,
                       Model_GO=False,performance_win_proven=False,
                       limits=['Original906 exact SDK file, current O01 hardware, one dedicated nonblocking stream.',
                               'No GPU work is queued; this is already-drained SDK-call cost through ctypes/Python.',
                               'Runtime::syncStream wrapper, Tensor destruction and pending-compute wait costs are not measured.',
                               'No Model throughput, numerical math, service SLO, GPUbusy or final optimization acceptance.']))
