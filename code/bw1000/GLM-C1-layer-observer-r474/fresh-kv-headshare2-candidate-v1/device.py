"""One existing eight-DCU node per task. Four bounded paired packing windows."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes as C
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
from statistics import median
import sys
from threading import Barrier, Event
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from prepare import CUDA_PROVIDER, CUDA_SHA, EXPORTS, LABEL as BUILD_LABEL, digest, pinned
HIP = Path('/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0')
HIP_SHA = 'b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b'
HELPER_SHA = '74a50bc161afa681043f498977d7c0e79e9de1defb3352905013645bcda16c89'
TIMING_SHA = '6962475a0135702b647a2237eec7fc328676e5e8b277f05bcf6c0c9514f260bf'
DSO = Path('/refactor/candidates')/BUILD_LABEL/'libGLM_fresh_kv_headshare.so'
ORDER = (0,1,1,0)
TOKENS = 512
WINDOW = 5
BURST = 128
PLANNED_SAMPLES = 8*len(ORDER)*(WINDOW-1)


def need(value, explanation):
    if not value:
        raise ValueError(explanation)


def load(name, sha):
    path = HERE/name
    pinned(path,sha,64*1024)
    spec = importlib.util.spec_from_file_location('glm_headshare_'+name.split('.')[0],path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    raw = (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    pending = path.with_suffix('.pending')
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(pending,path)
    finally:
        pending.unlink()


def bind(helper, receipt):
    need(Path(receipt['shared']) == DSO and DSO.resolve(strict=True) == DSO
         and receipt['passed'] is True and receipt['completed'] is True and receipt['actual_device_TUs'] == 1
         and receipt['actual_GPU_steps'] == 0 and receipt['actual_native_links'] == 1
         and receipt['fresh_M_SHA256'] == receipt['actual_MD_SHA256']
         and receipt['cuda_provider_SHA256'] == CUDA_SHA and set(receipt['exports']) == set(EXPORTS)
         and DSO.stat().st_size == receipt['shared_bytes'] and helper.digest(DSO) == receipt['shared_SHA256'],
         'target compiler/link artifacts are incomplete or changed')
    need(helper.digest(HIP)==HIP_SHA and helper.digest(Path(CUDA_PROVIDER))==CUDA_SHA,
         'actual runtime provider differs')
    lookup = C.CDLL('libdl.so.2').dladdr
    lookup.argtypes, lookup.restype = [helper.V,C.POINTER(helper.DlInfo)],helper.I
    mode = os.RTLD_NOW|os.RTLD_LOCAL
    cuda = Path(CUDA_PROVIDER).resolve(strict=True)
    runtime = C.CDLL(str(cuda),mode=mode)
    helper.symbol(runtime,lookup,'cudaGetLastError',[],cuda)  # Passive provider identity only.
    hip_owner = C.CDLL(str(HIP),mode=mode)
    hip = helper.Hip(hip_owner,lookup,HIP)
    library = C.CDLL(str(DSO),mode=mode)
    bindings = [helper.symbol(library,lookup,name,[helper.V]*4+[helper.S]*2+[helper.V],DSO) for name in EXPORTS]
    return (runtime,library), hip, bindings


def identify(hip, helper, ordinal, expected):
    hip.call('SetDevice',ordinal)
    value = C.create_string_buffer(32)
    hip.call('DeviceGetPCIBusId',value,len(value),ordinal)
    pci = helper.pci_identity(value.value)
    need(hip.scalar('GetDevice')==ordinal and pci in expected,'actual ordinal or PCI changed')
    return pci


def window(hip, stream, first, last, call, argv, busy, tick, row):
    started = time.monotonic()
    end = started+WINDOW
    rows = []
    groups = completed = 0
    gpu_event_ms = 0.
    while time.monotonic()<end:
        tick()
        hip.call('EventRecord',first,stream)
        for index in range(BURST):
            if index%32==0:
                tick()
            need(call(*argv)==0,'native pack enqueue failed')
            completed += 1
        hip.call('EventRecord',last,stream)
        hip.call('StreamSynchronize',stream)
        elapsed = C.c_float()
        hip.call('EventElapsedTime',C.byref(elapsed),first,last)
        need(math.isfinite(elapsed.value) and elapsed.value>0,'invalid same-stream interval')
        gpu_event_ms += elapsed.value
        groups += 1
        row.update(partial_complete_groups=groups, partial_complete_submissions=groups*BURST)
        observed = time.monotonic() - started
        offset = len(rows)+1
        if offset <= WINDOW-1 and offset<=observed<offset+.5:
            percent = int(busy.read_text().strip())
            need(0<=percent<=100,'invalid driver busy sample')
            rows.append(dict(planned_offset_s=offset,observed_offset_s=observed,busy_percent=percent))
    wall = time.monotonic()-started
    need(completed>0 and groups*BURST==completed and WINDOW<=wall<=WINDOW+0.5
         and len(rows)==WINDOW-1,'incomplete real work or driver sample window')
    return dict(full_submissions=completed, groups=groups, wall_seconds=wall,
                completed_rows=completed*TOKENS*4, gpu_event_ms=gpu_event_ms, driver=rows)


def measure_card(card, scope, hip, helper, math_source, exports, stop, gate, output):
    rows = []
    result = dict(card=card, complete=False, planned_arms=len(ORDER), rows=rows)
    began_wall, began_mono = time.time(), time.monotonic()
    def tick():
        if (stop.is_set() or time.time()>=scope['deadline_epoch']-8
                or time.monotonic()-began_mono>=scope['deadline_epoch']-began_wall-8):
            raise TimeoutError('bounded device interval expired or O01 yielded')
    try:
        pci = identify(hip,helper,card,scope['PCI'])
        result['PCI'] = pci
        [busy] = [path for path in Path('/sys/class/drm').glob('renderD*/device/gpu_busy_percent')
                  if path.resolve().parent.name==pci]
        inputs = math_source.fixture(TOKENS)
        expected = math_source.reference(TOKENS,*inputs)
        second = tuple(bytes(value ^ 0x01 if index%2==0 else value for index,value in enumerate(raw))
                       for raw in inputs)
        expected_second = math_source.reference(TOKENS,*second)
        memory = hip.mem()
        with ExitStack() as cleanup:
            buffers = [helper.Guarded(hip,cleanup,size) for size in math_source.sizes(TOKENS)]
            with hip.stream() as anchors:
                stream, first, last = anchors
                argv = tuple(item.ptr for item in buffers)+(TOKENS,4,stream)
                for raw, gold in ((inputs,expected),(second,expected_second),(inputs,expected)):
                    tick()
                    for buffer,payload in zip(buffers[2:],raw,strict=True):
                        buffer.upload(payload)
                    for launch in exports:
                        for buffer in buffers[:2]:
                            buffer.sentinel(stream)
                        need(launch(*argv)==0,'native numerical invocation rejected')
                        hip.call('StreamSynchronize',stream)
                        math_source.audit(buffers,gold,tick)
                for launch in exports:
                    need(launch(*argv[:-3],1,4,stream)==1
                         and launch(argv[0],argv[0],*argv[2:])==1,'invalid native CABI input was accepted')
                gate.wait(timeout=25)
                for pos,which in enumerate(ORDER):
                    tick()
                    for buffer in buffers[:2]:
                        buffer.sentinel(stream)
                    hip.call('StreamSynchronize',stream)
                    row = dict(position=pos,heads_per_block=which+1,state='STARTED')
                    rows.append(row)
                    row.update(window(hip,stream,first,last,exports[which],argv,busy,tick,row))
                    row.update(state='COMPLETE',output_SHA256=math_source.audit(buffers,expected,tick),free_bytes=hip.mem()['free_bytes'])
        result.update(complete=True, memory_before=memory, memory_after_release=hip.mem(),
                      input_SHA256=[digest(raw) for raw in inputs], captured_model_activations=False)
    except Exception as error:
        gate.abort()
        stop.set()
        result['error'] = dict(type=type(error).__name__,reason=str(error)[:180])
    save(output/f'card-{card}.json',result)
    return result


def read_scope(scope, node, actual_step, now):
    need(scope['node']==node and re.fullmatch(r'1103606\.\d+',actual_step)
         and re.fullmatch(r'[0-9a-f]{32}',scope['request']) and scope['parent']=='1103606'
         and len(scope['PCI'])==len(set(scope['PCI']))==8 and all(
             re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci) for pci in scope['PCI'])
         and re.fullmatch(r'1103606\.\d+',scope['pre_step'])
         and str(scope['pre_step'])!=actual_step and scope['root_admitted'] is True
         and scope['deadline_epoch']==scope['epoch']+150 and scope['epoch']<=now<scope['deadline_epoch']-40,
         'wrong current allocation or new shared deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ('scope','out','build-receipt'):
        parser.add_argument('--'+item,type=Path,required=True)
    for item in ('scope-sha256','build-receipt-sha256'):
        parser.add_argument('--'+item,required=True)
    args = parser.parse_args()
    scope = json.loads(pinned(args.scope,args.scope_sha256,16384))
    node = socket.gethostname().split('.')[0]
    actual_step = os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    read_scope(scope,node,actual_step,time.time())
    need(os.environ['SLURM_CPUS_PER_TASK']=='12' and len(os.sched_getaffinity(0))>=12
         and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==scope['boot_id']
         and not list(Path('/sys/class/kfd/kfd/proc').iterdir()),'node boot or preexisting KFD holder differs')
    build = json.loads(pinned(args.build_receipt,args.build_receipt_sha256,512*1024))
    need(build['manifest_SHA256']==scope['build_source_manifest_SHA256']
         and build['original_step']==scope['build_step'] and build['label']==BUILD_LABEL,
         'wrong independently adopted compile origin')
    args.out.mkdir(mode=0o700)
    save(args.out/'intent.json',dict(scope,step=actual_step,planned_cards=8,planned_arms=32,
                                    planned_driver_samples=PLANNED_SAMPLES,replay_allowed=False))
    stopping = Event()
    signal.signal(signal.SIGUSR1,lambda *_:stopping.set())
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,scope['deadline_epoch']-time.time())
    helper = load('hip_runtime.py',HELPER_SHA)
    math_source = load('timing.py',TIMING_SHA)
    libraries, hip, exports = bind(helper,build)
    need(len(libraries)==2 and hip.scalar('GetDeviceCount')==8,'expected complete eight-DCU lease')
    gate=Barrier(8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(measure_card,i,scope,hip,helper,math_source,exports,stopping,gate,args.out) for i in range(8)]
        cards=[future.result() for future in futures]
    valid=[card for card in cards if card['complete']]
    complete=(len(valid)==8 and {card['PCI'] for card in valid}==set(scope['PCI']))
    pairs=[]
    for card in valid:
        rates=[row['completed_rows']/row['wall_seconds'] for row in card['rows']]
        pairs.append([rates[1]/rates[0],rates[2]/rates[3]])
    medians=[median(pair[index] for pair in pairs) for index in range(2)] if pairs else []
    result=dict(request=scope['request'],step=actual_step,node=node,complete=complete,planned_cards=8,
                valid_cards=len(valid), planned_arms=32, completed_arms=sum(
                    row['state']=='COMPLETE' for card in cards for row in card['rows']),
                planned_driver_samples=PLANNED_SAMPLES, planned_pairs=16, actual_pairs=2*len(pairs),
                pair_order_medians=medians, sharding_candidate=complete and min(medians)>=1.03,
                independent_clean_post_required=True, measured_model_performance=False,
                captured_model_activations=False, replay_allowed=False)
    save(args.out/'terminal.json',result)
    signal.setitimer(signal.ITIMER_REAL,0)
    return int(not complete)


if __name__=='__main__':
    raise SystemExit(main())
