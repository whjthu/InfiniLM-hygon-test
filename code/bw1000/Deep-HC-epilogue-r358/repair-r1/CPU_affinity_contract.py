"""No GPU/Native load. Actual r2392 helper in two DSOs and real Linux allowed masks."""
import os,sys,pathlib,json,hashlib,subprocess,ctypes,shlex
from affinity_protocol import rank_plan
S=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('/refactor/candidates/Deep-HC-epilogue-r358-cpu-r1')
H=pathlib.Path('/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM/csrc')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
HEADER_SHA='de2ffe645329558634740ed6b8a9ac7c70030160786eab38d065ce513afce71d'

def worker(device,narrow):
    initial,expected=rank_plan(device if device<8 else 0)
    if narrow: os.sched_setaffinity(0,initial[:4])
    allowed=sorted(os.sched_getaffinity(0));results=[]
    for suffix in ['A','B','A']:
        lib=ctypes.CDLL(str(B/f'affinity-{suffix}.so'))
        lib.probe.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint),ctypes.c_void_p]
        values=(ctypes.c_int*1024)();changes=ctypes.c_uint();error=ctypes.create_string_buffer(1024)
        count=lib.probe(device,values,ctypes.byref(changes),error)
        results.append(dict(DSO=suffix,selected=list(values[:max(0,count)]),count=count,changes=changes.value,error=error.value.decode()))
        if count<0:break
    return dict(device=device,initial=initial,allowed_before_actual_helper=allowed,expected=expected,results=results,final=sorted(os.sched_getaffinity(0)))

if len(sys.argv)>1:
    print(json.dumps(worker(int(sys.argv[1]),sys.argv[2]=='1')));sys.exit(0)
assert sha(H/'execution/owner_affinity_r2392.hpp')==HEADER_SHA
deps={}
for suffix in ['A','B']:
    md=B/f'affinity-{suffix}.d';lib=B/f'affinity-{suffix}.so'
    args=['g++','-std=c++17','-O2','-fPIC','-pthread','-shared','-I'+str(H)]
    subprocess.run(args+['-MD','-MF',str(md),str(S/'CPU_affinity_probe.cpp'),'-o',str(lib)],check=True,capture_output=True,text=True,timeout=30)
    parse=lambda t:set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]))
    fresh=subprocess.check_output(args+['-M',str(S/'CPU_affinity_probe.cpp')],text=True,timeout=20)
    assert parse(fresh)==parse(md.read_text())
    deps[suffix]={p:sha(p)for p in parse(fresh)}
env=dict(os.environ,INFINILM_OWNER_CPU_WIDTH_R2392='4')
def run(d,narrow=False):
    return json.loads(subprocess.check_output([sys.executable,'-B',str(__file__),str(d),str(int(narrow))],env=env,text=True,timeout=10))
good=[run(d)for d in range(8)]
assert len({tuple(v['initial'])for v in good})==1
assert all(all(t['selected']==v['expected']and t['changes']==1 and not t['error']for t in v['results'])for v in good)
sets=[set(v['final'])for v in good]
assert all(len(v)==4 for v in sets)and all(not sets[i]&sets[j]for i in range(8)for j in range(i))
narrow=run(0,True);assert 'too narrow'in narrow['results'][0]['error']
invalid=run(8);assert 'eight-device'in invalid['results'][0]['error']
logical0=[run(0)for _ in range(8)]
assert len({tuple(v['final'])for v in logical0})==1
q=dict(passed=True,CPU_only=True,GPU_initialized=False,Native_dlopen=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],header_path=str(H/'execution/owner_affinity_r2392.hpp'),header_SHA256=HEADER_SHA,dependencies_SHA256=deps,positive_physical_Runtime_device=good,negative_narrow_mask=narrow,negative_invalid_device=invalid,negative_logical0_consumer_rejected_overlap=True,negative_logical0_selected=logical0[0]['final'],scope='Exact production helper executed across two independent CPU DSOs with readback. Runtime device dispatch is rank 0..7; no GPU Runtime create performed.')
(B/'CPU-affinity.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps(dict(passed=True,actual_step=q['actual_step'],eight_disjoint_masks=[v['final']for v in good],GPU_initialized=False)))
