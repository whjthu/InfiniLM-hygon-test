"""Exact original r2392 helper across two DSOs through the production thread entrance."""
import pathlib,os,sys,json,ctypes,subprocess,hashlib,shlex,time,traceback
from GPU_driver import run_backend
from affinity_protocol import rank_plan
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-HC-full-pre-fusion-r372-thread-CPU-v1'
H=R/'candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM/csrc'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

class HeaderOwner:
    def __init__(self,rank,case,libs):self.rank=rank;self.case=case;self.libs=libs;self.closed=False;self.calls=[];self.initial=None
    def setup(self):
        self.initial,self.expected=rank_plan(self.rank)
        if self.case=='setup_failure'and self.rank==3:raise RuntimeError('injected setup failure')
        if self.case=='narrow_before_helper':os.sched_setaffinity(0,self.expected)
    def actual_helper(self):
        d=0 if self.case=='logical0_overlap'else self.rank
        for lib in self.libs:
            values=(ctypes.c_int*1024)();changes=ctypes.c_uint();error=ctypes.create_string_buffer(1024)
            count=lib.probe(d,values,ctypes.byref(changes),error)
            if count<0:raise RuntimeError(error.value.decode())
            selected=list(values[:count]);assert selected==sorted(os.sched_getaffinity(0))
            if self.case!='logical0_overlap':assert selected==self.expected and changes.value==1
        return selected
    def execute(self,phase):
        tick=time.perf_counter();trial=phase.get('trial',0);repeat=[]
        for i in range(phase['repeats']):
            assert os.environ['INFINILM_DEEP_HC_FULL_R372']==str(phase['mode'])
            selected=self.actual_helper()
            if self.case=='baseline_failure'and self.rank==3:raise RuntimeError('injected independent baseline rejection')
            if self.case=='measurement_failure'and self.rank==3 and phase['kind']=='measurement':raise RuntimeError('injected measurement rejection')
            # Deliberately synthetic markers, only for same-trial max8 indexing.
            repeat.append(((self.rank+trial)%8+1+trial*.001+i*.00001)*1e-6)
        ended=time.perf_counter();self.calls.append(dict(phase));self.selected=selected
        return dict(trial=phase.get('trial'),mode=phase['mode'],repeats=phase['repeats'],CAPI_wall_s=sum(repeat),native_component_s=sum(repeat)*.9,local_loop_wall_s=ended-tick,loop_begin_s=tick,loop_end_s=ended,repeat_CAPI_wall_s=repeat)
    def finish(self):self.closed=True;self.final=sorted(os.sched_getaffinity(0));return dict(normal_close=True,actual_GPU_Runtime_created=False)

def scenario(case):
    parent=sorted(os.sched_getaffinity(0));assert len(parent)>=32
    libs=[ctypes.CDLL(str(B/f'affinity-{n}.so'))for n in ['A','B','A']]
    for lib in libs:lib.probe.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint),ctypes.c_void_p]
    owners=[None]*8
    def factory(rank):owners[rank]=HeaderOwner(rank,case,libs);return owners[rank]
    rejection=None
    observed=[]
    def observe(record,objects):
        assert len(record['owners'])==8
        assert all(objects[i].calls[-1]==record['phase']for i in range(8)if not(case in ['baseline_failure','measurement_failure']and i==3))if case not in ['setup_failure','narrow_before_helper']else True
        observed.append(record['phase'])
        if case=='observer_failure':raise RuntimeError('injected post-phase persistence rejection')
    try:result=run_backend(factory,observe)
    except AssertionError as exc:result=None;rejection=repr(exc)
    assert all(o and o.closed for o in owners)and sorted(os.sched_getaffinity(0))==parent
    if case=='positive':
        assert result['passed']and len(result['aligned_measurements'])==20
        for r in result['aligned_measurements']:
            i=r['trial'];expected=sum((8+i*.001+j*.00001)*1e-6 for j in range(32));assert abs(r['max8_CAPI_wall_s']-expected)<1e-15
        assert all(len(o.calls)==24 for o in owners)and len(observed)==24
    elif case=='logical0_overlap':
        assert result is None and rejection is not None and len({tuple(o.final)for o in owners})==1
    elif case=='measurement_failure':
        assert not result['passed']and result['incomplete_measurement_trials']==[0]and result['aligned_measurements']==[]
        assert 'injected measurement rejection'in result['errors'][3]['error']and all(result['threads_joined'])
    else:
        assert not result['passed']and not any(p['kind']=='candidate'for o in owners for p in o.calls)
        assert all(result['threads_joined'])
        if case=='narrow_before_helper':assert all('too narrow'in e['error']for e in result['errors'])
    return dict(case=case,passed=True,parent_allowed=parent,parent_unchanged=True,final_masks=[o.final for o in owners],all8_finished=True,phase_counts=[len(o.calls)for o in owners],production_protocol_passed=result['passed']if result else False,consumer_rejection=rejection,errors=result['errors']if result else None,thread_TIDs=[o['TID']for o in result['owners']]if result else None,max8_same_trial_index_contract_passed=case=='positive',marker_values_are_not_performance=True)

if len(sys.argv)>1:
    print(json.dumps(scenario(sys.argv[1])));sys.exit(0)
started=time.perf_counter();B.mkdir(exist_ok=False);deps={}
assert sha(H/'execution/owner_affinity_r2392.hpp')=='de2ffe645329558634740ed6b8a9ac7c70030160786eab38d065ce513afce71d'
for n in ['A','B']:
    md=B/f'affinity-{n}.d';args=['g++','-std=c++17','-O2','-fPIC','-pthread','-shared','-I'+str(H)]
    subprocess.run(args+['-MD','-MF',str(md),str(S/'CPU_affinity_probe.cpp'),'-o',str(B/f'affinity-{n}.so')],capture_output=True,text=True,check=True,timeout=30)
    parse=lambda t:set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]));fresh=subprocess.check_output(args+['-M',str(S/'CPU_affinity_probe.cpp')],text=True,timeout=20);assert parse(fresh)==parse(md.read_text());deps[n]={p:sha(p)for p in parse(fresh)}
env=dict(os.environ,INFINILM_OWNER_CPU_WIDTH_R2392='4')
cases=[json.loads(subprocess.check_output([sys.executable,'-B',str(__file__),case],text=True,env=env,timeout=90))for case in ['positive','setup_failure','baseline_failure','narrow_before_helper','logical0_overlap','observer_failure','measurement_failure']]
maps=pathlib.Path('/proc/self/maps').read_text();assert 'libDeep_HC_staged_r372.so'not in maps and 'libDeep_HC_T1_Model_r366.so'not in maps and 'libDeep_latest_native_r269.so'not in maps
q=dict(passed=all(v['passed']for v in cases),CPU_only=True,GPU_initialized=False,Native_dlopen=False,actual_GPU_Runtime_created=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_SHA256={p.name:sha(p)for p in S.iterdir()if p.is_file()},dependencies_SHA256=deps,DSO_SHA256={n:sha(B/f'affinity-{n}.so')for n in ['A','B']},cases=cases,seconds=time.perf_counter()-started,scope='Actual original r2392 header across twoCPU DSOs via production GPU_driver.run_backend/thread_protocol persistent8threads, positive and fourfailures. Max8 syntheticmarkers check indexing only; no GPU/Native numerical claim.')
(B/'CPU-contract.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(passed=q['passed'],actual_step=q['actual_step'],seconds=q['seconds'],cases=len(cases),GPU_initialized=False)))
