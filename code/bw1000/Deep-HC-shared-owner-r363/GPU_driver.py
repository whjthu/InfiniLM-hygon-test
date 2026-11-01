"""Same compiled HC358 component and fixtures, one process/eight persistent owner threads."""
import pathlib,json,hashlib,os,time,ctypes,traceback,numpy as np
from fixture import load
from affinity_protocol import rank_plan
from thread_protocol import run,aligned_measurements
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
P=ctypes.c_void_p;I=ctypes.c_int;U=ctypes.c_uint;D=ctypes.c_double

def configure(lib):
    for n in ['error','report']:getattr(lib,'deep_hc_epilogue_'+n+'_r358').restype=ctypes.c_char_p
    lib.deep_hc_epilogue_create_r358.argtypes=[I,D,D,I,P,P,P];lib.deep_hc_epilogue_create_r358.restype=P
    lib.deep_hc_epilogue_forward_r358.argtypes=[P,I,ctypes.POINTER(P),P];lib.deep_hc_epilogue_forward_r358.restype=I
    lib.deep_hc_epilogue_destroy_r358.argtypes=[P,ctypes.POINTER(I)];lib.deep_hc_epilogue_hit_r358.argtypes=[P,ctypes.POINTER(U)]
    lib.deep_hc_epilogue_state_r358.argtypes=[P,ctypes.POINTER(P)]
    lib.deep_hc_canonical_calls_r7130.argtypes=[I,I];lib.deep_hc_canonical_calls_r7130.restype=ctypes.c_uint64

def run_backend(factory,observe=None):
    """Production thread entrance, also exercised with the real-header CPU backend."""
    owners=[None]*8
    def setup(rank):
        owners[rank]=factory(rank);owners[rank].setup();return owners[rank]
    def execute(rank,owner,phase):return owner.execute(phase)
    def finish(rank,owner):return owners[rank].finish()if owners[rank]is not None else dict(normal_close=True,context_created=False)
    result=run(setup,execute,finish,observe=(lambda record:observe(record,owners))if observe is not None else None)
    complete=[r for r in result['records']if r['phase']['kind']!='measurement'or all(v is not None for v in r['owners'])]
    result['incomplete_measurement_trials']=[r['phase']['trial']for r in result['records']if r['phase']['kind']=='measurement'and any(v is None for v in r['owners'])]
    result['aligned_measurements']=aligned_measurements(complete)
    if result['passed']:
        masks=[set(o['final_allowed'])for o in result['owners']]
        assert len({o['TID']for o in result['owners']})==8
        assert len({tuple(o['initial_allowed'])for o in result['owners']})==1
        assert all(len(m)==4 for m in masks)and all(not masks[i]&masks[j]for i in range(8)for j in range(i))
    return result

class GPUOwner:
    def __init__(self,rank,prepared,lib,torch,step):
        self.rank=rank;self.prepared=prepared;self.lib=lib;self.torch=torch;self.ctx=None;self.started=time.perf_counter()
        self.out=R/'results'/f'Deep363-GPU-v2-{step}-owner{rank}.json';assert not self.out.exists()
        self.q=dict(rank=rank,actual_step=step,completed=False,passed=False,calls=[],warmup=[],Model_loaded=False)
    def save(self):
        tmp=self.out.with_suffix('.tmp');tmp.write_text(json.dumps(self.q,indent=2)+'\n');tmp.replace(self.out)
    def check(self,rc):
        if rc:raise RuntimeError(self.lib.deep_hc_epilogue_error_r358().decode())
    def setup(self):
        rank=self.rank;torch=self.torch;lib=self.lib;q=self.q;data,proof=self.prepared
        initial,assigned=rank_plan(rank);self.assigned=assigned;torch.cuda.set_device(rank)
        props=torch.cuda.get_device_properties(rank);assert 'gfx936'in getattr(props,'gcnArchName','')
        q.update(CPU_affinity_initial=initial,CPU_affinity_assigned=assigned,fixture=proof,memory_before_context=list(torch.cuda.mem_get_info()),hardware=dict(node=os.environ['SLURMD_NODENAME'],Runtime_device=rank,visible_device_list=os.environ['HIP_VISIBLE_DEVICES'],name=props.name,arch=props.gcnArchName))
        weights=data['weights'];self.ctx=lib.deep_hc_epilogue_create_r358(rank,data['rms_eps'],data['hc_eps'],data['iterations'],*[P(a.ctypes.data)for a in weights]);assert self.ctx,lib.deep_hc_epilogue_error_r358().decode()
        q['CPU_affinity_after_Runtime_create']=sorted(os.sched_getaffinity(0));assert q['CPU_affinity_after_Runtime_create']in [initial,assigned]
        self.backing=[torch.empty(16384+64,device=f'cuda:{rank}',dtype=torch.bfloat16),torch.empty(4096+64,device=f'cuda:{rank}',dtype=torch.bfloat16),torch.empty(4+16,device=f'cuda:{rank}',dtype=torch.float32),torch.empty(16+16,device=f'cuda:{rank}',dtype=torch.float32)]
        b=self.backing;self.x=b[0][:16384].view(1,4,4096);self.x.copy_(torch.from_numpy(data['input_bits']).view(torch.bfloat16).to(f'cuda:{rank}'))
        self.outputs=[b[1][:4096].view(1,4096),b[2][:4].view(1,4),b[3][:16].view(1,4,4)];self.ptrs=(P*4)(*[P(t.data_ptr())for t in [self.x,*self.outputs]])
        self.tails=[b[0][16384:],b[1][4096:],b[2][4:],b[3][16:]]
        for t in self.tails:t.fill_(17.)
        q['weights_state_initial']=self.state();self.save()
    def state(self):
        data=self.prepared[0];weights=data['weights'];arrays=[np.empty_like(a)for a in weights]+[np.empty_like(a)for a in weights]+[np.empty(3,np.float32)]
        ptrs=(P*7)(*[P(a.ctypes.data)for a in arrays]);self.check(self.lib.deep_hc_epilogue_state_r358(self.ctx,ptrs))
        assert all(np.array_equal(a,b)for a,b in zip(arrays[:3],weights))and all(np.array_equal(a,b)for a,b in zip(arrays[3:6],weights));assert np.array_equal(arrays[6],np.array([data['rms_eps'],data['hc_eps'],2],np.float32))
        return [hashlib.sha256(a.tobytes()).hexdigest()for a in arrays]
    def numeric(self):
        data=self.prepared[0];torch=self.torch;o=self.outputs
        actual=[o[0].cpu().view(torch.uint16).numpy().reshape(4096).copy(),o[1].cpu().numpy().reshape(4).copy(),o[2].cpu().numpy().reshape(4,4).copy()];metrics=[]
        for name,a in zip(['collapsed_BF16','post','comb'],actual):
            b=data['expected'][name];av=(a.astype(np.uint32)<<16).view(np.float32)if a.dtype==np.uint16 else a;bv=(b.astype(np.uint32)<<16).view(np.float32)if b.dtype==np.uint16 else b;assert np.isfinite(av).all()and np.isfinite(bv).all();delta=av.astype(np.float64)-bv.astype(np.float64);nrmse=float(np.linalg.norm(delta)/(np.linalg.norm(bv.astype(np.float64))+1e-30));assert nrmse<(.005 if name=='collapsed_BF16'else 1e-4),(name,nrmse)
            metrics.append(dict(name=name,NRMSE=nrmse,max_abs=float(np.max(np.abs(delta))),exact_diagnostic=bool(np.array_equal(a,b))))
        assert all(bool(torch.all(t==17.))for t in self.tails);assert np.array_equal(self.x.cpu().view(torch.uint16).numpy(),data['input_bits']);return metrics
    def call(self,mode,events=False):
        torch=self.torch;lib=self.lib;rank=self.rank
        assert os.environ['INFINILM_DEEP_HC_EPILOGUE_R358']==str(mode)
        for t in self.outputs:t.fill_(float('nan'))
        torch.cuda.synchronize();start=end=None
        if events:start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        before=int(lib.deep_hc_canonical_calls_r7130(rank,2));tick=time.perf_counter();self.check(lib.deep_hc_epilogue_forward_r358(self.ctx,mode,self.ptrs,P(torch.cuda.current_stream().cuda_stream)));wall=time.perf_counter()-tick
        if events:end.record();end.synchronize()
        hits=int(lib.deep_hc_canonical_calls_r7130(rank,2))-before;assert hits==1-mode
        detail=json.loads(lib.deep_hc_epilogue_report_r358().decode());hit=U();self.check(lib.deep_hc_epilogue_hit_r358(self.ctx,ctypes.byref(hit)));assert detail['candidate']==bool(mode)and hit.value==mode
        return dict(mode=mode,CAPI_wall_s=wall,native=detail,GPU_hit=hit.value,actual_afc6_baseline_hits=hits,producer_stream_event_ms=start.elapsed_time(end)if events else None,numerics=self.numeric())
    def execute(self,phase):
        q=self.q;kind=phase['kind'];mode=phase['mode'];started=time.perf_counter()
        entries=[self.call(mode,kind in ['baseline','candidate'])for _ in range(phase['repeats'])];ended=time.perf_counter()
        result=dict(trial=phase.get('trial'),kind=kind,mode=mode,repeats=phase['repeats'],loop_begin_s=started,loop_end_s=ended,local_loop_wall_s=ended-started,CAPI_wall_s=sum(v['CAPI_wall_s']for v in entries),native_component_s=sum(v['native']['HCpre_staging_outputcopy_drained_seconds']for v in entries),GPU_hits=sum(v['GPU_hit']for v in entries),max_NRMSE=max(z['NRMSE']for v in entries for z in v['numerics']),repeat_CAPI_wall_s=[v['CAPI_wall_s']for v in entries],repeat_native_component_s=[v['native']['HCpre_staging_outputcopy_drained_seconds']for v in entries])
        if kind=='baseline':
            q['baseline_numeric_first']=entries[0];q['CPU_affinity_after_first_baseline']=sorted(os.sched_getaffinity(0));assert q['CPU_affinity_after_first_baseline']==self.assigned
        elif kind=='candidate':q['candidate_numeric']=entries[0]
        elif kind=='warm':q['warmup'].append(entries[0])
        else:
            if not q['calls']:q['prepare_and_qualification_s']=started-self.started
            q['calls'].append(result)
        return result
    def finish(self):
        q=self.q
        try:
            if self.ctx and len(q['calls'])==20:
                q['weights_state_final']=self.state();assert q['weights_state_final']==q['weights_state_initial'];q['memory_after_measurement']=list(self.torch.cuda.mem_get_info());q['passed']=True
        except BaseException as exc:q.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
        finally:
            if self.ctx:
                expired=I();q['destroy_code']=self.lib.deep_hc_epilogue_destroy_r358(self.ctx,ctypes.byref(expired));q['Runtime_expired']=expired.value;q['normal_close']=q['destroy_code']==0 and expired.value==1;self.ctx=None
            else:q.update(normal_close=True,context_created=False)
            q.update(completed=True,overall_wall_s=time.perf_counter()-self.started);self.save()
        return dict(normal_close=q['normal_close'],Runtime_expired=q.get('Runtime_expired'),passed=q['passed'],path=str(self.out),SHA256=sha(self.out))

def main():
    started=time.perf_counter();step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/f'Deep363-GPU-v2-{step}.json';assert not out.exists()
    q=dict(completed=False,passed=False,actual_step=step,Model_loaded=False,processes=1,owner_threads=8,ports_required=[],network_service_started=False)
    try:
        mf=S/'GPU_MANIFEST.json';assert sha(mf)==os.environ['DEEP363_MANIFEST_SHA'];m=json.loads(mf.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items());q['manifest_SHA256']=sha(mf)
        bp=R/'candidates/Deep-HC-epilogue-r358-cpu-r2/CPU-build.json';assert sha(bp)==m['CPU700_receipt_SHA256'];build=json.loads(bp.read_bytes());assert build['completed']and not build['GPU_initialized']and not build['Native_dlopen'];library=pathlib.Path(build['library']);assert sha(library)==build['library_SHA256']==m['library_SHA256'];assert sha(build['Native_path'])==m['Native_SHA256']=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
        cp=R/'candidates/Deep-HC-shared-owner-r363-CPU-v3/CPU-contract.json';assert sha(cp)==m['CPU_contract_SHA256'];contract=json.loads(cp.read_bytes());assert contract['passed']and not contract['GPU_initialized']and not contract['Native_dlopen']
        import torch
        torch.set_num_threads(1);assert torch.cuda.device_count()==8;q['parent_initial_allowed']=sorted(os.sched_getaffinity(0));assert len(q['parent_initial_allowed'])>=32
        fp=bp.with_name('CPU-fixtures.json');assert sha(fp)==build['CPU_fixture_receipt_SHA256'];fixtures=json.loads(fp.read_bytes());prepared=[]
        for rank in range(8):
            data,proof=load(rank);old=fixtures['fixtures'][rank];assert proof['weights']==old['weights']and proof['input_SHA256']==old['input_SHA256']and proof['config_SHA256']==old['config_SHA256'];prepared.append((data,proof))
        lib=ctypes.CDLL(str(library));configure(lib);q['parent_prepare_before_threads_s']=time.perf_counter()-started
        def observe(record,owners):
            # All owners have completed this phase and wait at the next begin barrier.
            # Keep JSON serialization/NFS writes outside every concurrent CAPI interval.
            for owner in owners:owner.save()
            q['last_completed_phase']=record['phase'];tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(out)
        result=run_backend(lambda rank:GPUOwner(rank,prepared[rank],lib,torch,step),observe);q['protocol']=result;q['passed']=result['passed']and all(c.get('passed')for c in result['closes']);assert result['parent_final_allowed']==q['parent_initial_allowed']
        q['timing_scope']='Trial=one32call ABBAarmblock. Startbarrier precedes eachowner loop; endbarrier is outside localwall. Primary max8 sums are same trial only; per-repeat logical-index maxima are secondary and not synchronized GPUbusy. Wholephase/batchwall include barriers/hostchecks. Events only coldqualification; noABBAevents.'
        q['scope']='Same CPU700 C++/HIP andactual8smallHCfixtures as1277; oneprocess8Runtimeowners only. No C++math/sharedSDK/Model changes; noModelprojection.'
    except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
    q.update(completed=True,whole_driver_wall_s=time.perf_counter()-started);out.write_text(json.dumps(q,indent=2)+'\n');assert q['passed'],q.get('error')
if __name__=='__main__':main()
