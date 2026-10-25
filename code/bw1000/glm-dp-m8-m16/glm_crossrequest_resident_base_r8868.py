"""Private Model owner control. No CUDA calls until instantiated by worker."""
import ctypes, hashlib, json, threading, os, time
GiB=1024**3
from pathlib import Path
BANK_BYTES=json.loads(Path(__file__).with_name("GLM-CROSSREQUEST-Model-source-manifest-r8868.json").read_bytes())["resident_physical_bytes_per_layer"]
ALLOWANCE=64*1024**2
RESERVE=6*GiB
FLOOR=512*1024**2
BOUNDS=(0,38,78)
def affordable(free_bytes, stage):
    return max(0,min(38 if stage==0 else 40,(int(free_bytes)-RESERVE-ALLOWANCE)//BANK_BYTES))
def make_mask(stage, count):
    assert stage in (0,1) and 0<=count<=(38 if stage==0 else 40)
    bits=[0]*78
    for i in range(BOUNDS[stage], BOUNDS[stage]+count):bits[i]=1
    return bits
def expected_dispatch(events, selected):
    return {'append_D2D_bytes':8192*sum(e['flat_tokens']for e in events),
            'exact_shared_calls':sum(e['round']>0 and len(e['rows'])in(8,16) for e in events),
            'resident_original601_row_calls':sum(len(e['rows']) for e in events if e['round']>0 and len(e['rows'])not in(8,16))}
class CandidateRejected(RuntimeError):pass
class ResidentControl:
    def __init__(self,lib,ctx,stage,rank,control,check,report,save):
        import torch
        self.torch=torch;self.lib=lib;self.ctx=ctx;self.stage=stage;self.rank=rank
        self.control=control;self.check=check;self.report=report;self.save=save
        self.participant=stage*8+rank;self.epoch=0;self.chosen=None;self.count=0
        lib.modern_glm_model_resident_mask_r5640.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.c_int]
        lib.modern_glm_model_resident_report_r5640.argtypes=[ctypes.c_void_p]
        lib.modern_glm_model_resident_report_r5640.restype=ctypes.c_char_p
        lib.modern_glm_idle_trim_r3771.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
        q=self.snapshot();assert q['Context_mask']==[False]*78 and not q['any_enabled']
        self.runtime=q['Runtime_identity'];report['resident_default_OFF']=True
    def snapshot(self):
        p=self.lib.modern_glm_model_resident_report_r5640(self.ctx);assert p
        q=json.loads(p);assert [q['layer_begin'],q['layer_end']]==list(BOUNDS[self.stage:self.stage+2])
        assert len(q['Context_mask'])==78 and len(q['layers'])==BOUNDS[self.stage+1]-BOUNDS[self.stage]
        assert all(x['Runtime_identity']==q['Runtime_identity']for x in q['layers'])
        return q
    def all_rows(self,values):
        # All values are nonnegative exact FP32 integers <2**24. One contributor
        # per participant slot, world SUM gathers metadata without rounding.
        assert all(isinstance(v,int)and 0<=v<2**24 for v in values)
        x=self.torch.zeros((16,len(values)+1),device='cuda',dtype=self.torch.float32)
        x[self.participant]=self.torch.tensor([1]+values,device='cuda',dtype=self.torch.float32)
        self.control(x);rows=x.cpu().tolist();assert all(r[0]==1 for r in rows)
        return [list(map(int,r[1:]))for r in rows]
    def vote(self,ok):return all(r[0]==1 for r in self.all_rows([int(bool(ok))]))
    def setter(self,bits,trace):
        a=(ctypes.c_uint8*78)(*bits)
        return self.lib.modern_glm_model_resident_mask_r5640(self.ctx,a,78,int(trace))
    def contract_guards(self):
        # Already EMPTY after the original reset. No pointer faults/capture.
        q=self.snapshot();a=(ctypes.c_uint8*78)(*([0]*78));bad=(ctypes.c_uint8*78)(*([0]*78))
        bad[BOUNDS[1-self.stage]]=1
        codes=[self.lib.modern_glm_model_resident_mask_r5640(self.ctx,a,77,0),self.lib.modern_glm_model_resident_mask_r5640(self.ctx,a,78,2),self.lib.modern_glm_model_resident_mask_r5640(self.ctx,bad,78,0)]
        foreign=[]
        t=threading.Thread(target=lambda:foreign.append(self.lib.modern_glm_model_resident_mask_r5640(self.ctx,a,78,0)))
        t.start();t.join(15);assert not t.is_alive()
        assert all(c!=0 for c in codes) and foreign==[1] and self.snapshot()==q
        self.report['resident_Model_guard_checks']={'invalid_count':codes[0],'invalid_trace':codes[1],'foreign_stage':codes[2],'foreign_owner':foreign[0],'report_unchanged':True,'active17_GPU_negative':'NOT_RUN; CPU Slots contract only','fault_injection':'NOT_RUN'}
    def apply(self,label,on,trace):
        # Caller public() reset first. Policy OFF drains and retires resident banks,
        # then existing idletrim and Torch cache trim, before fresh admission.
        print('GLM_RESIDENT_POLICY_BEGIN',label,self.stage,self.rank,os.getpid(),threading.get_ident(),time.time_ns(),flush=True)
        self.epoch+=1
        self.check(self.setter([0]*78,False))
        a=(ctypes.c_uint64*6)();self.check(self.lib.modern_glm_idle_trim_r3771(self.ctx,a,6))
        assert list(a)[3:]==[self.rank,self.stage,2]
        self.torch.cuda.synchronize(self.rank);self.torch.cuda.empty_cache()
        free=self.torch.cuda.mem_get_info(self.rank)[0]
        phase=int(hashlib.sha256(label.encode()).hexdigest()[:6],16)
        rows=self.all_rows([free//1024**2,self.epoch,phase,int(on),int(trace)])
        assert all(r[1:]==rows[0][1:]for r in rows),'phase/epoch/intent disagreement'
        if on and self.chosen is None:
            maxs=[min(affordable(r[0]*1024**2,s)for r in rows[s*8:(s+1)*8])for s in (0,1)]
            if min(maxs)<1:
                self.report['resident_admission_rejection']={'phase':label,'allowner_fresh_MiB':rows,'affordable':maxs};self.save()
                raise CandidateRejected('fresh admission cannot fit any resident layer at original reserve')
            self.chosen=list(maxs)
        count=self.chosen[self.stage]if on else 0
        bits=make_mask(self.stage,count)
        local_ok=free-count*BANK_BYTES>=RESERVE+ALLOWANCE
        if not self.vote(local_ok):
            self.report['resident_admission_rejection']={'phase':label,'allowner_fresh_MiB':rows,'selected':self.chosen};self.save()
            raise CandidateRejected('fresh admission original 6GiB+64MiB failed')
        code=-1;detail=''
        try:
            code=self.setter(bits,trace)
            if code:self.check(code)
        except Exception as e:detail=str(e)
        statuses=self.all_rows([int(code==0),count,self.epoch,phase])
        if not all(r[0]==1 for r in statuses):
            # All owners still EMPTY. Successful contexts OFF before stopping.
            self.check(self.lib.modern_glm_model_reset(self.ctx));self.check(self.setter([0]*78,False))
            self.report['resident_setter_rejection']={'phase':label,'statuses':statuses,'local_detail':detail};self.save()
            raise CandidateRejected('allowner resident setter rejected before Model forward')
        assert all(r[1]==(self.chosen[s]if on else 0)for s in (0,1)for r in statuses[s*8:(s+1)*8])
        q=self.snapshot();assert q['resident_physical_bytes_per_layer']==BANK_BYTES;assert q['Context_mask']==list(map(bool,bits)) and q['Runtime_identity']==self.runtime
        assert q['memory_trace']==bool(trace) and bool(q['any_enabled'])==bool(count)
        for layer in q['layers']:
            assert layer['enabled']==bool(bits[layer['global_layer']]) and layer['active_rows']==0
            if layer['enabled']:assert layer['key_shape']==layer['value_shape']==[32,640,8,256]
        actualfree=self.torch.cuda.mem_get_info(self.rank)[0]
        if not self.vote(actualfree>=RESERVE+ALLOWANCE):
            assert self.vote(self.lib.modern_glm_model_reset(self.ctx)==0),'postadmit rollback reset'
            assert self.vote(self.setter([0]*78,False)==0),'postadmit rollback OFF'
            raise CandidateRejected('post allocation 6GiB+64MiB admission')
        self.count=count
        receipt={'phase':label,'epoch':self.epoch,'on':on,'trace':trace,'selected_counts':self.chosen,'global_layer_ids':[i for i,b in enumerate(bits)if b],'physical_budget_bytes':count*BANK_BYTES,'logical_bank_bytes':count*160*1024**2,'fresh_free_before':free,'free_after_admission':actualfree,'allowner_fresh_MiB_epoch_phase_on_trace':rows,'allowner_setter_ok_counts_epoch_phase':statuses,'native':q}
        self.report.setdefault('resident_policy_phases',[]).append(receipt);self.save()
        print('GLM_RESIDENT_POLICY_END',label,self.stage,self.rank,os.getpid(),threading.get_ident(),q['Runtime_identity'],count,time.time_ns(),flush=True)
        return q
    def finish(self,label,before,events):
        q=self.snapshot();assert q['Runtime_identity']==self.runtime
        expect=expected_dispatch(events,self.count)
        for a,b in zip(before['layers'],q['layers']):
            assert a['global_layer']==b['global_layer'] and a['enabled']==b['enabled']
            expected=expect if a['enabled'] else {k:0 for k in expect}
            for k,w in expected.items():assert b[k]-a[k]==w,('resident dispatch',label,b['global_layer'],k,b[k]-a[k],w)
            assert b['packed_prefix_bytes']==0
        free=self.torch.cuda.mem_get_info(self.rank)[0]
        observed=q['min_observed_free']if q['memory_samples']else free
        floor_ok=min(free,observed)>=FLOOR
        if not self.vote(floor_ok):raise RuntimeError(('resident 512MiB runtime observed floor',label,free,observed))
        self.report.setdefault('resident_phase_after',[]).append({'phase':label,'native':q,'free_after':free,'observed_is_peak':False,'expected_per_selected_layer_delta':expect,'runtime_floor_observed_passed':True})
        return q
