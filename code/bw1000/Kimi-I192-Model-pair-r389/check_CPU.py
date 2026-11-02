"""Actual six-arm driver/controller/bootstrap; device and Slurm leaves recorded.

Original335 Publisher/greedy and362b authenticated TCP are real CPU consumers.
No weights, Native dlopen, GPU, mathematical or Model qualification is claimed.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,tempfile,time

D=Path(__file__).resolve().parent
ROOT=Path('/refactor') if Path('/refactor/incoming').exists() else D.parents[2]
PARENT=ROOT/'incoming' if ROOT==Path('/refactor') else D.parent
sys.path[:0]=[str(D/'target'),str(PARENT/'Kimi-long-target-publisher-r335'),str(PARENT/'Kimi-long-pair-driver-r337')]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def driver_checks():
    from spec_driver import run_pair,ids_sha
    from counters import FIXED
    rec=load('original335_recording',PARENT/'Kimi-long-target-publisher-r335/check_publisher_CPU.py')
    cases=[]
    class Control:
        def collect(self,label,value):
            if label.startswith('long-publish:'):return [dict(node=n,value=value['value']) for n in range(4)]
            return [dict(value,node=n) if 'node' in value else value.copy() for n in range(4)]
    class Native(rec.RecordingNative):
        vocab=163840
        def __init__(self,miss=False):
            super().__init__([]);self.control=Control();self.calls=[];self.hist=[[0]*65 for _ in range(8)];self.miss=miss
        def Tstats(self):return [row[:] for row in self.hist]
        def call(self,*a,**k):
            result=super().call(*a,**k);T=len(a[3][0])
            if os.environ['INFINILM_KIMI_I192_GROUPED_R8669Y']=='1' and 8<=T<=64 and not self.miss:
                for d,row in enumerate(self.hist):row[T]+=d+1  # Deliberately unequal owner counts, no93 assumption.
            self.calls.append(result[-1]);return result
        def call_append(self,*a,**k):
            r=super().call_append(*a,**k);self.calls.append(r[-1]);return r
    class RPC:
        def __init__(self):self.closed=False;self.aborted=False;self.begin_count=0
        def command(self,cmd,epoch=None,body=b'',**fields):
            if cmd=='begin':self.begin_count+=1
            if cmd=='close':self.closed=True
            if cmd=='query':return dict(mode='draft',proposals=[10,11,12,13,14,15,163585][:fields['K']],context_tokens=fields['past'])
            return dict(context_tokens=fields.get('past',0))
        def abort(self):self.aborted=True
    def setup(miss=False):
        os.environ.update(FIXED);os.environ.pop('INFINILM_MXFP4_TILE_TP_R2559',None)
        n=Native(miss);n.top=[10,11,12,13,14,15,163585,17]
        ids=[1]*8635
        frozen=dict(order=[['retained8635','AR'],['near32k','AR'],['near32k','spec'],['retained8635','spec']],fixtures=[
            dict(name=name,prompt_tokens=items,input_tokens=len(items),prompt_ID_SHA256=ids_sha(items),output_budget=128,eos=[163585,163586])
            for name,items in [('retained8635',ids),('near32k',[1]*32638)]])
        return n,RPC(),frozen
    def run(miss=False,cancel=False,badgrade=False):
        n,r,f=setup(miss);saved=[];connected=[]
        def save(q):saved[:]=[json.loads(json.dumps(q))]
        def connect():connected.append(len(saved[-1]['requests']) if saved else 0);return r
        def grade(*args):
            if badgrade:raise ValueError('recorded wrong answer')
            assert args[2]=='eos';return {'recorded_quality_only':True}
        error=None
        try:q=run_pair(n,f,connect,grade,deadline=1e100,cancelled=lambda:cancel,save=save,enabled=True)
        except (RuntimeError,ValueError) as e:error=e;q=saved[-1] if saved else {}
        return n,r,q,error,connected
    n,r,q,error,connected=run()
    assert error is None and q['completed'] and len(q['requests'])==6 and connected==[0]
    assert [x['GROUPED'] for x in q['requests']]==[0,1,0,1,1,0]
    assert r.begin_count==6 and r.closed and not r.aborted
    assert all(x['finish']=='eos' and x['normal_reset'] and x['all_service_cards']==33 for x in q['requests'])
    assert all(x['T']==512 or x['T']==443 or x['T']==8 for r in q['requests'] for x in r['I192']['calls'])
    assert all(all(v>0 for v in x['I192']['T8_HOST_selected_by_local_owner']) for x in q['requests'] if x['GROUPED'])
    cases.append('actual6Specdriver335Publisher/greedy;fixed8635;2warm+ABBA;onefirstSpecconnect;unequal8ownerhist;EOS/reset/close')
    n,r,q,error,connected=run(miss=True)
    assert error is not None and len(q['requests'])==2 and q['I192_failed_request'] and r.aborted
    cases.append('firstONwarmNoHit saves actualpartial counters and stops beforeformal')
    n,r,q,error,_=run(badgrade=True)
    assert error is not None and len(q['requests'])==1 and q['requests'][0]['normal_reset'] and r.aborted
    cases.append('wrongwarmanswer saves tokens/EOS/reset and aborts beforelaterarm')
    n,r,q,error,connected=run(cancel=True)
    assert error is not None and not n.calls and not connected
    cases.append('originalcancel beforefirstconnect/forward')
    # T1 is ineligible even with ON; exercise exact wrapper independently.
    from counters import CounterNative
    n,r,f=setup();c=CounterNative(n,lambda q:None);c.begin_request(0);c.begin_request(1)
    n.seal_prompt=lambda *a,**k:None
    c.call('verify',[1],[0],[[9]],False)
    assert c.records[-1]['source_fallback_expected'] and all(not sum(x) for x in c.records[-1]['T_hist_delta_by_local_owner'])
    cases.append('ON T1 HOST histogram remainszero;512/443 observedzero inwhole6driver')
    return dict(passed=True,cases=cases,Native=False,mathematics_recorded=True)


def main():
    q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,weights=False,Slurm_submitted=False)
    q['driver']=driver_checks()
    # Execute original whole bootstrap lifecycle with six-arm recording leaf.
    import check_entry_CPU
    q['bootstrap']=check_entry_CPU.run()
    # Actual old362b/335 binary TCP source-identity join, loader recorded.
    import check_worker_binding_CPU
    import publisher,batch_wire_r9128
    q['old_worker_TCP']=check_worker_binding_CPU.run({'modules':{'publisher':publisher,'batch_wire_r9128':batch_wire_r9128}})
    # Same real controller class and markers, only Slurm/device observation recorded.
    sys.path.insert(0,str(D/'controller'))
    ctl=load('controller389_CPU',D/'controller/check_controller_CPU.py')
    scenarios=['normal','partial','exit_lag','account_lag','preflight_failure','target_failed','draft_failed',
               'bad_nonce','bad_node','native_deadline','cancel','listener_failure','evidence_failure','post_failure']
    q['controller']={name:ctl.run_controller_scenario(name) for name in scenarios}
    q['immutable_intent']=ctl.test_existing_markers()
    q.update(completed=True,passed=True,qualification='CPUcallcontractonly;NoNative/Model/deviceproof')
    return q


if __name__=='__main__':print(json.dumps(main(),indent=2))
