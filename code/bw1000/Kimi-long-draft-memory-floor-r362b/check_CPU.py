"""Actual private guard/worker main/wire/RPC plan; CPU recording loader/math only."""
from pathlib import Path
from types import SimpleNamespace
import contextlib,hashlib,json,os,socket,sys,tempfile,threading,time
from memory_floor import guarded_codec,FLOOR


class Allocator:
    def __init__(self,before,after=None,fail=False):
        self.free=before;self.after=after;self.fail=fail;self.trims=self.syncs=0
    def mem_get_info(self):return self.free,64<<30
    def memory_allocated(self):return 14<<30
    def memory_reserved(self):return (60 if not self.trims else 16)<<30
    def max_memory_allocated(self):return 15<<30
    def max_memory_reserved(self):return 62<<30
    def empty_cache(self):
        self.trims+=1
        if self.fail:raise RuntimeError('CPU injected trim error')
        self.free=self.after
    def synchronize(self):self.syncs+=1


def main():
    D=Path(__file__).resolve().parent
    runtime=Path('/refactor/incoming/Kimi-long-draft-RPC-CPU-r333')
    if not runtime.exists():runtime=D.parent/'Kimi-long-draft-RPC-CPU-r333'
    sys.path[:0]=[str(runtime),str(D)]
    from batch_wire_r9128 import send,recv
    import long_rpc,worker,pressure
    cases=[]
    class Base:
        def fence(self):pass
    for name,before,after,fail,success in [
        ('healthy_no_trim',FLOOR+1,None,False,True),
        ('exact_floor_no_trim',FLOOR,None,False,True),
        ('pressure_recovered_once',FLOOR-1,FLOOR+9,False,True),
        ('pressure_still_low_fail_closed',FLOOR-1,FLOOR-1,False,False),
        ('trim_error_fail_closed',FLOOR-1,FLOOR+9,True,False)]:
        allocator=Allocator(before,after,fail);report={};codec=guarded_codec(Base,SimpleNamespace(cuda=allocator),report)()
        error=None
        try:codec.fence()
        except RuntimeError as e:error=str(e)
        assert (error is None)==success
        assert allocator.trims==int(before<FLOOR)
        assert len(report['memory_fences'])==1
        assert report['memory_fences'][0]['free_before_bytes']==before
        if not success:assert error in ['Original4GiB draft memory floor','CPU injected trim error']
        cases.append(dict(name=name,passed=True,trim_count=allocator.trims))
    import torch  # CPU-only inference context; no CUDA operation is invoked.
    for mode in ['success','floor_failure']:
        with tempfile.TemporaryDirectory()as temporary:
            result=Path(temporary)/'worker.json';sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
            state={};errors=[];hello=worker.expected_hello('CPU-test-nonce-only-'+'0'*32)
            def factory(*args):
                report=args[-1];allocator=Allocator(FLOOR-1,FLOOR+100 if mode=='success'else FLOOR-1)
                Codec=guarded_codec(long_rpc.TorchFeatureCodec,SimpleNamespace(cuda=allocator),report)
                class RecordingCodec(Codec):
                    def decode(self,body,rows):return body
                    def proposals(self,values,debug,K):return list(range(K))
                class RecordingContext:
                    def __init__(self):self.length=0;self.phase='closed';self.published=0
                    def _check(self,*a):assert self.phase in ['prompt','decode']
                    def begin(self,epoch,request,prompt_tokens):self.phase='prompt';self.prompt_tokens=prompt_tokens
                    def append_prompt(self,epoch,request,past,part,fence):
                        fence();self.length=past+len(part)//(35840*2);self.published+=1
                    def seal_prompt(self,*a):assert self.length==32760;self.phase='decode'
                    def query(self,*a,**kw):return dict(mode='draft',values=(None,{}))
                    def retire(self,*a):self.phase='retired';return True
                    def close(self):self.phase='closed';state['context_closed']=True
                context=RecordingContext();codec=RecordingCodec('cpu');session=long_rpc.DraftSession(context,None,codec,enabled=True)
                state.update(context=context,session=session,allocator=allocator,loader_calls=1)
                def close():session.close();state['runtime_closed']=True
                return SimpleNamespace(session=session,close=close)
            original=(worker.Runtime,worker.verify_sources,worker.signal.signal,worker.signal.alarm,sys.argv)
            worker.Runtime=factory;worker.verify_sources=lambda *a:(runtime,D)
            worker.signal.signal=lambda *a:None;worker.signal.alarm=lambda *a:None
            os.environ.update(INFINILM_KIMI_LONG_RPC_R334='1',KIMI_DRAFT_NONCE_R9046=hello['nonce'],SLURM_JOB_ID=os.environ.get('SLURM_JOB_ID','0'),SLURM_STEP_ID=os.environ.get('SLURM_STEP_ID','0'))
            sys.argv=['worker.py','--manifest-sha','CPU-boundary','--host','127.0.0.1','--port',str(port),'--deadline-unix',str(time.time()+120),'--output',str(result)]
            def serve():
                try:worker.main()
                except BaseException as error:errors.append(repr(error))
            thread=threading.Thread(target=serve);thread.start();client=None
            try:
                # CPU fixture only: wait for the producer to save before one connection.
                until=time.time()+5
                while not result.exists()and time.time()<until:time.sleep(.01)
                time.sleep(.05)
                client=socket.create_connection(('127.0.0.1',port),timeout=5);client.settimeout(10)
                send(client,hello);assert recv(client)[0]['ready']
                full=bytes(512*35840*2);tail=bytes(504*35840*2)
                try:rows=pressure.run_rpc(client,send,recv,full,tail,lambda rows:None)
                except (EOFError,ConnectionError):pass
                client.close();client=None;thread.join(10);assert not thread.is_alive()
                q=json.loads(result.read_bytes());assert mode=='floor_failure' or not errors,(q,errors);assert q['completed']and state['runtime_closed']and state['context_closed']
                if mode=='success':
                    assert q['passed']and q['normal_protocol_close']and not errors
                    assert q['phase_counts']==dict(begin=1,append_prompt=64,seal=1,query=1,retire=1,close=1)
                    assert q['feature_bytes']==32760*35840*2 and state['context'].published==64
                else:
                    assert not q['passed']and errors and state['session'].poisoned
                    assert state['context'].published==0 and q['error']=="RuntimeError('Original4GiB draft memory floor')"
                cases.append(dict(name='actual_worker_main_TCP_RPC_plan_'+mode,passed=True,
                                  loader_boundary_recorded=True,actual_model_math=False,
                                  published_appends=state['context'].published))
            finally:
                if client is not None:client.close()
                thread.join(12)
                worker.Runtime,worker.verify_sources,worker.signal.signal,worker.signal.alarm,sys.argv=original
    out=dict(completed=True,passed=True,GPU=0,Native_dlopen=False,actual_CPU=os.environ.get('SLURM_JOB_ID','0')+'.'+os.environ.get('SLURM_STEP_ID','0'),
             source_manifest_SHA256=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
             cases=cases,actual_private_guard_worker_main_TCP_wire_and_RPC_plan=True,
             strict_loader_and_context_query_math_recording_boundaries=True,weights_opened=False,Model_qualified=False)
    print(json.dumps(out));return out


if __name__=='__main__':main()
