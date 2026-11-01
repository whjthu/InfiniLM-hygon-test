"""Actual Torch BF16 projection and bounded socket RPC, no checkpoint or GPU."""
import datetime,hashlib,json,os,pathlib,socket,threading,traceback
import torch
from long_context_r323 import LongCommittedFeatureContext
from long_rpc import DraftSession,TorchFeatureCodec,serve_connection
from batch_wire_r9128 import send,recv

torch.set_num_threads(1)


class Norm(torch.nn.Module):
    variance_epsilon=1e-6
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(8))
    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.variance_epsilon)).to(x.dtype)*self.weight


class Model(torch.nn.Module):
    def __init__(self,overflow=False):
        super().__init__();self.context_proj=torch.nn.Linear(40,8,bias=False);self.context_norm=Norm()
        if overflow:
            with torch.no_grad():self.context_proj.weight.fill_(3e38)


def feature_bytes(part):return part.contiguous().view(torch.uint16).numpy().tobytes()


def connected(overflow=False):
    client,server=socket.socketpair();client.settimeout(20);server.settimeout(20);state={};ready=threading.Event()
    def target():
        try:
            torch.manual_seed(333);model=Model(overflow).to(dtype=torch.bfloat16).eval()
            context=LongCommittedFeatureContext(model,'CPU333-synthetic',enabled=True)
            session=DraftSession(context,None,TorchFeatureCodec('cpu',raw_dim=40),enabled=True)
            state.update(model=model,context=context,session=session);ready.set()
            with torch.inference_mode():serve_connection(server,session)
        except BaseException as error:state['error']=repr(error);ready.set()
        finally:server.close()
    thread=threading.Thread(target=target);thread.start();assert ready.wait(10)and 'session'in state
    return client,thread,state


def main():
    D=pathlib.Path(__file__).resolve().parent;O=pathlib.Path('/refactor/candidates/Kimi-long-draft-RPC-CPU-r333');O.mkdir(exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'passed':False,
       'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,
       'GPU':0,'checkpoint_weights':0,'target_Native':0,'production_HTTP':0,'cases':[]}
    try:
        assert sha(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_RPC_CPU_MANIFEST_SHA']
        mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in mf['files_SHA256'].items())
        q['source_manifest_SHA256']=sha(D/'SOURCE_MANIFEST.json');assert not torch.cuda.is_initialized()
        client,thread,state=connected();seq=0;capture=0
        def request(cmd,body=b'',**kw):
            nonlocal seq
            meta=dict(cmd=cmd,seq=seq)
            if cmd!='close':meta.update(epoch=1,request=1)
            meta.update(kw);send(client,meta,body);r,b=recv(client);assert not b and r['seq']==seq;seq+=1;return r
        request('begin',prompt_tokens=1024)
        parts=[torch.randn(1,512,40,dtype=torch.bfloat16)for _ in range(2)]
        for i,part in enumerate(parts):
            capture+=1;r=request('append_prompt',feature_bytes(part),past=i*512,rows=512,capture_epoch=capture)
            assert r['past_after']==(i+1)*512
        request('seal',past=1024)
        part=torch.randn(1,2,40,dtype=torch.bfloat16);capture+=1
        proof=dict(capture_epoch=capture,past_before=1024,past_after=1026,accepted_inputs=2,complete=True)
        request('append_committed',feature_bytes(part),past=1024,rows=2,capture_epoch=capture,target_commit=proof)
        with torch.inference_mode():
            model=state['model'];expected=torch.cat([model.context_norm(model.context_proj(p[0])).unsqueeze(0)for p in [*parts,part]],1)
        torch.testing.assert_close(state['context'].projected,expected,rtol=0,atol=0)
        assert torch.equal(state['context'].raw,torch.cat([*parts,part],1))
        request('retire');request('close');client.close();thread.join(10)
        assert not thread.is_alive()and state['context'].raw is None and 'error'not in state
        q['cases'].append('real_BF16_socket512x2_seal_committed2_projection_RMS0_retire_close')
        # A real long consumer exercises AR tail without a fake draft query/model.
        client,thread,state=connected();seq=0;capture=0;request('begin',prompt_tokens=32761)
        for past in range(0,32761,512):
            n=min(512,32761-past);capture+=1
            request('append_prompt',feature_bytes(torch.zeros(1,n,40,dtype=torch.bfloat16)),past=past,rows=n,capture_epoch=capture)
        request('seal',past=32761);r=request('query',past=32761,anchor=163584,K=7)
        assert r['mode']=='ar_tail'and r['proposals']==[]
        client.close();thread.join(10);assert not thread.is_alive()and state['context'].raw is None
        q['cases'].append('real32761context_query_ARtail_and_EOF_cleanup')
        for name,value,overflow in [('NaN_frame',float('nan'),False),('Inf_frame',float('inf'),False),
                                    ('finite_input_projected_overflow',1.,True)]:
            client,thread,state=connected(overflow);seq=0;request('begin',prompt_tokens=1)
            send(client,dict(cmd='append_prompt',seq=1,epoch=1,request=1,past=0,rows=1,capture_epoch=1),
                 feature_bytes(torch.full((1,1,40),value,dtype=torch.bfloat16)))
            try:recv(client)
            except EOFError:pass
            else:raise AssertionError('Nonfinite frame/projection accepted')
            client.close();thread.join(10)
            assert not thread.is_alive()and state['session'].poisoned and state['context'].length==0
            assert state['context'].raw is None and 'Nonfinite'in state['error']
            q['cases'].append(name+'_reject_before_publication')
        # Production feature width/frame bound, no giant projection/checkpoint.
        client,server=socket.socketpair();client.settimeout(30);server.settimeout(30);seen={}
        def geometry():
            try:
                meta,body=recv(server);assert meta=={'rows':512,'raw_dim':35840}
                p=TorchFeatureCodec('cpu').decode(body,512)
                assert p.shape==(1,512,35840)and p.dtype==torch.bfloat16
                seen['bytes']=len(body);send(server,{'finite':True,'bytes':len(body)})
            except BaseException as e:seen['error']=repr(e)
            finally:server.close()
        thread=threading.Thread(target=geometry);thread.start()
        send(client,{'rows':512,'raw_dim':35840},bytes(512*35840*2));assert recv(client)[0]['finite']
        client.close();thread.join(10);assert not thread.is_alive()and 'error'not in seen
        q['cases'].append('production35840col_512row_36700160byte_socket_Torch_decode_finite')
        q.update(passed=len(q['cases'])==6,actual_Torch_BF16_projection=True,actual_socket_transport=True,
            no_actual_draft_query_or_real_capture_proof=True,synthetic_projection_geometry='40to8',
            production_frame_bytes=seen['bytes'],GPU_runtime_initialized=torch.cuda.is_initialized())
        assert not q['GPU_runtime_initialized']
    except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        q['completed']=True;(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
    return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())
