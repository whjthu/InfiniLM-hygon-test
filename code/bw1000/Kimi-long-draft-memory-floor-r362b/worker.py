"""Default-OFF, one authenticated C1 long-draft connection. No target launcher."""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

RUNTIME_SHA='358016b6b9e334817d2d4929ed613d67548dcd6764a20f2372b7748be3c99a2e'
TRAINED_SHA='3c7c6cfbbec4be2a217df6d4fdeb2710123d5bb68e391130702571514ba84a07'
TARGET_NATIVE_SHA='4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505'
PROTOCOL='Kimi-C1-long-r334'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_hello(nonce):
    if not isinstance(nonce,str) or len(nonce)<32:raise ValueError('Explicit nonce required')
    return dict(cmd='hello',nonce=nonce,protocol=PROTOCOL,runtime_manifest_SHA256=RUNTIME_SHA,
                target_Native_SHA256=TARGET_NATIVE_SHA)


def serve_authenticated(listener,hello,runtime_factory,report,save):
    """Authenticate before touching weights/CUDA; always close the sole session."""
    from batch_wire_r9128 import recv,send
    conn=runtime=None
    try:
        conn,_=listener.accept();conn.settimeout(120)
        meta,body=recv(conn)
        if meta!=hello or body:raise ValueError('Exact long draft hello/nonce/source identity required')
        report['authenticated']=True;save()
        runtime=runtime_factory()
        send(conn,dict(ready=True,protocol=PROTOCOL,C=1,context=32768,block_size=8,
            proposals=7,prompt_append_max=512,committed_append_max=8,
            runtime_manifest_SHA256=RUNTIME_SHA,target_Native_SHA256=TARGET_NATIVE_SHA))
        report['ready']=True;save()
        while not runtime.session.closed:
            meta,body=recv(conn)
            if report.get('commands',0)>=4096:raise ValueError('Bounded command count exceeded')
            answer=runtime.session.dispatch(meta,body)
            report['commands']=report.get('commands',0)+1
            report['feature_bytes']=report.get('feature_bytes',0)+len(body)
            counts=report.setdefault('phase_counts',{});cmd=meta['cmd'];counts[cmd]=counts.get(cmd,0)+1
            report['handler_wall_s']=report.get('handler_wall_s',0.)+answer.get('handler_wall_s',0.)
            send(conn,answer)
            if cmd in ['retire','close']:save()
        report['normal_protocol_close']=True
    finally:
        try:
            if runtime is not None:runtime.close()
        finally:
            try:
                if conn is not None:conn.close()
            finally:listener.close()


def verify_sources(directory,root,expected):
    assert sha(directory/'SOURCE_MANIFEST.json')==expected
    m=json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())
    assert all(sha(directory/n)==h for n,h in m['files_SHA256'].items())
    runtime=root/'incoming/Kimi-long-draft-RPC-CPU-r333'
    assert sha(runtime/'SOURCE_MANIFEST.json')==RUNTIME_SHA
    rm=json.loads((runtime/'SOURCE_MANIFEST.json').read_bytes())
    assert all(sha(runtime/n)==h for n,h in rm['files_SHA256'].items())
    incoming=root/'incoming';trained=incoming/'DFlash2-trained-manifest-r9019.json'
    assert sha(trained)==TRAINED_SHA
    tm=json.loads(trained.read_bytes())
    assert all(sha(incoming/n)==h for n,h in tm['incoming_source_SHA256'].items())
    assert sha(incoming/'DFlash2-target-metadata-r9019.json')==m['target_metadata_SHA256']
    receipt=root/'candidates/Kimi-long-draft-RPC-CPU-r333/CPU-result.json'
    assert sha(receipt)==m['CPU643_receipt_SHA256']
    q=json.loads(receipt.read_bytes())
    assert q['completed']and q['passed']and q['actual_CPU']=='910407.643'
    assert q['source_manifest_SHA256']==RUNTIME_SHA and not q['GPU_runtime_initialized']
    return runtime,incoming


class Runtime:
    def __init__(self,root,runtime,incoming,report):
        self.loaded=self.head=self.session=self.mapping=self.weight=None
        self.closed=False
        sys.path[:0]=[str(runtime),str(incoming)]
        import torch
        from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
        from long_context_r323 import LongCommittedFeatureContext
        import long_rpc
        assert Path(long_rpc.__file__).resolve()==(runtime/'long_rpc.py').resolve()
        torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
        prop=torch.cuda.get_device_properties(0)
        report["actual_device"]={"visible_count":1,"logical_device":0,"name":prop.name,"total_memory":prop.total_memory,"gcnArchName":getattr(prop,"gcnArchName",None)}
        assert torch.cuda.mem_get_info()[0]>16<<30
        try:
            tick=time.perf_counter()
            meta=json.loads((incoming/'DFlash2-target-metadata-r9019.json').read_bytes())
            assert sha(Path('/shared/Kimi-K3/config.json'))==meta['config_SHA256']
            h=meta['tensors']['language_model.lm_head.weight'];shard=Path(h['shard'].replace('/public/home/aclsylqidf/shared','/shared'))
            with shard.open('rb')as f:
                n=int.from_bytes(f.read(8),'little');assert hashlib.sha256(f.read(n)).hexdigest()==h['header_SHA256']
            self.mapping=torch.from_file(str(shard),shared=False,size=shard.stat().st_size,dtype=torch.uint8)
            a,b=h['data_offsets'];self.weight=self.mapping.narrow(0,h['payload_offset']+a,b-a).view(torch.bfloat16).reshape(h['shape']).to('cuda:0')
            self.head=ExternalTargetHead(self.weight,values_compatibility=None,proof={'actual_target_key':'language_model.lm_head.weight','shard_header_SHA256':h['header_SHA256']})
            checkpoint=root/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors'
            self.loaded=load_pinned_worker(checkpoint,self.head,enabled=True,device='cuda:0',source_dir=incoming)
            model=self.loaded.worker.model
            assert len(model.layers)==5 and all(l.self_attn.max_position_embeddings==32768 for l in model.layers)
            assert tuple(model.context_proj.weight.shape)==(7168,35840)
            context=LongCommittedFeatureContext(model,self.loaded.receipt['full_SHA256'],enabled=True)
            from memory_floor import guarded_codec
            GuardedCodec=guarded_codec(long_rpc.TorchFeatureCodec,torch,report)
            self.session=long_rpc.DraftSession(context,self.head,GuardedCodec('cuda:0'),enabled=True)
            torch.cuda.synchronize();report.update(loader_receipt=self.loaded.receipt,load_wall_s=time.perf_counter()-tick)
        except BaseException:
            self.close();raise

    def close(self):
        if self.closed:return
        self.closed=True
        errors=[]
        for obj in [self.session,self.loaded,self.head]:
            if obj is not None:
                try:obj.close()
                except BaseException as e:errors.append(repr(e))
        self.session=self.loaded=self.head=self.weight=self.mapping=None
        gc.collect()
        import torch
        torch.cuda.empty_cache();torch.cuda.synchronize()
        if errors:raise RuntimeError('Owned runtime cleanup failed: '+repr(errors))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('INFINILM_KIMI_LONG_RPC_R334')!='1':raise RuntimeError('Long worker default OFF')
    left=args.deadline_unix-time.time()
    if not math.isfinite(left)or not 30<left<=29400 or not 1<=args.port<=65535:raise ValueError('Concrete bounded deadline/port required')
    hello=expected_hello(os.environ['KIMI_DRAFT_NONCE_R9046'])
    runtime,incoming=verify_sources(Path(__file__).resolve().parent,Path('/refactor'),args.manifest_sha)
    sys.path.insert(0,str(runtime));from batch_wire_r9128 import listen
    if args.output.exists():raise ValueError('Do not overwrite a worker result')
    q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'passed':False,
       'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
       'authenticated':False,'ready':False,'normal_protocol_close':False,'GPU_count':1,
       'source_manifest_SHA256':args.manifest_sha,'runtime_manifest_SHA256':RUNTIME_SHA,
       'nonce_SHA256':hashlib.sha256(hello['nonce'].encode()).hexdigest(),'no_retry_reload':True}
    def save():
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2)+'\n');temp.replace(args.output)
    def expired(*_):raise TimeoutError('Original worker absolute deadline exceeded')
    left=args.deadline_unix-time.time()
    if left<=30:raise TimeoutError('Original budget exhausted during source verification')
    signal.signal(signal.SIGALRM,expired);signal.alarm(int(left))
    def create_runtime():
        if args.deadline_unix-time.time()<=30:raise TimeoutError('Original budget exhausted before checkpoint loading')
        return Runtime(Path('/refactor'),runtime,incoming,q)
    listener=None;save()
    try:
        listener=listen(args.host,args.port);listener.settimeout(min(left,120))
        # Inference mode must cover both model construction and serving, as601.
        import torch
        with torch.inference_mode():
            serve_authenticated(listener,hello,create_runtime,q,save)
        q['passed']=q['normal_protocol_close']
    except BaseException as e:q['error']=repr(e);raise
    finally:
        if listener is not None:listener.close()
        q['completed']=True;save()
    return 0


if __name__=='__main__':raise SystemExit(main())
