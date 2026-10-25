"""One-card resident pinned DFlash2 worker for true target context, greedy only."""
import pathlib,json,os,time,hashlib,signal,gc,traceback,sys
sys.path.append("/refactor/incoming")
from wire import listen,recv,send

def main():
    signal.alarm(2640)
    import torch
    from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
    torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    r=pathlib.Path('/refactor');inc=r/'incoming';out=r/'results'/('Kimi-draft-live-r9046-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');assert not out.exists()
    report={'status':'loading','GPU_count':1,'greedy_only':True,'cycles':[],'normal_close':False,'synthetic_features_only_for_warmup':True};loaded=None;sock=conn=None;context=None
    def save():
        p=out.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(out)
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();x=fn();torch.cuda.synchronize();return x,time.perf_counter()-start
    try:
        start=time.perf_counter();meta=json.loads((inc/'DFlash2-target-metadata-r9019.json').read_text());h=meta['tensors']['language_model.lm_head.weight'];shard=pathlib.Path(h['shard'].replace('/public/home/aclsylqidf/shared','/shared'))
        with shard.open('rb')as f:header=f.read(int.from_bytes(f.read(8),'little'));assert hashlib.sha256(header).hexdigest()==h['header_SHA256']
        mapping=torch.from_file(str(shard),shared=False,size=shard.stat().st_size,dtype=torch.uint8);a,b=h['data_offsets'];weight=mapping.narrow(0,h['payload_offset']+a,b-a).view(torch.bfloat16).reshape(h['shape']).to('cuda:0');head=ExternalTargetHead(weight,values_compatibility=None,proof={'actual_target_key':'language_model.lm_head.weight','shard_header_SHA256':h['header_SHA256']})
        checkpoint=r/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors';loaded=load_pinned_worker(checkpoint,head,enabled=True,device='cuda:0',source_dir=inc);model=loaded.worker.model;report['weights']=loaded.receipt;report['loading_s']=time.perf_counter()-start
        def query(features,anchor):
            length=features.shape[1];ctx=model.context_norm(model.context_proj(features));ids=torch.full((1,8),163592,dtype=torch.long,device='cuda');ids[0,0]=anchor;x=model.embed_tokens(ids);cp=torch.arange(length,device='cuda').view(1,-1);dp=torch.arange(length,length+8,device='cuda').view(1,-1);mask=(torch.abs(dp[:,:,None]-torch.arange(length+8,device='cuda')[None,None,:])<4096).unsqueeze(1)
            for i,layer in enumerate(model.layers):x=layer(x,ctx,dp,cp,mask if i<4 else None)
            hidden=model.final_norm(x);logits=head(hidden[:,1:]);previous=torch.tensor([anchor],device='cuda');props=[]
            for j in range(7):
                scores,candidates=model.candidate_selector.score_candidates(hidden[:,j+1],logits[:,j],previous);previous=candidates.gather(-1,scores.argmax(-1,keepdim=True))[:,0];props.append(previous.clone())
            return torch.stack(props,1)
        with torch.inference_mode():
            warm=torch.zeros((1,512,35840),device='cuda',dtype=torch.bfloat16)
            for _ in range(2):warmout=query(warm,163584)
            torch.cuda.synchronize();del warm,warmout;report['status']='ready';report['initial_peak_GPU_bytes']=torch.cuda.max_memory_allocated();save()
            sock=listen(os.environ['KIMI_DRAFT_HOST_R9046'],int(os.environ['KIMI_DRAFT_PORT_R9046']));sock.settimeout(2600);conn,address=sock.accept();conn.settimeout(180);hello,body=recv(conn);assert hello=={'cmd':'hello','nonce':os.environ['KIMI_DRAFT_NONCE_R9046']}and not body;send(conn,{'ready':True,'greedy_only':True,'revision':'e77935fb4804e17eb55085bffd045eae1d779769','block_size':8,'proposals':7})
            expected=0
            while True:
                m,body=recv(conn);assert m['seq']==expected;expected+=1;cmd=m['cmd']
                if cmd=='close':send(conn,{'closed':True,'seq':m['seq']});break
                if cmd=='reset':context=None;send(conn,{'reset':True,'seq':m['seq']});continue
                assert cmd=='query'and 0<int(m['K'])<=7 and 1<=m['rows']<=512 and len(body)==m['rows']*35840*2
                old=0 if context is None else context.shape[1];assert m['past']==old and old+m['rows']<=1024
                tick=time.perf_counter();host=torch.frombuffer(bytearray(body),dtype=torch.bfloat16).reshape(1,m['rows'],35840);part=host.to('cuda');context=part if context is None else torch.cat([context,part],dim=1);torch.cuda.synchronize();append=time.perf_counter()-tick
                props,wall=timed(lambda:query(context,int(m['anchor'])));assert bool(torch.isfinite(props).all());ids=props[0].cpu().tolist();row={'seq':m['seq'],'context_tokens':context.shape[1],'append_rows':m['rows'],'append_H2D_s':append,'draft_query_s':wall,'proposals':ids[:m['K']],'K':m['K']};report['cycles'].append(row);send(conn,{**row,'greedy_only':True});
                if len(report['cycles'])%8==0:save()
        report.update(status='COMPLETED',normal_close=True,final_peak_GPU_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if conn is not None:conn.close()
        if sock is not None:sock.close()
        if loaded is not None:loaded.close()
        save()
    print('KIMI_DRAFT_LIVE_R9046_NORMAL',flush=True)
if __name__=='__main__':main()
