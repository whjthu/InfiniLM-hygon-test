"""One-card resident pinned DFlash2 worker for true target context, greedy only."""
import pathlib,json,os,time,hashlib,signal,gc,traceback,sys
sys.path.append("/refactor/incoming")
from draft_context_r9128 import ContextStore
from batch_wire_r9128 import listen,recv,send

def main():
    assert os.environ.get("KIMI_TRUE_DRAFT_BATCH_R9142")=="1", "r9142 defaultOFF"
    signal.alarm(5200)
    import torch
    from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
    torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    r=pathlib.Path('/refactor');inc=r/'incoming';out=r/'results'/('Kimi-modern-PP4-draft-r9145b-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');assert not out.exists()
    report={'status':'loading','GPU_count':1,'greedy_only':True,'cycles':[],'normal_close':False,'synthetic_features_only_for_warmup':True};loaded=None;sock=conn=None;store=ContextStore();cohort_epoch=0
    def save():
        p=out.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(out)
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();x=fn();torch.cuda.synchronize();return x,time.perf_counter()-start
    try:
        start=time.perf_counter();meta=json.loads((inc/'DFlash2-target-metadata-r9019.json').read_text());h=meta['tensors']['language_model.lm_head.weight'];shard=pathlib.Path(h['shard'].replace('/public/home/aclsylqidf/shared','/shared'))
        with shard.open('rb')as f:header=f.read(int.from_bytes(f.read(8),'little'));assert hashlib.sha256(header).hexdigest()==h['header_SHA256']
        mapping=torch.from_file(str(shard),shared=False,size=shard.stat().st_size,dtype=torch.uint8);a,b=h['data_offsets'];weight=mapping.narrow(0,h['payload_offset']+a,b-a).view(torch.bfloat16).reshape(h['shape']).to('cuda:0');head=ExternalTargetHead(weight,values_compatibility=None,proof={'actual_target_key':'language_model.lm_head.weight','shard_header_SHA256':h['header_SHA256']})
        checkpoint=r/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors';loaded=load_pinned_worker(checkpoint,head,enabled=True,device='cuda:0',source_dir=inc);model=loaded.worker.model;report['weights']=loaded.receipt;report['loading_s']=time.perf_counter()-start
        from batched_query_r9142 import query_batched
        from batch_frames_r9142 import receive_bodies
        def query(features,anchors,Ks):return query_batched(model,head,features,anchors,Ks,enabled=True)
        with torch.inference_mode():
            warm=torch.zeros((1,512,35840),device='cuda',dtype=torch.bfloat16)
            for _ in range(2):warmout=query([warm],[163584],[7])
            torch.cuda.synchronize();del warm,warmout;report['status']='ready';report['initial_peak_GPU_bytes']=torch.cuda.max_memory_allocated();save()
            sock=listen(os.environ['KIMI_DRAFT_HOST_R9046'],int(os.environ['KIMI_DRAFT_PORT_R9046']));sock.settimeout(5200);conn,address=sock.accept();conn.settimeout(5200);hello,body=recv(conn);assert hello=={'cmd':'hello','nonce':os.environ['KIMI_DRAFT_NONCE_R9046']}and not body;send(conn,{'ready':True,'greedy_only':True,'revision':'e77935fb4804e17eb55085bffd045eae1d779769','block_size':8,'proposals':7,'request_keys':True,'max_requests':32,'true_torch_batch':True,'chunked_features':True})
            expected=0
            while True:
                m,body=recv(conn);assert m['seq']==expected;expected+=1;cmd=m['cmd']
                if cmd=='close':
                    report['remaining_request_keys_before_close']=[list(k)for k in store.close()];send(conn,{'closed':True,'seq':m['seq']});break
                if cmd=='begin_cohort':
                    cohort_epoch=int(m['epoch']);store.begin(cohort_epoch);send(conn,{'begun':True,'epoch':cohort_epoch,'seq':m['seq']});continue
                assert int(m['epoch'])==cohort_epoch
                if cmd=='retire':
                    removed=store.retire(cohort_epoch,int(m['request']));send(conn,{'retired':True,'removed':removed,'epoch':cohort_epoch,'request':m['request'],'seq':m['seq']});continue
                assert cmd=='query_batch' and not body;items=m['requests'];store.validate_many(cohort_epoch,items);assert all(1<=q['K']<=7 for q in items);bodies=receive_bodies(conn,m);tick=time.perf_counter();parts=[torch.frombuffer(bytearray(b),dtype=torch.bfloat16).reshape(1,q['rows'],35840).to('cuda')for q,b in zip(items,bodies)];contexts=store.append_many(cohort_epoch,items,parts,lambda a,b:torch.cat([a,b],dim=1));torch.cuda.synchronize();append=time.perf_counter()-tick
                props,wall=timed(lambda:query(contexts,[q['anchor']for q in items],[q['K']for q in items]));assert bool(torch.isfinite(props).all());values=props.cpu().tolist();requests=[{'request':q['request'],'context_tokens':q['past']+q['rows'],'proposals':row[:q['K']]}for q,row in zip(items,values)];row={'seq':m['seq'],'epoch':cohort_epoch,'actual_model_batch_C':len(items),'requests':requests,'draft_batch_query_s':wall,'append_batch_H2D_s':append,'feature_frames':len(bodies),'execution':'ONE paddedragged model tensor batch; feature cache perrequest; no draftKV/noisyproposalcache'};report['cycles'].append(row);send(conn,row)
                if len(report['cycles'])%8==0:save()
        report.update(status='COMPLETED',normal_close=not report['remaining_request_keys_before_close'],draft_execution='true_padded_model_batch_C1_to32_shared_weights',final_peak_GPU_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if conn is not None:conn.close()
        if sock is not None:sock.close()
        if loaded is not None:loaded.close()
        save()
    print('KIMI_DRAFT_TRUEBATCH_PP4_R9145B_NORMAL',flush=True)
if __name__=='__main__':main()
