"""Remote-only canonical full-recompute proposals on real ordinary-A target taps.

No incremental worker calls, synthetic features, oracle proposal replacement,
target generation, serving acceptance or end-to-end speed claim.
"""
import argparse,pathlib,json,hashlib,time,os,signal,gc,traceback,statistics
p=argparse.ArgumentParser();p.add_argument('--target-result',required=True);p.add_argument('--out',required=True);args=p.parse_args()
signal.alarm(440)
import torch
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
from target_tap_sequence_r9024 import assemble_sequence
torch.set_num_threads(1);r=pathlib.Path('/refactor');inc=r/'incoming';source=pathlib.Path(args.target_result);out=pathlib.Path(args.out);assert not out.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report={'status':'RUNNING','scope':'actual ordinary A target features; trained canonical fullrecompute offline proposal agreement','target_acceptance_qualified':False,'serving_acceptance_claim':False,'spec_speedup_claim':False,'incremental_used':False,'synthetic_features':False,'raw_activations_stay_remote':True,'query_rows':8,'proposal_count':7,'tap_training_backend_or_AttnRes_convention_proven':False,'tap_initial_interpretation':'0-based layers[idx] module outputs; HF hidden_states list uses idx+1 due embedding entry','cases':[]};loaded=head=mapping=None;started=time.monotonic()
def save():
 t=out.with_suffix('.tmp');t.write_bytes(json.dumps(report,indent=2).encode());t.replace(out)
def timed(fn):
 torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize();return value,time.perf_counter()-t
def filehash(path,offset,length):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb')as f:
  f.seek(offset)
  while length:
   b=f.read(min(length,8<<20));assert b;h.update(b);length-=len(b)
 return h.hexdigest()
try:
 source_bytes=source.read_bytes();packet=json.loads(source_bytes);A=next(x for x in packet['crossover_arms']if x['arm']=='A');reference=A['anchor']+A['actual_ids'];assert len(reference)==33 and len(A['anchor'])==1 and len(A['actual_ids'])==32
 calls=[x for x in packet['crossover_calls']if x['arm']=='A'];assert len(calls)==33 and calls[0]['phase']=='prefill' and len(calls[0]['arrays']['input_ids'])==512
 owners=calls[-1]['owners'];stages=[]
 for node in [0,2]:
  owner=next(x for x in owners if x['node']==node);trace=owner['trace_after'];assert trace['arm']=='A' and trace['finite'] and trace['records'];stages.append(trace)
 report.update(target_result=str(source),target_result_snapshot_SHA256=hashlib.sha256(source_bytes).hexdigest(),ordinary_A_reference_IDs33=reference,target_completed_at_snapshot=packet.get('completed'),feature_receipt_epochs=[s['epoch']for s in stages],feature_receipt_nodes=[0,2]);save()
 assert torch.cuda.device_count()==1;torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();assert free>16<<30
 report.update(torch_version=torch.__version__,HIP_version=torch.version.hip,Torch_C_SHA256=sha(torch._C.__file__),GPU_count=1,initial_memory={'free':free,'total':total})
 # Reuse the exact979 strict loader/creator pins and actual target head shard.
 meta=json.loads((inc/'DFlash2-target-metadata-r9019.json').read_text());hmeta=meta['tensors']['language_model.lm_head.weight'];emeta=meta['tensors']['language_model.model.embed_tokens.weight'];shard=pathlib.Path(hmeta['shard'].replace('/public/home/aclsylqidf/shared','/shared'));assert sha('/shared/Kimi-K3/config.json')==meta['config_SHA256']
 with shard.open('rb')as f:header=f.read(int.from_bytes(f.read(8),'little'));assert hashlib.sha256(header).hexdigest()==hmeta['header_SHA256']
 mapping=torch.from_file(str(shard),shared=False,size=shard.stat().st_size,dtype=torch.uint8);a,b=hmeta['data_offsets'];weight=mapping.narrow(0,hmeta['payload_offset']+a,b-a).view(torch.bfloat16).reshape(hmeta['shape']).to('cuda:0');head=ExternalTargetHead(weight,values_compatibility=None,proof={'actual_target_key':'language_model.lm_head.weight','shard_header_SHA256':hmeta['header_SHA256']})
 checkpoint=r/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors'
 loaded,load_s=timed(lambda:load_pinned_worker(checkpoint,head,enabled=True,device='cuda:0',source_dir=inc));model=loaded.worker.model;report.update(load_s=load_s,strict_loader_receipt=loaded.receipt,source_SHA256=sha(__file__))
 dh=json.loads((inc/'actual-safetensors-header.json').read_text());emb=dh['embed_tokens.weight'];target_hash=filehash(shard,emeta['payload_offset']+emeta['data_offsets'][0],emeta['data_offsets'][1]-emeta['data_offsets'][0]);draft_hash=filehash(checkpoint,9816+emb['data_offsets'][0],emb['data_offsets'][1]-emb['data_offsets'][0]);assert target_hash==draft_hash
 report['actual_embedding_value_binding']={'target_SHA256':target_hash,'draft_SHA256':draft_hash,'equal':True};report['actual_target_head_value_SHA256']=filehash(shard,hmeta['payload_offset']+a,b-a);save()
 def canonical(features,anchor):
  length=features.shape[1];context=model.context_norm(model.context_proj(features));ids=torch.full((1,8),163592,dtype=torch.long,device='cuda');ids[0,0]=anchor;x=model.embed_tokens(ids);cp=torch.arange(length,device='cuda').view(1,-1);dp=torch.arange(length,length+8,device='cuda').view(1,-1)
  mask=(torch.abs(dp[:,:,None]-torch.arange(length+8,device='cuda')[None,None,:])<4096).unsqueeze(1)
  for i,layer in enumerate(model.layers):x=layer(x,context,dp,cp,mask if i<4 else None)
  hidden=model.final_norm(x);logits=head(hidden[:,1:]);previous=torch.tensor([anchor],device='cuda');proposals=[];selector_scores=[]
  for j in range(7):
   scores,candidates=model.candidate_selector.score_candidates(hidden[:,j+1],logits[:,j],previous);selector_scores.append(scores);previous=candidates.gather(-1,scores.argmax(-1,keepdim=True))[:,0];proposals.append(previous.clone())
  return hidden,logits,torch.stack(proposals,1),torch.stack(selector_scores,1)
 with torch.inference_mode():
  for N in [512,520,528,536]:
   host_features,positions=assemble_sequence(stages,0,N);assert list(host_features.shape)==[1,N,35840] and positions==list(range(N));features=host_features.to('cuda:0');anchor=reference[N-512];expected=reference[N-512+1:N-512+8];assert len(expected)==7
   (hidden,logits,proposals,selector_scores),wall=timed(lambda:canonical(features,anchor));assert bool(torch.isfinite(hidden).all()and torch.isfinite(logits).all()and torch.isfinite(selector_scores).all())
   cold_ids=proposals[0].cpu().tolist();warm,excluded_warm_s=timed(lambda:canonical(features,anchor));assert warm[2][0].cpu().tolist()==cold_ids;del warm
   warm_samples=[];ownrepeat=True
   for repeat_index in range(3):
    (rh,rl,rp,rs),sample_s=timed(lambda:canonical(features,anchor));assert bool(torch.isfinite(rh).all()and torch.isfinite(rl).all()and torch.isfinite(rs).all());ownrepeat=ownrepeat and rp[0].cpu().tolist()==cold_ids;warm_samples.append(sample_s);del rh,rl,rp,rs
   assert ownrepeat,'same-path proposal IDs changed across warm repetitions'
   actual=proposals[0].cpu().tolist();raw_top1=logits.argmax(-1)[0].cpu().tolist();matches=[x==y for x,y in zip(actual,expected)];prefix=0
   for match in matches:
    if not match:break
    prefix+=1
   row={'context_tokens':N,'feature_absolute_range':[0,N],'anchor_ID':anchor,'ordinary_A_anchor_index':N-512,'query_input_IDs':[anchor]+[163592]*7,'ordinary_A_true_next7_IDs':expected,'draft_selector_top1_IDs':actual,'draft_raw_head_top1_IDs':raw_top1,'position_matches':matches,'matched_prefix_0_to7':prefix,'finite':True,'query_including_context_projection_head_selector_wall_s':wall,'timing_excludes':['model/head loading','strict file SHA validation','feature file reading/assembly','feature CPU_to_GPU transfer','CPU result materialization'],'timing_includes':['context projection/norm fullrecompute','all5 canonical layers','8query final_norm','7row actual target head','candidate_selector sequential predecessor scoring'],'cold_first_case':N==512,'serving_acceptance_claim':False}
   row.update(cold_query_wall_s=wall,excluded_warmup_query_wall_s=excluded_warm_s,warm_query_wall_samples_s=warm_samples,warm_query_wall_median_s=statistics.median(warm_samples),ownrepeat_proposal_IDs_consistent=ownrepeat,logits_bitexact_required=False)
   report['cases'].append(row);save();del host_features,features,hidden,logits,proposals,selector_scores
 report.update(status='ACTUAL_A_OFFLINE_CANONICAL_PROPOSALS_COMPLETE',actual_peak_allocated=torch.cuda.max_memory_allocated())
except BaseException as e:report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
 if loaded is not None:loaded.close()
 if head is not None:head.close()
 loaded=head=mapping=None
 for name in ['model','weight','features','host_features','hidden','logits','proposals','selector_scores']:globals().pop(name,None)
 gc.collect()
 if torch.cuda.is_initialized():torch.cuda.synchronize();torch.cuda.empty_cache()
 report.update(shutdown_returncode=0,elapsed_s=time.monotonic()-started);save();print('DFLASH2_ACTUAL_A_NORMAL',flush=True)
