"""Small actual BF16 file only. Never opens a real checkpoint weight payload."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.environ.get('ROCR_VISIBLE_DEVICES')==''
import pathlib,json,hashlib,tempfile,time,traceback,copy
import torch
import torch.nn.functional as F
from dflash2_strict_loader_r9007 import _Spec,_validate,_load_verified,creator_key,ExternalTargetHead,load_pinned_worker
from dflash2_creator_factory_r9007 import make_factory
torch.set_num_threads(1);assert not torch.cuda.is_initialized();root=pathlib.Path('/refactor');inc=root/'incoming';out=root/'results/DFlash2-loader-CPU-r9008.json';d={'completed':False,'CPU_only':True,'trained_weight_payload_opened':False,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};start=time.monotonic()
try:
 cfg=json.loads((inc/'config.json').read_text());cfg.update(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=16,kv_lora_rank=12,qk_nope_head_dim=8,qk_rope_head_dim=8,v_head_dim=8,vocab_size=32,max_position_embeddings=128,eos_token_id=2);cfg['dflash_config']['mask_token_id']=4
 sentinel=__import__('sys').modules.get('torchspec.models.draft.dspark');factory=make_factory(cfg,inc);assert __import__('sys').modules.get('torchspec.models.draft.dspark')is sentinel;old=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
 try:model=factory()
 finally:torch.set_default_dtype(old)
 torch.manual_seed(9007)
 with torch.no_grad():
  for name,param in model.named_parameters():
   param.copy_(torch.randn_like(param)*.025)
   if name.endswith('norm.weight')or'layernorm.weight'in name:param.add_(1)
   if name.endswith('base_kernel'):param.zero_();param[:,0]=1
 serving={};reverse={'context_proj.weight':'fc.weight','context_norm.weight':'hidden_norm.weight','final_norm.weight':'norm.weight'}
 for key,param in model.state_dict().items():serving[reverse.get(key,key)]=param
 assert len(serving)==87
 with tempfile.TemporaryDirectory(prefix='dflash2-loader-tiny-',dir='/dev/shm')as tmp:
  tmp=pathlib.Path(tmp);file=tmp/'model.safetensors';header={};parts=[];offset=0
  for key,value in sorted(serving.items()):
   b=value.contiguous().view(torch.uint8).numpy().tobytes();header[key]={'dtype':'BF16','shape':list(value.shape),'data_offsets':[offset,offset+len(b)]};parts.append(b);offset+=len(b)
  raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8);blob=len(raw).to_bytes(8,'little')+raw+b''.join(parts);assert len(blob)<2<<20;file.write_bytes(blob);spec=_Spec(len(blob),hashlib.sha256(blob).hexdigest(),hashlib.sha256(raw).hexdigest(),header)
  weight=torch.randn((32,32),dtype=torch.bfloat16)*.03;head=ExternalTargetHead(weight,values_compatibility=None);loaded=_load_verified(file,spec,factory,head,enabled=True,eos_ids=(2,));actual=dict(loaded.worker.model.named_parameters());assert set(actual)==set(model.state_dict());assert all(torch.equal(actual[k],v)for k,v in model.state_dict().items());assert all(v.dtype==torch.bfloat16 for v in actual.values());assert loaded.receipt['external_head_values_compatibility']is None
  # All CPU parameters share one mapped storage, distinct from source fixture.
  ptrs={v.untyped_storage().data_ptr()for v in actual.values()};assert len(ptrs)==1 and next(iter(ptrs))==loaded._storage.untyped_storage().data_ptr()
  context=torch.randn((1,9,160),dtype=torch.bfloat16)*.2;loaded.worker.append_context(context,torch.arange(9).view(1,-1));a=loaded.worker.speculate(3);b=loaded.worker.speculate(3);assert a['hidden'].shape==(1,8,32)and a['tokens'].shape==(1,7)and torch.isfinite(a['hidden']).all()and torch.equal(a['hidden'],b['hidden']);loaded.worker.crop_context(7);assert loaded.worker.positions.shape[1]==7;loaded.close();del actual;__import__('gc').collect();assert not head.closed and torch.equal(head.weight,weight),'borrowed head must not be closed';assert loaded._storage is None
  failures=[]
  def reject(name,callback):
   try:callback()
   except(ValueError,RuntimeError,AssertionError)as e:failures.append({'case':name,'error':str(e)});return
   raise AssertionError('invalid fixture accepted '+name)
  partial=tmp/'model.safetensors.partial';partial.write_bytes(blob);reject('partial_even_complete_length',lambda:_validate(partial,spec));short=tmp/'short.safetensors';short.write_bytes(blob[:-1]);reject('short_length',lambda:_validate(short,spec));extra=tmp/'extra.safetensors';extra.write_bytes(blob+b'x');reject('extra_length',lambda:_validate(extra,spec));corrupt=tmp/'corrupt.safetensors';corrupt.write_bytes(blob[:-1]+bytes([blob[-1]^1]));reject('wrong_full_SHA',lambda:_validate(corrupt,spec));reject('missing_external_head',lambda:_load_verified(file,spec,factory,None,enabled=True));reject('default_OFF',lambda:_load_verified(file,spec,factory,head));reject('production_entry_rejects_tiny',lambda:load_pinned_worker(file,head,enabled=True,source_dir=inc))
  bad=copy.deepcopy(header);bad['fc.weight']['dtype']='F16';reject('wrong_dtype_schema',lambda:_validate(file,_Spec(spec.file_bytes,spec.sha256,spec.header_sha256,bad)));bad=copy.deepcopy(header);bad['norm.weight']['shape']=[31];reject('wrong_shape_schema',lambda:_validate(file,_Spec(spec.file_bytes,spec.sha256,spec.header_sha256,bad)));bad=copy.deepcopy(header);del bad['norm.weight'];reject('missing_key_schema',lambda:_validate(file,_Spec(spec.file_bytes,spec.sha256,spec.header_sha256,bad)))
  # Full recompute through the source layers at BF16, same operands and masks.
  with torch.inference_mode():
   ctx=model.context_norm(model.context_proj(context));ids=torch.tensor([[3,4,4,4,4,4,4,4]]);x=model.embed_tokens(ids);cp=torch.arange(9).view(1,-1);dp=torch.arange(9,17).view(1,-1);mask=(torch.abs(dp[:,:,None]-torch.arange(17)[None,None,:])<4096).unsqueeze(1)
   for i,layer in enumerate(model.layers):x=layer(x,ctx,dp,cp,mask if i<4 else None)
   expected=model.final_norm(x);error=float((expected.float()-a['hidden'].float()).abs().max());assert error<.02,error
  d.update(completed=True,fixture_bytes=len(blob),fixture_BF16_tensors=87,creator_key_mapping={k:creator_key(k)for k in ['fc.weight','hidden_norm.weight','norm.weight']},all_parameters_exact=True,single_mapped_storage=True,worker_forward_finite=True,ownrepeat_exact=True,worker_vs_full_BF16_maxabs=error,missing_values_compatibility_not_promoted=True,borrowed_head_lifetime=True,bad_files_rejected=failures,receipt=loaded.receipt,trained_GPU_qualified=False,GPU_initialized=torch.cuda.is_initialized())
except BaseException as e:d.update(completed=False,error=repr(e),traceback=traceback.format_exc());raise
finally:d['elapsed_s']=time.monotonic()-start;assert not out.exists();out.write_bytes((json.dumps(d,indent=2)+'\n').encode());print(json.dumps(d))
