"""Default-OFF TorchSpec DFlash2 worker; original query-block math preserved."""
import threading
import torch
import torch.nn.functional as F

class IncrementalDFlash2:
 def __init__(self,model,lm_head,*,window=4096,enabled=False,eos_ids=()):
  if not enabled:raise RuntimeError('DFlash2 incremental defaultOFF')
  assert len(model.layers)==5 and float(getattr(getattr(model,'config',None),'input_embedding_scale',1.))==1.,'pinned five-layer scale1 architecture required'
  self.model=model;self.head=lm_head;self.window=window;self.owner=threading.get_ident();self.closed=False;self.positions=None;self.kv=[];self.last_query_tails=[];self.eos_ids=frozenset(eos_ids)
  self.proj=getattr(model,'context_proj',None)or model.fc;self.ctxnorm=getattr(model,'context_norm',None)or model.hidden_norm;self.norm=getattr(model,'final_norm',None)or model.norm
 def check(self):
  assert not self.closed and threading.get_ident()==self.owner,'closed/foreign owner'
 def rope(self,a,x,pos):
  cos,sin=a.rotary_emb(x,seq_len=int(pos.max())+1);cos=cos[0,0,pos,:x.shape[-1]//2].repeat_interleave(2,-1).unsqueeze(1);sin=sin[0,0,pos,:x.shape[-1]//2].repeat_interleave(2,-1).unsqueeze(1)
  rotated=torch.stack((-x[...,1::2],x[...,::2]),-1).flatten(-2);return x*cos+rotated*sin
 def project_kv(self,a,x,pos):
  b,t,_=x.shape;combined=a.kv_a_proj_with_mqa(x);latent,kr=torch.split(combined,[a.kv_lora_rank,a.qk_rope_head_dim],-1);kv=a.kv_b_proj(a.kv_a_layernorm(latent)).view(b,t,a.num_heads,a.qk_nope_head_dim+a.v_head_dim).transpose(1,2);kn,v=torch.split(kv,[a.qk_nope_head_dim,a.v_head_dim],-1);kr=self.rope(a,kr.unsqueeze(1),pos).expand(-1,a.num_heads,-1,-1);return torch.cat((kn,kr),-1),v
 @torch.inference_mode()
 def append_context(self,features5tap,positions):
  self.check()
  if isinstance(features5tap,(tuple,list)):assert len(features5tap)==5
  features=torch.cat(features5tap,-1)if isinstance(features5tap,(tuple,list))else features5tap
  assert features.shape[-1]==self.proj.weight.shape[1]
  assert positions.ndim==2 and positions.shape==features.shape[:2] and features.shape[1]>0
  start=0 if self.positions is None else self.positions.shape[1];assert torch.equal(positions,torch.arange(start,start+features.shape[1],device=positions.device).view(1,-1))
  context=self.ctxnorm(self.proj(features));parts=[self.project_kv(layer.self_attn,context,positions)for layer in self.model.layers]
  # Source layers always use the same projected target context, not prior-layer
  # attention outputs. Thus these per-layer K/V are immutable across queries.
  if self.positions is None:self.kv=parts;self.positions=positions.clone()
  else:self.kv=[(torch.cat((old[0],new[0]),2),torch.cat((old[1],new[1]),2))for old,new in zip(self.kv,parts)];self.positions=torch.cat((self.positions,positions),1)
  return self.positions.shape[1]
 @torch.inference_mode()
 def speculate(self,anchor):
  self.check();assert self.positions is not None and self.positions.shape[0]==1
  if int(anchor)in self.eos_ids:return {'terminal':True,'tokens':torch.empty((1,0),dtype=torch.long,device=self.positions.device),'context_length':self.positions.shape[1],'query_KV_committed':False}
  start=self.positions.shape[1];device=self.positions.device;query_ids=torch.full((1,8),int(getattr(self.model,'mask_token_id',getattr(getattr(self.model,'config',None),'mask_token_id',4))),device=device,dtype=torch.long);query_ids[:,0]=int(anchor);pos=torch.arange(start,start+8,device=device).view(1,-1);x=self.model.embed_tokens(query_ids);self.last_query_tails=[]
  for i,layer in enumerate(self.model.layers):
   residual=x;y=layer.input_layernorm(x);y,attention_kernel=layer.attention_conv.prepare(y);a=layer.self_attn;b,t,_=y.shape;q=a.q_b_proj(a.q_a_layernorm(a.q_a_proj(y))).view(b,t,a.num_heads,a.qk_head_dim).transpose(1,2);qn,qr=torch.split(q,[a.qk_nope_head_dim,a.qk_rope_head_dim],-1);q=torch.cat((qn,self.rope(a,qr,pos)),-1);qk,qv=self.project_kv(a,y,pos);key=torch.cat((self.kv[i][0],qk),2);value=torch.cat((self.kv[i][1],qv),2)
   keypos=torch.cat((self.positions,pos),1);mask=(torch.abs(pos[:,:,None]-keypos[:,None,:])<self.window).unsqueeze(1)if i<4 else None
   z=F.scaled_dot_product_attention(q,key,value,attn_mask=mask,dropout_p=0.,is_causal=False,scale=a.softmax_scale).transpose(1,2).reshape(b,t,a.num_heads*a.v_head_dim);z=a.o_proj(z);z=layer.attention_conv.finish(z,attention_kernel);x=residual+z
   residual=x;y=layer.post_attention_layernorm(x);y,mlp_kernel=layer.mlp_conv.prepare(y);z=layer.mlp(y);x=residual+layer.mlp_conv.finish(z,mlp_kernel)
   # Original GroupedConv is causal *inside each eight-query block* and resets
   # at block boundaries. These tails are diagnostic/transient, never context.
   self.last_query_tails.append((attention_kernel[:,-1:].clone(),mlp_kernel[:,-1:].clone()))
  hidden=self.norm(x);proposal_hidden=hidden[:,1:];logits=self.head(proposal_hidden);previous=torch.tensor([anchor],device=device,dtype=torch.long);tokens=[]
  for j in range(7):
   scores,ids=self.model.candidate_selector.score_candidates(proposal_hidden[:,j],logits[:,j],previous);previous=ids.gather(-1,scores.argmax(-1,keepdim=True))[:,0];tokens.append(previous.clone())
  return {'hidden':hidden,'tokens':torch.stack(tokens,1),'context_length':start,'query_KV_committed':False}
 def crop_context(self,acceptedprefix):
  self.check();assert self.positions is not None and 0<=acceptedprefix<=self.positions.shape[1]
  self.positions=self.positions[:,:acceptedprefix].clone();self.kv=[(k[:,:,:acceptedprefix].clone(),v[:,:,:acceptedprefix].clone())for k,v in self.kv];self.last_query_tails=[];return acceptedprefix
 def close(self):
  self.check();self.kv=[];self.positions=None;self.last_query_tails=[];self.model=None;self.head=None;self.proj=None;self.ctxnorm=None;self.norm=None;self.closed=True
