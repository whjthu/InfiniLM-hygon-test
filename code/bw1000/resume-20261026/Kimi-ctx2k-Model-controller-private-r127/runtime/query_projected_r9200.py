"""Private paired control/candidate. Fixed8 transformer positions; only Kmax head/selector crop."""
def query_projected(model,head,contexts,anchors,Ks,enabled=False,return_debug=False,profile=False):
 if not enabled:raise RuntimeError('r9182 defaultOFF')
 import torch
 B=len(contexts);assert 1<=B<=32 and B==len(anchors)==len(Ks) and all(type(K)is int and 1<=K<=7 for K in Ks)
 lengths=[int(x.shape[1])for x in contexts];assert all(x.shape[0]==1 and x.shape[2]==int(model.context_proj.weight.shape[0]) and 1<=n<=2048 for x,n in zip(contexts,lengths));L=max(lengths);dev=contexts[0].device
 Kmax=max(Ks);events=[]
 def point(label):
  if profile:
   event=torch.cuda.Event(enable_timing=True);event.record();events.append((label,event))
 point('start')
 features=contexts[0].new_zeros((B,L,int(model.context_proj.weight.shape[0])))
 for i,x in enumerate(contexts):features[i,:lengths[i]]=x[0]
 point('packed_context')
 ctx=features;point('context_proj_norm')
 ids=torch.full((B,8),163592,dtype=torch.long,device=dev);ids[:,0]=torch.tensor(anchors,device=dev);x=model.embed_tokens(ids)
 cp=torch.arange(L,device=dev).view(1,-1).expand(B,-1);lens=torch.tensor(lengths,device=dev);dp=lens[:,None]+torch.arange(8,device=dev)[None,:];keypos=torch.cat([cp,dp],1);valid=torch.cat([cp<lens[:,None],torch.ones((B,8),device=dev,dtype=torch.bool)],1);globalmask=valid[:,None,None,:].expand(-1,1,8,-1);localmask=globalmask & ((dp[:,:,None]-keypos[:,None,:]).abs()<4096)[:,None,:,:]
 point('embedding_masks')
 for i,layer in enumerate(model.layers):
  x=layer(x,ctx,dp,cp,localmask if i<4 else globalmask);point('layer'+str(i))
 hidden=model.final_norm(x);point('final_norm')
 prefix=hidden[:,1:1+Kmax]
 head_input=prefix.contiguous().view(B*Kmax,prefix.shape[-1])
 logits=head(head_input)
 logits=logits.view(B,Kmax,-1)
 point('head')
 previous=torch.tensor(anchors,device=dev);props=[]
 for j in range(Kmax):
  scores,candidates=model.candidate_selector.score_candidates(hidden[:,j+1],logits[:,j],previous);previous=candidates.gather(-1,scores.argmax(-1,keepdim=True))[:,0];props.append(previous.clone())
 values=torch.stack(props,1);point('selector');assert values.shape==(B,Kmax)
 debug={'hidden':hidden,'model_batch_C':B,'padded_context_L':L,'fixed_transformer_positions':8,'head_positions':Kmax,'selector_steps':Kmax,'event_points':events,'profile_extra_drains':0,'head_input_shape':list(head_input.shape),'head_input_stride':list(head_input.stride()),'head_input_contiguous':head_input.is_contiguous(),'prefix_stride':list(prefix.stride()),'flatten_head':True,'cached_projected_context':True}
 return (values,debug)if return_debug else values
