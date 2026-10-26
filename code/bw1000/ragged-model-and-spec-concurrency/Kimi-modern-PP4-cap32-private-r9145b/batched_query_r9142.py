"""DefaultOFF ragged DFlash2 batch. Original worker caches features, not draft KV."""
def query_batched(model,head,contexts,anchors,Ks,enabled=False,return_debug=False):
 if not enabled:raise RuntimeError('r9142 defaultOFF')
 import torch
 B=len(contexts);assert 1<=B<=32 and B==len(anchors)==len(Ks) and all(1<=K<=7 for K in Ks)
 lengths=[int(x.shape[1])for x in contexts];assert all(x.shape[0]==1 and x.shape[2]==35840 and 1<=n<=1024 for x,n in zip(contexts,lengths));L=max(lengths);dev=contexts[0].device
 # Padding is masked in EVERY layer; layer4 global attention also must exclude it.
 features=contexts[0].new_zeros((B,L,35840))
 for i,x in enumerate(contexts):features[i,:lengths[i]]=x[0]
 ctx=model.context_norm(model.context_proj(features));ids=torch.full((B,8),163592,dtype=torch.long,device=dev);ids[:,0]=torch.tensor(anchors,device=dev);x=model.embed_tokens(ids)
 cp=torch.arange(L,device=dev).view(1,-1).expand(B,-1);lens=torch.tensor(lengths,device=dev);dp=lens[:,None]+torch.arange(8,device=dev)[None,:];keypos=torch.cat([cp,dp],1);valid=torch.cat([cp<lens[:,None],torch.ones((B,8),device=dev,dtype=torch.bool)],1);globalmask=valid[:,None,None,:].expand(-1,1,8,-1);localmask=globalmask & ((dp[:,:,None]-keypos[:,None,:]).abs()<4096)[:,None,:,:]
 for i,layer in enumerate(model.layers):x=layer(x,ctx,dp,cp,localmask if i<4 else globalmask)
 hidden=model.final_norm(x);logits=head(hidden[:,1:]);previous=torch.tensor(anchors,device=dev);props=[]
 for j in range(7):
  scores,candidates=model.candidate_selector.score_candidates(hidden[:,j+1],logits[:,j],previous);previous=candidates.gather(-1,scores.argmax(-1,keepdim=True))[:,0];props.append(previous.clone())
 values=torch.stack(props,1);assert values.shape==(B,7)
 return (values,{"hidden":hidden,"model_batch_C":B,"padded_context_L":L})if return_debug else values
