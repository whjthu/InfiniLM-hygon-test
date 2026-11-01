"""Independent CPU float32 epilogue; actual baseline must pass this first."""
import numpy as np

def bf16_rn(x):
 x=np.asarray(x,dtype=np.float32);u=x.view(np.uint32);rounded=u+np.uint32(0x7fff)+((u>>16)&np.uint32(1));return (rounded>>16).astype(np.uint16)

def epilogue(xf,mixes,base,scale,constants):
 xf=np.asarray(xf,dtype=np.float32).reshape(4,4096);m=np.asarray(mixes,dtype=np.float32).reshape(24);b=np.asarray(base,dtype=np.float32).reshape(24);s=np.asarray(scale,dtype=np.float32).reshape(3);c=np.asarray(constants,dtype=np.float32).reshape(3)
 assert all(np.isfinite(v).all()for v in [xf,m,b,s,c])and c[1]>0
 affine=lambda x,k,offset:np.add(np.multiply(x,s[k],dtype=np.float32),b[offset:offset+len(x)],dtype=np.float32)
 sigmoid=lambda x:np.divide(np.float32(1),np.add(np.float32(1),np.exp(-x,dtype=np.float32),dtype=np.float32),dtype=np.float32)
 pre=np.add(sigmoid(affine(m[:4],0,0)),c[1],dtype=np.float32);post=np.multiply(sigmoid(affine(m[4:8],1,4)),c[2],dtype=np.float32)
 logits=affine(m[8:],2,8).reshape(4,4);ex=np.exp(logits-logits.max(axis=1,keepdims=True),dtype=np.float32);comb=np.add(ex/ex.sum(axis=1,keepdims=True,dtype=np.float32),c[1],dtype=np.float32)
 weighted=xf*pre[:,None];collapsed=(weighted[0]+weighted[1])+(weighted[2]+weighted[3])
 return dict(collapsed_BF16=bf16_rn(collapsed),post=post,comb_before_original_Sinkhorn=comb,pre=pre)
