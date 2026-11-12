"""CPU structured state comparison,including native NumPy RNG arrays."""
from . import hy_resume_checkpoint as common


def compare_trees(a,b):
    import numpy as np
    import torch
    count=0;arrays=0;bad=0;maximum=0.
    def visit(x,y):
        nonlocal count,arrays,bad,maximum
        if isinstance(x,torch.Tensor):
            count+=1
            if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:
                bad+=1;return
            if x.device.type!='cpu' or y.device.type!='cpu':raise ValueError('comparison requires CPU tensors')
            bad+=int(not torch.equal(x,y))
            if x.numel():maximum=max(maximum,float((x.float()-y.float()).abs().max()))
        elif isinstance(x,np.ndarray):
            arrays+=1
            if not isinstance(y,np.ndarray) or x.shape!=y.shape or x.dtype!=y.dtype:
                bad+=1;return
            bad+=int(not np.array_equal(x,y))
            if x.size and np.issubdtype(x.dtype,np.number):maximum=max(maximum,float(np.abs(x.astype(np.float64)-y.astype(np.float64)).max()))
        elif isinstance(x,dict):
            if not isinstance(y,dict) or set(x)!=set(y):bad+=1;return
            for k in x:visit(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            if type(x) is not type(y) or len(x)!=len(y):bad+=1;return
            for l,r in zip(x,y):visit(l,r)
        elif type(x) is not type(y) or x!=y:bad+=1
    visit(a,b)
    return dict(tensor_leaves=count,numpy_array_leaves=arrays,mismatched_leaves=bad,max_abs=maximum,tree_equal=common.tree_sha256(a)==common.tree_sha256(b))
