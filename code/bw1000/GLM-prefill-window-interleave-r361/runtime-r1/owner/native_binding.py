"""Exact current038d forward ABI consumer; tensors are supplied by existing owner.

No library load, context create, device initialization or weight access here.
The eventual owner supplies its existing torch/ptr/stream/check boundaries.
"""
import ctypes
import time


def create_args(rank, stage, cfg_bytes, world, tp, control):
    """Arguments for the existing factory; actual construction is NOT performed."""
    import json
    cfg=json.loads(cfg_bytes)
    if (rank not in range(8) or stage not in range(4)
            or cfg.get('num_hidden_layers')!=78 or cfg.get('hidden_size')!=6144
            or cfg.get('vocab_size')!=154880 or cfg.get('max_position_embeddings',0)<32768):
        raise ValueError('Current GLM geometry required')
    return rank,stage,2,32768,cfg_bytes,world,tp,control


class Forward:
    def __init__(self, lib, ctx, stage, rank, torch, ptr, stream, check, vocab=154880):
        if stage not in range(4) or rank not in range(8) or vocab != 154880:
            raise ValueError('Current TP8/PP4 vocabulary identity required')
        self.lib,self.ctx,self.stage,self.rank=lib,ctx,stage,rank
        self.torch,self.ptr,self.stream,self.check,self.vocab=torch,ptr,stream,check,vocab

    def __call__(self, *, tokens, count, descriptor, input_ids, positions):
        if (tokens,count,len(descriptor),len(input_ids),len(positions)) != (1,1,3,1,1):
            raise ValueError('Only independent M1 decode supported')
        row,past,n=descriptor
        if row not in (0,1) or type(past) is not int or not 0<=past<32768 or n!=1 or positions!=(past,):
            raise ValueError('Native row/past mapping mismatch')
        t=self.torch
        desc=(ctypes.c_size_t*3)(*descriptor)
        ids=t.tensor(input_ids if self.stage==0 else (0,),device='cuda',dtype=t.int64)
        pos=t.tensor(positions,device='cuda',dtype=t.int64)
        logits=t.full((1,self.vocab),float('nan'),device='cuda',dtype=t.bfloat16)
        t.cuda.synchronize(self.rank)
        tick=time.perf_counter()
        self.check(self.lib.modern_glm_model_forward(self.ctx,1,1,desc,
                    self.ptr(ids),self.ptr(pos),self.ptr(logits),0,self.stream()))
        ended=time.perf_counter()
        if self.stage==3 and not bool(t.isfinite(logits).all()):
            raise ValueError('Nonfinite actual Native logits')
        return logits,ended-tick,tick,ended
