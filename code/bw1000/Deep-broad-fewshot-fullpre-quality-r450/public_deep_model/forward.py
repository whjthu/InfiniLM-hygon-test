"""Exact qualified d206 CAPI forward body, statically bound to HC0 defaults."""
import ctypes,os
import numpy as np

def make_forward(lib,ctx,cfg,report,check):
 HC_mode=projection_tile_mode=c4_tile_mode=layout_mode=tail_SUM_mode=0
 precision_mode=5
 csa_layers=sum(int(v)==4 for v in cfg['compress_ratios'][:cfg['num_hidden_layers']])
 def forward(ids,positions,rows,past,total,offsets):
   assert projection_tile_mode in [0,16] and 0<len(rows)<=128
   assert len(offsets)==len(rows)+1 and offsets[0]==0 and offsets[-1]==len(ids)
   widths=[b-a for a,b in zip(offsets,offsets[1:])]
   decode=all(w==1 for w in widths) and all(p>0 for p in past)
   assert all(t==p+w for t,p,w in zip(total,past,widths))
   tile_enabled=projection_tile_mode==16 and decode and 16<len(rows)<=64
   os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16'if tile_enabled else'0'
   tile_before=[int(lib.deep_output_tile_calls_r4808(k))for k in range(8)]
   assert c4_tile_mode in[0,16]and precision_mode==5
   check(lib.modern_deep_c4_tile_policy_r5920(c4_tile_mode));assert lib.modern_deep_c4_tile_get_r5920()==c4_tile_mode
   c4_enabled=c4_tile_mode==16 and precision_mode==5 and decode and 16<len(rows)<=64
   c4_before=[[int(lib.modern_deep_c4_tile_calls_r5920(d,k))for k in range(2)]for d in range(8)]
   FP32_before=[int(lib.modern_deep_FP32_calls_r3597(d))for d in range(8)]
   hc_policy=lib.deep_hc_canonical_get_r7130();assert hc_policy==HC_mode
   hc_before=[[int(lib.deep_hc_canonical_calls_r7130(d,k)) for k in range(3)] for d in range(8)]
   assert lib.modern_deep_layout_get_r7052()==layout_mode
   assert lib.deep_moe_tail_get_r7211()==tail_SUM_mode
   tail_before=[[int(lib.deep_moe_tail_calls_r7211(d,k))for k in range(5)]for d in range(8)]
   layout_before=[[int(lib.modern_deep_layout_calls_r7052(d,k))for k in range(4)]for d in range(8)]
   values=[np.ascontiguousarray(x,dtype=np.int64 if i>=4 else np.int32)for i,x in enumerate([rows,past,total,offsets,ids,positions])]
   logits=np.full((len(rows),cfg['vocab_size']),0x7fc0,dtype=np.uint16);emitted=np.full(len(rows),-999,dtype=np.int64)
   code=lib.modern_native_engine_forward(ctx,len(ids),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in values],ctypes.c_void_p(logits.ctypes.data),ctypes.c_void_p(emitted.ctypes.data))
   if code==0:
    delta=[int(lib.deep_output_tile_calls_r4808(k))-v for k,v in enumerate(tile_before)]
    assert delta==[43 if tile_enabled else 0]*8,('All43projectiondispatchgate',tile_enabled,delta)
    c4_delta=[[int(lib.modern_deep_c4_tile_calls_r5920(d,k))-c4_before[d][k]for k in range(2)]for d in range(8)]
    assert c4_delta==[[2*csa_layers if c4_enabled else 0]*2 for d in range(8)],('C4_actual_same_image_path',c4_enabled,c4_delta)
    FP32_delta=[int(lib.modern_deep_FP32_calls_r3597(d))-FP32_before[d]for d in range(8)]
    assert FP32_delta==[43 if precision_mode==5 and len(rows)>1 else 0]*8,('FP32scope_actual',precision_mode,len(rows),FP32_delta)
    hc_after=[[int(lib.deep_hc_canonical_calls_r7130(d,k)) for k in range(3)] for d in range(8)];hc_delta=[[hc_after[d][k]-hc_before[d][k] for k in range(3)] for d in range(8)];hc_kind=(1 if HC_mode else 0) if len(ids)==512 else 2;assert hc_delta==[[86 if k==hc_kind else 0 for k in range(3)] for d in range(8)],('actualall86HCdispatch',HC_mode,len(ids),hc_delta)
    report.setdefault('HC_dispatch_checked_forwards',0);report['HC_dispatch_checked_forwards']+=1
    layout_after=[[int(lib.modern_deep_layout_calls_r7052(d,k))for k in range(4)]for d in range(8)];layout_delta=[[layout_after[d][k]-layout_before[d][k]for k in range(4)]for d in range(8)];eligible=all(p==0 for p in past)and len(set(widths))>1 and len(rows)*max(widths)<=4096;target=len(rows)==1 and len(ids)==512 and all(p==0 for p in past);tail_selected=bool(tail_SUM_mode and target);moe=int(bool(layout_mode&1)and eligible)or int(tail_selected);attn=int(bool(layout_mode&2)and len(rows)==1);expected_layout=[43*moe,43*(33554432 if tail_selected else len(rows)*max(widths)*4096*4 if moe else 0),43*attn,43*(1-moe)];assert layout_delta==[expected_layout]*8,('all43layoutdispatch',layout_mode,widths,past,layout_delta,expected_layout)
    report.setdefault('layout_dispatch_checked_forwards',0);report['layout_dispatch_checked_forwards']+=1
    tail_after=[[int(lib.deep_moe_tail_calls_r7211(d,k))for k in range(5)]for d in range(8)];tail_delta=[[tail_after[d][k]-tail_before[d][k]for k in range(5)]for d in range(8)];expected_tail=[43*int(tail_selected),43*33554432*int(tail_selected),43*8388608*int(tail_selected),43*int(target and not tail_selected),43*int(not target)];assert tail_delta==[expected_tail]*8,('all43strict_tailSUM_dispatch',tail_SUM_mode,rows,widths,past,tail_delta,expected_tail)
    key_tail=f'flag{tail_SUM_mode}-target{int(target)}-selected{int(tail_selected)}-rows{len(rows)}-T{len(ids)}';report.setdefault('tail_SUM_dispatch_checks',{});report['tail_SUM_dispatch_checks'][key_tail]=report['tail_SUM_dispatch_checks'].get(key_tail,0)+1
 
    key=f"{'decode' if decode else 'prefill'}-rows{len(rows)}-tile{16 if tile_enabled else 0}-c4{c4_tile_mode}-active{int(c4_enabled)}"
    stats=report.setdefault('dispatch_checks_r4842',{})
    stats[key]=stats.get(key,0)+1
   return code,logits,emitted
 return forward
