"""CPU coordinate contract only. No checkpoint/activation read or GPU work."""
import pathlib,json,hashlib,datetime
r=pathlib.Path(__file__).resolve().parent
E,N,K=896,384,3584
qorder=[0,2,4,6,1,3,5,7]
decoded=[0,4,1,5,2,6,3,7]
assert [qorder[j]for j in decoded]==list(range(8))
# Full local W13 matrix: every destination code maps to exactly one original
# coordinate; unlike a single-qword example, include N64 blocks and K64 loops.
seen=bytearray(N*K);count=0;scale_count=0
for nblock in range(N//64):
 for n in range(64):
  row=nblock*64+n
  for kblock in range(K//64):
   for col in range(2):
    scale_idx=row*(K//32)+2*kblock+col
    assert scale_idx==row*(K//32)+(kblock*64+col*32)//32
    scale_count+=1
    for word in range(4):
     kbase=kblock*64+col*32+word*8
     # dequant writes these positions as consecutive bf16 entries into LDS_B.
     result=[kbase+qorder[j]for j in decoded]
     assert result==list(range(kbase,kbase+8))
     for logicalk in result:
      p=row*K+logicalk;assert not seen[p];seen[p]=1;count+=1
assert count==N*K and all(seen)
# Expert base offsets disjoint and exhaust exactly E whole local matrices.
assert all(e*N*K+(N*K-1)<(e+1)*N*K for e in range(E-1))
out={"scope":"CPU full local-matrix coordinate/scale-index contract; no actual weights/repack GPU validation","utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"passed":True,"shape":[E,N,K],"nibble_destination_order":qorder,"official_decode_order":decoded,"all_local_matrix_codes_checked":count,"all_local_matrix_scales_checked":scale_count,"expert_axis_disjoint_intervals":E,"global_codes_covered_by_product_proof":E*N*K,"global_weight_bytes":E*N*K//2,"global_scale_bytes":E*N*K//32,"limits":["No original checkpoint bytes read; actual repack roundtrip still required at component qualification","No proof of official MMA lane/egress beyond source-derived qword and global coordinate mapping","Original kernel accumulates FP32 scalar products; MMA reduction association changes and final model IDs remain future acceptance"]}
p=r/'Kimi-official-WFP4-layout-CPU-r8669zr.json';assert not p.exists();p.write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
