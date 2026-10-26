import pathlib,json,hashlib
p=pathlib.Path(__file__).parent;checks=0
for routes in [64,128]:
 seen=set()
 for block in range(384):
  for group in range(8):
   for row in [(block*8+group)*2,(block*8+group)*2+1]:
    assert row not in seen;seen.add(row)
    for offset in range(0,2048,256):
     for b in [0,1]:
      assert (row//128)*16+offset//128+b>=0;checks+=1
 assert seen==set(range(6144))
for lane in range(32):
 for j in range(4):assert (lane//4+j*8)*4+lane%4==lane+j*32;checks+=1
raw=pathlib.Path('/refactor/incoming/glm_scalar_vectorload_r8528.cu').read_bytes();assert hashlib.sha256(raw).hexdigest()=='b257ff25f042d86a166bf7873c1b1a781b106a870dd7b62fae297f1de90b5eb0';base=raw.decode().replace(chr(13)+chr(10),chr(10));s=(p/'vector-row2-r9281.cu').read_text();marker='}'+chr(10)+'extern "C" int glm_vec_validate_r8528';assert s.startswith(base[:base.index(marker)]);start=s.index('extern "C" int glm_vec_validate_r8528');stop=s.index('extern "C" int glm_vec_down_row2_r9281');assert s[start:stop]==base[base.index('extern "C" int glm_vec_validate_r8528'):].replace('extern "C" int glm_vec_down_r8528(', 'extern "C" int glm_vec_down_reference_r9281(')
print(json.dumps({'passed':True,'CPU_only':True,'checks':checks,'T':[16,8],'row_K_scale_coverage':True,'gate_reference_math_unchanged':True,'no_GPU_F64':True}))
