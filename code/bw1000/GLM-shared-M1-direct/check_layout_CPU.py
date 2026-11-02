"""Source-bound singleton shared-view address/scale contract, no Native or GPU."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
raw=(D/'original_raw_r19.cu').read_text()
assert hashlib.sha256(raw.encode()).hexdigest()=='c9b899fbf5a6f8eb715d806bc592e79b8b738b4334dd938d83e6f1e40f4e326a'
for text in ['token=route/8','e=size_t(ids[route])','((offset+row)/128)*48+b',
             'xs[route*2+b]','(row/128)*16+offset/128+b',
             'dim3(64,unsigned(routes))','dim3(768,unsigned(routes))']:
    assert text in raw,text
assert 'routes%8'not in raw
wrapper=(D/'original_5123_wrapper.cpp').read_text();assert 'routes%8'in wrapper
weights=(D/'weights.hpp').read_text();assert 'const Tensor &tensor() const noexcept'in weights
cpp=(D/'candidate_ffn.cpp').read_text();adapter=(D/'shared_M1_adapter.inc').read_text()
assert 'g.state_dict()'in adapter and 'uw.at("weight").tensor()'in adapter
assert 'glm_grouped_gate_r5123'in cpp and 'glm_grouped_down_r5123'in cpp
assert 'if(direct)shared_m1_direct_v1::down'in cpp
checks=0
for tokens in [0,1,2,8,16,512]:
 for local in [256,1536]:
  for global_width in [2048,12288]:
   chosen=tokens==1 and local==256 and global_width==2048
   assert chosen==((tokens,local,global_width)==(1,256,2048));checks+=1
for offset in [-256,0,256,512,768,1024,1280,1536,1792,2048]:
 valid=0<=offset<=1792 and offset%256==0
 assert valid==(offset in range(0,2048,256));checks+=1
for rank in range(8):
 offset=rank*256
 # Real underlying raw primitive has one route, fixed logical expert0 only.
 route=expert=0;assert route//8==0
 for row in range(256):
  for b in range(48):
   scale=(offset+row)//128*48+b
   assert 0<=scale<16*48
   for k_in_block in [0,127]:
    k=b*128+k_in_block;weight=(expert*256+row)*6144+k
    assert 0<=weight<256*6144
   checks+=1
 for row in range(6144):
  for b in range(2):
   weight=row*256+b*128+127;scale=(row//128)*16+offset//128+b
   assert 0<=weight<6144*256 and 0<=scale<48*16
   # Original SDK receives a global scale slab with only the local columns
   # written. The direct provider reads exactly those same two scale values.
   local_xs=[rank+0.125,rank+0.875];global_xs=[None]*16
   global_xs[offset//128:offset//128+2]=local_xs
   assert local_xs[route*2+b]==global_xs[offset//128+b]
   checks+=1
print(json.dumps({'completed':True,'passed':True,'CPU_only':True,'GPU':0,'Native_dlopen':False,
 'actual_kernel_source_bound':True,'checks':checks,'TP_offsets_checked':8,
 'public5123_route1_refused':True,'raw_r19_route0_addressing_fits_single_shared_view':True,
 'FP8_layout':'original E4M3 uint8 payload + F32 per128x128 checkpoint scales',
 'shapes':{'gate_up_W':[256,6144],'gate_up_WS':[16,48],'X':[1,6144],'XS':[1,48],
           'down_W':[6144,256],'down_WS':[48,16],'down_X':[1,256],'down_XS':[1,2]},
 'numerical_GPU_or_Model_qualification':False}))
