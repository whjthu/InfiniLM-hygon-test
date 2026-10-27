"""CPU byte/layout fixtures plus source binding/invariants; NOT C++ execution."""
from pathlib import Path
import hashlib,json,copy
ROOT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bindings=json.loads((ROOT/'source_bindings_r124.json').read_text())
for b in bindings:assert sha(ROOT/'base'/b['file'])==b['private_base_SHA256']
actual=json.loads((ROOT/'actual_binding_readonly_r124.json').read_text())
for u in actual['units']:
 assert actual['current_link']['objects'][u['object']]==u['object_SHA256']
 assert actual['files'][u['source']]['SHA256']==u['source_SHA256']
checks=[]
for name,stage in [('deepseek_v4_swa_attention.cpp','stage_swa'),('deepseek_v4_hca_attention_batch_r784.cpp','stage_hca'),('deepseek_v4_csa_attention_batch_r784.cpp','stage_csa')]:
 s=(ROOT/'src'/name).read_text();assert s.count('if(prefix_r124::active)')==1;assert stage in s;assert '}else{cache_->store' in s;checks.append(name+': opt-in stage + original OFF commit')
index=(ROOT/'src/deep_c4_smallt_r9252.cpp').read_text();assert 'return q.tokens==1' in index
decoder=(ROOT/'src/deepseek_v4_native_routed_decoder_r20.cpp').read_text();assert 'decoder_done(prefix_layer_)' in decoder
model=(ROOT/'src/deepseek_v4_routed_text_model_r20.cpp').read_text();assert 'begin_prefix' in model and 'forward(r,ids,pos,q,true)' in model and 'prefix pending: ordinary forward forbidden' in model
adapter=(ROOT/'deep_prefix_native_r124.cpp').read_text();assert 'batch.score->narrow' in adapter and 'self.state_->shape()' in adapter;assert 'PerChannelQuant' not in adapter and 'I8Gemm' not in adapter and 'DeepseekV4Compress>' not in adapter
checks+=['43 decoder identity staging','all-position head selection','ordinary forward blocked while pending','projected score slice reuse without new math op']
# Fixed-size compressor persistence: compare original kernel owner selection
# against adapter sequential score-row scatter for real D512/D128 byte widths.
fixtures=0
for ratio,dim in [(4,512),(4,128),(128,512)]:
 width=8 if ratio==4 else 128;channels=4 if ratio==4 else 2;record_bytes=channels*dim*2
 for past in [2,3,4,5,7,126,127,128,129,2047]:
  initial=[bytes([slot%251])*record_bytes for slot in range(width)]
  scores=[bytes([201+t])*record_bytes for t in range(2)]
  for count in [1,2]:
   expected=initial[:];owners={}
   for t in range(count):p=past+t;slot=((p//4)%2)*4+p%4 if ratio==4 else p%128;owners[slot]=t
   for slot,t in owners.items():expected[slot]=scores[t]
   got=initial[:]
   for t in range(count):p=past+t;slot=((p//4)%2)*4+p%4 if ratio==4 else p%128;got[slot]=scores[t]
   assert got==expected;fixtures+=1
# Whole physical page COW: untouched packed data/scales/padding+validity exact.
stride=((256*584+575)//576)*576
for past in [3,127,128,255,256]:
 for count in [1,2]:
  base=bytearray((i*13+7)%256 for i in range(stride));valid=[False]*256
  valid[:min(past,128)]=[True]*min(past,128)
  old=bytes(base);shadow=base[:];newvalid=valid[:]
  touched=set()
  for t in range(count):slot=(past+t)%128;data=slot*576;scales=256*576+slot*8;touched.update(range(data,data+576));touched.update(range(scales,scales+8));shadow[data:data+576]=bytes([81+t])*576;shadow[scales:scales+8]=bytes([91+t])*8;newvalid[slot]=True
  assert bytes(base)==old and valid!=newvalid if past<128 else bytes(base)==old
  assert all(shadow[i]==base[i] for i in range(stride) if i not in touched)
  # Cancellation drops shadow; no base valid flag is changed.
  assert bytes(base)==old;fixtures+=1
result=dict(status='PASS_SOURCE_AND_CPU_BYTE_FIXTURES_ONLY',source_checks=checks,byte_fixtures=fixtures,CXX_compiled=False,native_executed=False,
            note='These fixtures validate byte-write-set arithmetic and source anchors. They do not execute C++ adapters, Tensor API, cache kernels or eight-owner atomic publication.')
(ROOT/'CPU_source_checks_r124.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
