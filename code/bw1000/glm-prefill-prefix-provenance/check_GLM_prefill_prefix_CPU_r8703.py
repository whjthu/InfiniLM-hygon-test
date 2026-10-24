import pathlib,json,hashlib,random,ast
r=pathlib.Path('.');rng=random.Random(8703)
# CPU equivalents of the actual r302 producer and existing r48 set/causality gate.
def remap(host,count,width,past):
 prefix=past+count;used=set()
 for t in range(count):
  for value in host[t*width:(t+1)*width]:
   if value<0 or value>=prefix:raise ValueError('range')
   if value<=past+t:used.add(value)
 logical=sorted(used);mapping={v:i for i,v in enumerate(logical)};mapped=[]
 for t in range(count):
  row=host[t*width:(t+1)*width];causal=[v for v in row if v<=past+t]
  if not causal:raise ValueError('no causal')
  mapped.extend(mapping[v] if v<=past+t else mapping[causal[0]] for v in row)
 return logical,mapped
def old_dense(logical,mapped,count,width,past):
 selected=[sorted({v for v in mapped[t*width:(t+1)*width] if logical[v]<=past+t}) for t in range(count)]
 return count>1 and len(logical)==past+count and logical==list(range(len(logical))) and all(row==list(range(past+t+1)) for t,row in enumerate(selected))
def proven(logical,mapped,count,width,past):
 prefix=past+count
 if count<2 or logical!=list(range(prefix)):return False
 seen=[0]*prefix
 for t in range(count):
  unique=0
  for value in mapped[t*width:(t+1)*width]:
   if value<0 or value>=prefix:raise ValueError('range')
   if value<=past+t and seen[value]!=t+1:seen[value]=t+1;unique+=1
  if unique!=past+t+1:return False
 return True
cases=0;eligible=0
for _ in range(4000):
 count=rng.randint(1,24);past=rng.randint(0,12);prefix=past+count;width=rng.randint(1,prefix+8)
 rows=[]
 for t in range(count):
  row=[rng.randrange(prefix) for _ in range(width)];row[0]=rng.randrange(past+t+1);rows.extend(row)
 logical,mapped=remap(rows,count,width,past);assert proven(logical,mapped,count,width,past)==old_dense(logical,mapped,count,width,past);cases+=1;eligible+=proven(logical,mapped,count,width,past)
for count,past in [(2,0),(8,3),(512,0),(128,512),(2,2046)]:
 prefix=past+count;host=[]
 for t in range(count):
  row=list(range(prefix));rng.shuffle(row);host.extend(row)
 logical,mapped=remap(host,count,prefix,past);assert proven(logical,mapped,count,prefix,past) and old_dense(logical,mapped,count,prefix,past);cases+=1;eligible+=1
# Same union of keys is insufficient: row0 misses a required key.
logical,mapped=remap([0,0,0,0,1,2],2,3,1);assert logical==[0,1,2] and not proven(logical,mapped,2,3,1);cases+=1
s=(r/'glm_prefill_prefix_dense_r8703.cpp').read_text();old=(r/'glm_moe_dsa_dense_modes_r48.cpp').read_text();a=s.index(' std::vector<rt::Tensor> held;');b=s.index('\nrt::Tensor sparse_attention_mapped_bounded_proven_r8703');a0=old.index(' std::vector<rt::Tensor> held;');b0=old.index('\n}\n\nnamespace infinilm::models::glm_moe_dsa {\nextern "C" int modern_glm_dense_mode_r48_set')
body=s[a:b].replace('bool dense=trusted8703||(tokens>1 && k->size(0)==past+tokens && k->size(0)<=2048);','bool dense=tokens>1 && k->size(0)==past+tokens && k->size(0)<=2048;').replace('for(size_t t=0;dense&&!trusted8703&&t<tokens;++t)','for(size_t t=0;dense&&t<tokens;++t)')
assert body==old[a0:b0], 'numeric block changed beyond proven dense predicate'
assert s.count('r.syncStream()')==old.count('r.syncStream()')
ast.parse((r/'prepare_GLM_prefill_prefix_r8703.py').read_text(encoding='utf-8-sig'))
m=json.loads((r/'GLM-prefill-prefix-CPU-source-r8703.json').read_text());m['CPU_checks']={'property_cases':cases,'eligible_cases':eligible,'proof_equivalent_to_old_dense_gate':True,'covers_actual_512_and_extreme_2048_prefix':True,'union_full_but_one_query_missing_rejected':True,'arithmetic_block_identical_except_proven_dense_predicate':True,'syncStream_count_unchanged':True,'native_compile':False,'actual_GPU_scope_hit':'NOT_RUN'}
(r/'GLM-prefill-prefix-CPU-source-r8703.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps(m['CPU_checks']))
