import pathlib,json,re,struct,array,hashlib,time
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');raw=r/'results/Kimi-layer0-raw-r9042c-910375.697';rings=[];tree={};logpaths=list((r/'logs').glob('Kimi-layer0-real-r9042c-910375.697-*.log'))
for p in logpaths:
 for line in p.read_text().splitlines():
  m=re.search(r'Channel \d+/32 :\s+((?:\d+\s+){15}\d+)\s*$',line)
  if m:
   v=list(map(int,m.group(1).split()))
   if v not in rings:rings.append(v)
  m=re.search(r'Trees \[0\] (-?\d+)/(-?\d+)/(-?\d+)->(\d+)->(-?\d+)',line)
  if m:tree[int(m[4])]=[int(m[k])for k in [1,2,3]if int(m[k])>=0]
assert len(rings)>0 and all(sorted(x)==list(range(16))for x in rings)
dec=lambda x:struct.unpack('<f',struct.pack('<I',int(x)<<16))[0]
def enc(x):
 u=struct.unpack('<I',struct.pack('<f',x))[0];return((u+0x7fff+((u>>16)&1))>>16)&65535
def add(a,b):return[enc(dec(x)+dec(y))for x,y in zip(a,b)]
def row(arm,rank,point):
 p=raw/arm/f'stage{rank//8}-rank{rank%8}-layer0-kda-call25.jsonl';z=next(z for z in[json.loads(x)for x in p.read_text().splitlines()]if z['point']==point);return list(array.array('H',(raw/arm/z['file']).read_bytes()))[:7168]
q={'GPU':False,'new_capture':False,'raw_remote_only':True,'handle':'910375.697','known_init_rings':rings,'known_init_tree_channel0':tree,'cases':[],'interpretation':'BF16 round-after-each-add candidates only. Actualalgo/proto/perchunkroot/channelmapping UNKNOWN; init topology doesnotprove selectedalgorithm.'};t=time.monotonic()
for arm in ['A','B']:
 local=[row(arm,k,'r9042_o_proj_local')for k in range(16)];actual=row(arm,0,'r9042_o_proj_reduced');models=[]
 for rid,ring in enumerate(rings):
  for offset in range(16):
   order=ring[offset:]+ring[:offset];value=local[order[0]]
   for rank in order[1:]:value=add(value,local[rank])
   models.append({'kind':'ring_sequential_BF16_RNE','ring':rid,'rotation':offset,'different_elements':sum(x!=y for x,y in zip(value,actual)),'max_abs':max(abs(dec(x)-dec(y))for x,y in zip(value,actual)),'SHA256':hashlib.sha256(array.array('H',value).tobytes()).hexdigest()})
 if len(tree)==16:
  def visit(rank,ancestors):
   assert rank not in ancestors;v=local[rank]
   for child in tree[rank]:v=add(v,visit(child,ancestors+[rank]))
   return v
  value=visit(0,[]);models.append({'kind':'init_channel0_tree_selfthenchildren_BF16_RNE','different_elements':sum(x!=y for x,y in zip(value,actual)),'max_abs':max(abs(dec(x)-dec(y))for x,y in zip(value,actual)),'SHA256':hashlib.sha256(array.array('H',value).tobytes()).hexdigest()})
 q['cases'].append({'arm':arm,'position':536,'model_count':len(models),'min_different':min(z['different_elements']for z in models),'max_different':max(z['different_elements']for z in models),'top8':sorted(models,key=lambda z:z['different_elements'])[:8],'exact_matches':sum(z['different_elements']==0 for z in models)})
q['CPU_wall_s']=time.monotonic()-t;print(json.dumps(q))
