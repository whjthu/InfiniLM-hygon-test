import pathlib,json,hashlib,struct,subprocess,collections,array
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');raw=r/'results/Kimi-layer0-raw-r9039-910375.685';q={'handle':'910375.685','GPU':False,'new_capture':False,'raw_remote_only':True,'inventory_rank0':[],'matrix_points':[],'final_state':[],'all_journals':{'count':0,'snapshot_records':0,'nonfinite':0},'reference_cache_binding':[]}
def rows(arm,node,device,call):return [json.loads(x)for x in(raw/arm/f'stage{node}-rank{device}-layer0-kda-call{call}.jsonl').read_text().splitlines()]
def data(arm,rec):return(raw/arm/rec['file']).read_bytes()
def diff(a,b,dtype):
 assert len(a)==len(b)
 if dtype=='bf16':aa=array.array('H',a);bb=array.array('H',b);decode=lambda z:struct.unpack('<f',struct.pack('<I',z<<16))[0]
 elif dtype=='f32':aa=array.array('f',a);bb=array.array('f',b);decode=float
 else:aa=array.array('i'if dtype=='i32'else'q',a);bb=array.array('i'if dtype=='i32'else'q',b);decode=float
 changed=sum(x!=y for x,y in zip(aa,bb));ma=max((abs(decode(x)-decode(y))for x,y in zip(aa,bb)),default=0);return{'different_elements':changed,'max_abs':ma,'A_SHA256':hashlib.sha256(a).hexdigest(),'C_SHA256':hashlib.sha256(b).hexdigest()}
for node in range(2):
 for device in range(8):
  a0,a1,c=rows('A',node,device,25),rows('A',node,device,26),rows('C',node,device,25);cm={z['point']:z for z in c}
  if node==0 and device==0:q['inventory_rank0']=[{'point':z['point'],'shape':z['shape'],'dtype':z['dtype']}for z in c]
  for j,aa in enumerate([a0,a1]):
   for a in aa:
    b=cm.get(a['point'])
    if b and len(a['shape'])==2 and a['shape'][0]==1 and b['shape']==[8,a['shape'][1]]:
     da=data('A',a);dc=data('C',b);q['matrix_points'].append({'rank':node*8+device,'position':536+j,'point':a['point'],'dtype':a['dtype'],**diff(da,dc[j*len(da):(j+1)*len(da)],a['dtype'])})
  finalA={z['point']:z for z in rows('A',node,device,32)}
  for tag in ['conv_after','ssm_after']:
   a,b=finalA[tag+'_row_1'],cm[tag+'_row_9'];q['final_state'].append({'rank':node*8+device,'point':tag,'A_active':1,'C_active':9,**diff(data('A',a),data('C',b),a['dtype'])})
for arm,ncalls in [('A',33),('C',26)]:
 files=list((raw/arm).glob('*.jsonl'));assert len(files)==ncalls*16,(arm,len(files))
 for p in files:
  v=[json.loads(x)for x in p.read_text().splitlines()];q['all_journals']['count']+=1;q['all_journals']['snapshot_records']+=len(v);q['all_journals']['nonfinite']+=sum(x['nonfinite']for x in v)
original=json.loads((r/'results/kimi-crossover-r9026-TP16PP2-I1928669zh-8e319d252bb323d5-node2.json').read_text())
for arm in ['A','C']:
 found=None
 for call in original['crossover_calls']:
  if call['arm']!=arm:continue
  for owner in call['owners']:
   if owner['node']!=0:continue
   for z in owner['trace_after']['records']:
    if z['layer']==0 and z['point']=='attn_norm'and 536 in z['positions']:found=z
 assert found
 fresh={z['point']:z for z in rows(arm,0,0,25)}
 for point,key in [('conv_before_row_1','conv_read'),('ssm_before_row_1','ssm_read')]:q['reference_cache_binding'].append({'arm':arm,'position':536,'point':point,'old678_fnv1a64':found['recurrent_rows_at_boundary'][key]['fnv1a64'],'new685_fnv1a64':fresh[point]['fnv1a64'],'equal':found['recurrent_rows_at_boundary'][key]['fnv1a64']==fresh[point]['fnv1a64']})
p=r/'results/Kimi-layer0-existing-raw-analysis-r9039.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
