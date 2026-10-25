import pathlib,json,hashlib,sys,math
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed'] and q['status']=='COMPLETED_PACKED_BATCH_COMPONENT_ONLY'
assert len(q['normal_owners'])==8 and all(v['normal'] for v in q['normal_owners'])
assert [v['kind'] for v in q['fixtures']]==['swa','hca128','csa4']
PLAN=[0,1,2,0,1,2,2,1,0,0]
E={'swa':[1,0,508,4,1540,2048,3],'hca128':[1,0,512,12,1536,2096,3],'csa4':[1,0,512,384,1536,3584,3]}
for c in q['fixtures']:
 assert len({v['seed']for v in c['owners']})==len({v['operand_sample_SHA256']for v in c['owners']})==8
 assert c['ownrepeat_A'] and c['ownrepeat_B'] and c['ownrepeat_C']
 assert [(v['output'],v['phase'],v['arm']) for v in c['measurements']]==[(i,'warm'if i<3 else'measure'if i<9 else'restore',a)for i,a in enumerate(PLAN)]
 for row in c['measurements']:
  a=row['arm'];e=list(E[c['kind']]) if a else [0,0,2048,0,0,8192,0]
  if a==2:e[5]+=2*e[4]-2*e[3]
  assert len(row['owners'])==8 and row['in_performance_statistics']==(row['phase']=='measure')
  for v in row['owners']:assert v['CAPI_calls']==1 and v['counter_delta']==[0,1,0,3] and v['packed_counter_delta']==e and math.isfinite(v['seconds']) and v['seconds']>0
 for rows in c['comparisons'].values():
  for v in rows:assert all(math.isfinite(v[k]) and v[k]>=0 for k in ['maxabs','relativeRMS','ULP_max','ULP_mean'])
print(json.dumps({'passed':True,'actual_handle':q['step'],'normal_owners':8,'maps':3,'plan':'3warm+ABC/CBA+restoreA','intermediate_QK_PV':'UNKNOWN_NOT_CAPTURED','Softmax_dispatch_rowcount':'UNKNOWN','crossarm_bitexact_gate':False,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
