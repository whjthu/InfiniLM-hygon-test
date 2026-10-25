import pathlib,json,sys,math,hashlib
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['status']=='COMPLETED_SINK_BROADCAST_COMPONENT_ONLY';assert len(q['normal_owners'])==8 and all(v['normal']for v in q['normal_owners']);assert [v['kind']for v in q['fixtures']]==['swa','hca128','csa4']
for f in q['fixtures']:
 assert f['ownrepeat_A']and f['ownrepeat_B']and f['crossarm_bitcopy_exact'];assert [v['arm']for v in f['measurements']]==[0,1,0,1,1,0,0,1,0]
 for row in f['measurements']:
  assert len(row['owners'])==8
  for v in row['owners']:assert v['finite']and v['canary']and v['inputs_readonly']and v['seconds']>0 and math.isfinite(v['seconds'])and v['CAPI_calls']==1 and v['counter_delta']==[0,1,0,3];assert (v['broadcast_counter_delta'][0]>0)==bool(row['arm'])
print(json.dumps(dict(passed=True,handle=q['step'],normal_owners=8,three_maps=True,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
