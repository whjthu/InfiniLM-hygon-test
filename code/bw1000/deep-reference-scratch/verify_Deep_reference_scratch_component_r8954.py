import pathlib,json,hashlib,sys,math
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['status']=='COMPLETED_REFERENCE_SCRATCH_COMPONENT_ONLY';assert len(q['normal_owners'])==8 and all(v['normal']for v in q['normal_owners']);assert [v['kind']for v in q['fixtures']]==['swa','hca128','csa4']
for c in q['fixtures']:
 assert len({v['seed']for v in c['owners']})==len({v['operand_sample_SHA256']for v in c['owners']})==8
 assert [(v['output'],v['arm'])for v in c['measurements']]==list(enumerate([0,1,1,0]));assert c['ownrepeat_A']and c['ownrepeat_B']
 for row in c['measurements']:
  assert len(row['owners'])==8
  for v in row['owners']:assert v['CAPI_calls']==1 and v['counter_delta']==([0,1,0,3]if row['arm']else[1,0,6144,0])and math.isfinite(v['seconds'])and v['seconds']>0
 for rows in c['comparisons'].values():
  for v in rows:assert all(math.isfinite(v[k])and v[k]>=0 for k in ['maxabs','relativeRMS','ULP_max','ULP_mean'])
print(json.dumps({'passed':True,'actual_handle':q['step'],'normal_owners':8,'fixtures':3,'fixed_ABBA_per_fixture':True,'crossarm_bitexact_gate':False,'source_report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Componentonly;originalFP32reference scratchreuse. NoModelquality/performanceadoption.'}))
