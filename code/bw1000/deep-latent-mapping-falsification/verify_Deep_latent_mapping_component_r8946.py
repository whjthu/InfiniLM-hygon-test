import pathlib,json,hashlib,sys
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['status']=='COMPLETED_DIAGNOSTIC_NO_MODEL_QUALIFICATION'
assert [v['kind']for v in q['fixtures']]==['swa','hca128','csa4'];assert len(q['normal_owners'])==8 and all(v['normal']for v in q['normal_owners'])
for case in q['fixtures']:
 assert len({v['seed']for v in case['owners']})==8 and len({v['operand_sample_SHA256']for v in case['owners']})==8
 assert [(v['output'],v['width'])for v in case['measurements']]==[(0,2048),(1,2048),(2,512),(3,64)]
 for row in case['measurements']:
  assert len(row['owners'])==8
  for v in row['owners']:assert v['CAPI_calls']==2048//row['width']and v['counter_delta']==([1,2048,64]if row['output']==1 else[0,0,0])
 assert set(case['comparisons'])=={'(0, 1)','(0, 2)','(1, 3)'}
print(json.dumps({'passed':True,'actual_handle':q['step'],'normal_owners':8,'fixtures':3,'scope':'Engineeringfinite/count/schema auditonly;numericerrornotModelqualitygate','raw_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
