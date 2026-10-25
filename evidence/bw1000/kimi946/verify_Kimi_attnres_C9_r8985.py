import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text());assert d['status']=='COMPONENT_COMPLETED_NOT_MODEL_QUALIFIED' and d['shutdown_entered'] and d['shutdown_returncode']==0
arms=d['arms'];assert len(arms)==11
assert [x['phase']for x in arms]==['warm']*2+['measure']*8+['restore']
assert [x['mode']for x in arms]==[0,1,0,1,1,0,1,0,0,1,0]
assert all(x['finite']and x['canary']for x in arms)
assert arms[-1]['counter']==[[1,1],[4,4],[1,0]]
assert d['valid_bank_stride']==[64512,7168,1]
print(json.dumps({'passed':True,'calls':11,'warm':2,'measure':8,'restore':1,'one_owner':True,'level':'synthetic_component_not_Model_qualified'}))
