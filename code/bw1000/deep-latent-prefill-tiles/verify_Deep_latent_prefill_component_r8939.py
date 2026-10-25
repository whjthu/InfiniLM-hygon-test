import pathlib,json,sys,hashlib,math
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['status']=='QUALIFIED_COMPONENT_PENDING_AUDIT'and len(q['normal_owners'])==8 and sorted(v['device']for v in q['normal_owners'])==list(range(8))and len(set(v['thread']for v in q['normal_owners']))==8 and all(q['ownrepeat']);assert [v['arm']for v in q['timed']]==[0,1,1,0,1,0,0,1];assert len(q['quality'])==4
for t in q['timed']:
 assert sorted(v['device']for v in t['owners'])==list(range(8))and all(math.isfinite(v['ms'])and v['ms']>0 for v in t['owners'])
for c in q['quality']:
 assert len(c['error'])==8 and all(all(math.isfinite(v[k])for k in ['nrmse','maxabs','p99abs'])for v in c['error'])
print(json.dumps(dict(passed=True,normal_owners=8,component_only=True,Model_gain=False,quality=q['quality'],timed=q['timed'],source_report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())))
