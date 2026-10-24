import pathlib,json,hashlib,sys
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_text());r=p.parents[1];tag=q['step'];log=r/'logs'/('deep-owner-device-probe-r8923-'+tag+'.log');rows=[]
for line in log.read_text(errors='replace').splitlines():
 if line.startswith('{'):
  try:rows.append(json.loads(line))
  except ValueError:pass
guarded=[v for v in rows if v.get('phase')=='guarded_complete'];old=[v for v in rows if v.get('phase')=='unguarded_record_return'];complete=[v for v in rows if v.get('phase')=='owner_complete'];assert q['completed']and q['native_returncode']==0 and len(guarded)==len(old)==len(complete)==8;assert sorted(v['device']for v in guarded)==list(range(8));assert all(v['before']==v['after']==0 and v['during']==v['device']for v in guarded)
print(json.dumps(dict(passed=True,normal_owners=8,guarded=guarded,unguarded_codes=old,source_report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),log_SHA256=hashlib.sha256(log.read_bytes()).hexdigest(),scope='NoModelweightsCCL;device/event mechanismonly,notproof891rootcause,notperformance.')))
