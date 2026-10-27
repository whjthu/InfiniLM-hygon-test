"""Read-only named16 result metadata; no model execution, logs, tokens or environment."""
import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
audit=r/'results/GLM-critical-Model-r133-audit.json'
AUDIT_SHA='814ad28995f2172fe719c2c9a8facfbfb1e3f4626dec3af3b26649381c4ee164'
FILE_LIMIT=64*1024*1024;TOTAL_LIMIT=512*1024*1024
out={'scope':'diagnostic only;16 exact retained files; stat/SHA/producer only','expected_audit_SHA256':AUDIT_SHA,'per_file_read_limit_bytes':FILE_LIMIT,'total_read_limit_bytes':TOTAL_LIMIT,'files':[],'raw_content_exported':False}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  while True:
   block=f.read(1024*1024)
   if not block:break
   h.update(block)
 return h.hexdigest()
try:
 out['actual_audit_SHA256']=digest(audit)
 if out['actual_audit_SHA256']!=AUDIT_SHA:
  out['classification']='AUDIT_SHA_MISMATCH_STOP';print(json.dumps(out));raise SystemExit(0)
 a=json.loads(audit.read_bytes());owners=a['owners']
 if len(owners)!=16 or {(o['stage'],o['rank'])for o in owners}!={(s,k)for s in [0,1]for k in range(8)}:
  out['classification']='AUDIT_OWNER_SET_MISMATCH_STOP';print(json.dumps(out));raise SystemExit(0)
 total=0
 for o in owners:
  stage=o['stage'];rank=o['rank'];p=r/'results'/f"glm-arrival-r133-{a['run']}-stage{stage}-rank{rank}.json"
  row={'stage':stage,'rank':rank,'path':str(p),'expected_SHA256':o['SHA256'],'expected_producer':o['actual_step'],'old_size_gate_limit':10000000}
  out['files'].append(row)
  if not p.is_file():row['classification']=['FILE_MISSING_STOP'];continue
  st=p.stat();row.update(bytes=st.st_size,mtime_ns=st.st_mtime_ns,old_size_gate_passed=st.st_size<10000000)
  if st.st_size>FILE_LIMIT or total+st.st_size>TOTAL_LIMIT:
   row.update(classification=['BOUNDED_READ_LIMIT_STOP'],actual_SHA256=None,actual_producer=None);continue
  total+=st.st_size;row['actual_SHA256']=digest(p);row['SHA256_match']=row['actual_SHA256']==row['expected_SHA256'];row['classification']=[]
  if not row['SHA256_match']:row['classification'].append('RESULT_SHA_MISMATCH_STOP')
  try:
   with p.open('r',encoding='utf8')as f:q=json.load(f)
   producer=q.get('actual_producer');row['actual_producer']=producer if isinstance(producer,(str,int,type(None)))else{'type':type(producer).__name__}
   row['producer_match']=producer==row['expected_producer']=='910376.767'
   if not row['producer_match']:row['classification'].append('PRODUCER_MISMATCH_STOP')
  except Exception as e:row.update(parse_error_type=type(e).__name__,actual_producer=None,producer_match=False);row['classification'].append('JSON_PARSE_STOP')
  if not row['old_size_gate_passed']:row['classification'].append('OLD_10MB_LIMIT_ONLY'if row['SHA256_match']and row.get('producer_match')else'OLD_10MB_LIMIT_ALSO_FAILED')
  if not row['classification']:row['classification']=['ALL_THREE_ORIGINAL_GATES_PASS']
 out['total_bytes_read_once_for_hash']=total
 hard=any(not x.get('SHA256_match')or not x.get('producer_match')for x in out['files'])
 out['classification']='CONTRADICTION_OR_BOUNDED_READ_STOP'if hard else'SIZE_LIMIT_ONLY_SAFE_TO_EXPORT_WITH_64MIB_BOUND'if any(not x['old_size_gate_passed']for x in out['files'])else'ALL_ORIGINAL_GATES_PASS_FAILURE_NOT_REPRODUCED'
 print(json.dumps(out,separators=(',',':')))
except SystemExit:raise
except Exception as e:
 out.update(classification='DIAGNOSTIC_SOURCE_ERROR_STOP',error_type=type(e).__name__);print(json.dumps(out,separators=(',',':')))
