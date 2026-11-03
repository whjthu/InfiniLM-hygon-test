"""Exact completed CPU889 scope, preserved CPU888 failure, device not inherited."""
from pathlib import Path
import hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(m,source,host):
 assert m['GPU_GO']is True and m['Model_GO']is False
 assert all(sha(source/n)==h for n,h in m['source_SHA256'].items())
 p=host(m['CPU_receipt']);assert sha(p)==m['CPU_receipt_SHA256'];q=json.loads(p.read_bytes())
 assert q['completed']is True and q['passed']is True and q['actual_CPU']=='910407.889'
 assert q['HOST_compile_attempts']==1 and q['HIP_compile_attempts']==1 and q['reused_production_HOST_count']==2
 assert q['CPU888_preserved_failed']is True and q['actual_reused_CPU_semantics']['passed']and len(q['actual_reused_CPU_semantics']['cases'])==10 and q['actual_reused_typed_consumer']['passed']
 assert sha(host(m['bridge']))==m['bridge_SHA256']==q['library_SHA256']
 assert q['pinned_Native_DT_NEEDED']==m['Native']and sha(host(m['Native']))==m['Native_SHA256']
 for item in m['compiled_objects']:assert sha(host(item['object']))==item['object_SHA256']and sha(host(item['MD_path']))==item['MD_SHA256']
 assert sha(host(m['metadata_executable']))==m['metadata_executable_SHA256']
 return q
