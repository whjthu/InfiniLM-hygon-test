"""Exact artifact qualification; never reinterpret CPU851's producer status."""
from pathlib import Path
import json,hashlib
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate(m,source,host):
 assert m['GPU_GO']is True and m['Model_GO']is False
 assert all(sha(source/n)==h for n,h in m['source_SHA256'].items())
 p=host(m['CPU_original_receipt']);assert sha(p)==m['CPU_original_receipt_SHA256'];raw=json.loads(p.read_bytes())
 assert raw['completed']is True and raw['passed']is False and raw['actual_CPU']=='910407.851'
 cp=source/'CPU_ARTIFACT_CLOSURE.json';assert sha(cp)==m['CPU_existing_artifact_closure_SHA256'];c=json.loads(cp.read_bytes())
 assert c['original_CPU_producer_passed']is False and c['read_only_verifier_passed']is True and c['existing_artifact_CPU_math_and_link_checks_passed']is True
 assert c['original_validate_pass_count']==13 and c['known_CPU_backend_classified_cases']==['negative_underflow']
 assert sha(host(m['bridge']))==m['bridge_SHA256']==c['library']['SHA256']
 assert c['private_DSO_DT_NEEDED'].count(m['Native'])==1 and sha(host(m['Native']))==m['Native_SHA256']
 assert sha(host(m['metadata_executable']))==m['metadata_executable_SHA256']
 for item in m['compiled_objects']:assert sha(host(item['object']))==item['object_SHA256']and sha(host(item['MD_path']))==item['MD_SHA256']
 return c
