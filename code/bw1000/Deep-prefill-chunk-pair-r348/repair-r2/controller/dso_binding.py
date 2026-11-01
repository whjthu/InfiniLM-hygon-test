"""Bind repaired owner sources to CPU621 lookup evidence and unchanged CPU613/612 contracts."""
import pathlib,json,hashlib
CPU621_SHA='d6afe489c669580bd381be2400aad57fd03a52ddd858356b3d8418479c34e3e6'
BASE={'GLM':('GLM-latest-full-owner-r290','903053b1f6d8031bc03c698c7bd9498a8cab8b0802ff19f4c88a7b4d698e7674',{'GPU_inside_r290.sh','model_entry_r290.py'},{'common_mapping_audit.py'}),'Deep':('Deep-latest-full-owner-r287','12a4160d5703a9364929c41d5b9b69772a600254f2084ec2774d585023b17df1',{'GPU_inside_r277.sh','bootstrap_r206b.py','preweight_r206b.py'},{'common_mapping_audit.py','legacy_mapping_patch.py'})}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def validate_receipt(q,model,owner_sha):
 assert model in BASE and q['completed']and q['passed']and q['CPU_only']and q['actual_CPU']=='910407.621'
 assert q['Model_qualified']is False and q['LD_PRELOAD_cleared_for_CPU']is True
 assert all(q[k]==0 for k in ['GPU','Model','HTTP','weights','Native_dlopen','runtime_imports'])
 phases=q['phases'];assert len(phases)==6
 actual={(p['model'],p['network'],p['phase'])for p in phases}
 assert len(actual)==6 and actual=={(m,n,l)for m,ns in [('GLM',['Socket','shca']),('Deep',['retained'])]for n in ns for l in ['control','candidate']}
 rows=[p for p in phases if p['model']==model and p['phase']=='candidate'];assert rows and all(p['owner_manifest_SHA256']==owner_sha for p in rows)
 return rows

def verify_dso(root,source,model):
 root=pathlib.Path(root);source=pathlib.Path(source);owner_sha=sha(source/'SOURCE_MANIFEST.json')
 cp=root/'candidates/common-dso-audit-r301/CPU-inside.json';assert sha(cp)==CPU621_SHA;q=json.loads(cp.read_bytes());rows=validate_receipt(q,model,owner_sha)
 name,old_sha,allowed,added=BASE[model];old=root/'incoming'/name;assert sha(old/'SOURCE_MANIFEST.json')==old_sha
 m=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());base=json.loads((old/'SOURCE_MANIFEST.json').read_bytes());assert m['previous_owner_manifest_SHA256']==old_sha and set(m['changed_members'])==allowed
 assert set(m['files_SHA256'])==set(base['files_SHA256'])|added
 for n,h in base['files_SHA256'].items():
  assert sha(old/n)==h
  if n not in allowed:assert m['files_SHA256'][n]==h and sha(source/n)==h
 return {'CPU621_inside_SHA256':CPU621_SHA,'source_manifest_SHA256':owner_sha,'inherited_CPU_owner_manifest_SHA256':old_sha,'CPU613_or_CPU612_reused_for_unchanged_client_deadline_cleanup_contracts_only':True,'scope':'CPU621 dependency lookup plus pinned source compatibility; no live binding, Model, ABI or performance proof'}
