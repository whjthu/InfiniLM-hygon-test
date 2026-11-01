from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;OUT=D/'runtime-r1';repo=D.parents[2];results=repo/'results/bw1000/2026-11-02';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
probe=json.loads((results/'GLM361-r1-final-dependency-probe.json').read_bytes());assert probe['source_and_355_ancestry_verified'] and probe['namespace_verified']and not probe['GPU_submitted']and probe['full_package_members']==57
base=results/'GLM355-final-audit.json';b=json.loads(base.read_bytes());assert b['all_independent_post_passed']and b['owner_count']==32 and b['all32_normal_destroy']and b['quality_HTTP_total']==12 and b['cost']['completed']and not b['cost']['old719_gain_inherited']
claimpath=results/'GLM361-r1-source-CPU-resource-claim.json';claim=json.loads(claimpath.read_bytes());claim.update(state='SOURCE_CPU707_HOST_DEPENDENCY_CLOSED_READY_ONE_PUBLISHED_RESIDENCY',final_dependency_probe='results/bw1000/2026-11-02/GLM361-r1-final-dependency-probe.json',baseline_355_lifecycle_and_formal_cost_consumed=True,baseline_355_final_audit_path='results/bw1000/2026-11-02/GLM355-final-audit.json',baseline_355_final_audit_SHA256=sha(base),Astra158_route_selected=True,baseline_355_root_independent_timeline_review='Root executing separately; this packet consumes existing actual355 lifecycle/cost and does not inherit prior gain or claim that root review completed')
claimpath.write_bytes((json.dumps(claim,indent=2)+'\n').encode())
root_review=results/'GLM355-root-audited-C2-deferred-decode-scoped-PASS-clean.json'
if root_review.is_file():
 claim.update(baseline_355_root_independent_timeline_review='COMPLETE_PASS_SCOPED; no15/online claim',baseline_355_root_audit_path=root_review.relative_to(repo).as_posix(),baseline_355_root_audit_SHA256=sha(root_review),baseline_355_root_audit_publication='078f3b894eca84f880668bc2253e49eeb10474fb')
 claimpath.write_bytes((json.dumps(claim,indent=2)+'\n').encode())
local=json.loads((OUT/'controller-CPU-local.json').read_bytes());assert local['passed']and len(local['loop_cases'])==6 and local['GPU']==0
(results/'GLM361-r1-recording-controller-CPU.json').write_bytes((json.dumps(local,indent=2)+'\n').encode())
taskpath=repo/'coordination/tasks/GLM-prefill-window-interleave-r361.json';task=json.loads(taskpath.read_bytes());task.update(owner='/root/glm_operator_tuning',state='FULL_LINUX_CPU707_AND_HOST_DEPENDENCY_PASS_READY_PUBLISHED_ONE_RESIDENCY',original_snapshot_SHA256='11cf0416e344e0fe8795c809c2cf54a87588f9dc6c8093a8d203749068deed8e',original_snapshot_edited=False,runtime='code/bw1000/GLM-prefill-window-interleave-r361/runtime-r1',runtime_owner_manifest_SHA256=probe['owner_manifest_SHA256'],runtime_controller_manifest_SHA256=probe['controller_manifest_SHA256'],CPU_package_manifest_SHA256=probe['CPU_package_manifest_SHA256'],actual_CPU='910407.707',CPU707_receipt_SHA256=probe['CPU707_receipt_SHA256'],CPU='Actual new owner→actual callback→actual client12 gold at recording Native/HTTP boundary;32 actual owner loops/create-call AST/normal cleanup; cancel/partial error no replay and actual guarded exit2; no Native/GPU/weights',full_owner_Linux_qualification='CPU lifecycle/glue PASS, Native numerical/Model qualification not asserted',GPU_GO=False,GPU_started=False,GPU_claim='results/bw1000/2026-11-02/GLM361-r1-source-CPU-resource-claim.json',baseline_355_lifecycle_and_formal_cost_consumed=True,baseline_355_final_audit_SHA256=sha(base),Astra158_selected=True,immutable_new_package=True,legacy355_or_snapshot_source_edited=False,first_warm_path_failure='STOP_NO_SOURCE_EDIT_RETRY_OR_RELOAD',root359_362_status_untouched=True)
taskpath.write_bytes((json.dumps(task,ensure_ascii=False,indent=2)+'\n').encode())
if root_review.is_file():
 task.update(baseline_355_root_audit_path=root_review.relative_to(repo).as_posix(),baseline_355_root_audit_SHA256=sha(root_review),baseline_355_root_independent_review='COMPLETE_PASS_SCOPED')
 taskpath.write_bytes((json.dumps(task,ensure_ascii=False,indent=2)+'\n').encode())
mf=json.loads((OUT/'CPU_PACKAGE_MANIFEST.json').read_bytes());paths=[]
for n,h in mf['files_SHA256'].items():p=OUT/n;assert sha(p)==h,n;paths.append(p)
paths.append(OUT/'CPU_PACKAGE_MANIFEST.json')
for n in ['prepare_runtime_r1.py','prepare_CPU_r1.py','freeze_runtime_r1.py','prepare_claim_r1.py','read_CPU_r1.py','prepare_scoped_stage_r1.py']:paths.append(D/n)
for n in ['GLM361-CPU707-PASS.json','GLM361-r1-final-dependency-probe.json','GLM361-r1-source-CPU-resource-claim.json','GLM361-r1-recording-controller-CPU.json','GLM361-CPU-r1-submitted.json']:paths.append(results/n)
paths.append(taskpath);relative=sorted(set(p.relative_to(repo).as_posix()for p in paths));assert all('private-GO'not in p and 'remote_upload'not in p and 'request-00'not in p for p in relative)
(repo.parent/'GLM361-stage-paths.txt').write_bytes(b'\x00'.join(p.encode()for p in relative)+b'\x00')
print(json.dumps({'explicit_metadata_and_source_paths':len(relative),'runtime_members':len(mf['files_SHA256']),'original_snapshot_SHA256':sha(D/'SOURCE_MANIFEST.json'),'GPU_submitted':False}))
