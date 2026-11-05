"""HOST pure metadata; actual nonserve template CLI proof is original CPU978."""
from pathlib import Path
import hashlib,json
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
def check(deploy,provider,m):
 d,provider=Path(deploy),Path(provider);assert H(provider/'SOURCE_MANIFEST.json')==m['provider_source_SHA256']=='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec'
 c=json.loads((d/'config.json').read_bytes());expected=json.loads((provider/'config.template.json').read_bytes());expected.update(execute_authorized=True,experimental_profile='deep-hc0-fixed-cohort-c4',runtime_directory='/refactor/results/Deep-public-fixed-cohort-r465',node='a02r4n15',Native_seconds=14400,service_seconds=13800);assert c==expected
 assert(c['rows'],c['context'],c['HC_mode'],c['fullpre'],c['r366'])==(4,32768,0,0,0)
 ref=m['CPU978'];p=R/Path(ref['path']).relative_to('/refactor');assert H(p)==ref['SHA256'];q=json.loads(p.read_bytes())
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.978'and q['source_manifest_SHA256']==m['provider_source_SHA256']
 assert not any(q[k]for k in['GPU','Native_dlopen','SDK','weights','new_compile','reencode','Model_READY'])
 assert q['typed_binding']['rows']==4 and q['typed_binding']['objects']==50 and q['actual_nonserve_CLI']['experimental_profile']=='deep-hc0-fixed-cohort-c4'
 plan=R/Path('/refactor/candidates/Deep-public-C4-remaining-quality-CPU-v2/shard6-private.json').relative_to('/refactor');assert H(plan)==m['private_plan_SHA256'];v=json.loads(plan.read_bytes())
 assert v['cases']==32 and v['HTTP']==64 and v['ordinals']==[22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99, 106, 113, 120, 127, 150, 157, 164, 171, 178, 185, 192, 199, 206, 213, 220, 227, 234, 241, 248, 255]and v['encoding_CPU']=='910407.945'and not v['new_encode']
 assert len(v['records'])==32 and len({r['case_id']for r in v['records']})==32
 for r in v['records']:assert r['body']['max_tokens']==4096 and r['body']['chat_template_kwargs']['enable_thinking']is False and len(r['prompt_token_ids'])+4096<=32768
 return dict(passed=True,CPU978_template_nonserve_CLI=q['actual_nonserve_CLI'],actual_final_config_CLI_verified=False,final_config_CLI_required_before_Native='inside.sh first nonserve exactconfig; failure before Model load',source_and_config_metadata_verified=True,questions=32,HTTP=64,Native_dlopen=False,Slurm=0,GPU=0,Model_READY=False)
