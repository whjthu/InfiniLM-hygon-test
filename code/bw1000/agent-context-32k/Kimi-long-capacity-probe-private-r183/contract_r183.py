"""Host CPU gate only. No GPU/Native import or job submission on import."""
import pathlib,json,hashlib,os,re
ROOT=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');INC=ROOT/'incoming/Kimi-long-capacity-probe-r183';LABEL='Kimi-long-capacity-r183';NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'];PARENT='910380'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def host(p):return ROOT/str(p).removeprefix('/refactor/')if str(p).startswith('/refactor/')else pathlib.Path(p)
def clean_env():
 e={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:e.pop(k,None)
 return e

def gate(manifest_sha,go_file):
 mf=INC/'source-manifest-r183.json';assert sha(mf)==manifest_sha;m=json.loads(mf.read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(INC/n)==h,n
 p=json.loads((INC/'pins-r183.json').read_bytes());assert sha(host(p['CPU_receipt']))==p['CPU_receipt_SHA256'];q=json.loads(host(p['CPU_receipt']).read_bytes());assert q['completed']and q['CPU_only']and not q['GPU']and not q['Native_dlopen']and q['native_SHA256']==p['Native_SHA256'];assert q['new_HOST_TUs']==6 and q['new_HIP_TUs']==0 and sha(host(p['Native']))==p['Native_SHA256']
 for f,h in q['all_linked_object_SHA256'].items():assert sha(host(f))==h,f
 g=json.loads(pathlib.Path(go_file).read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['source_manifest_SHA256']==manifest_sha
 assert g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
 assert g['parent']==PARENT and g['nodes']==NODES and g['cards']==32 and g['only_constructor_two_Loading_resets_destroy']is True
 assert g['weight_payload_allowed']is False and g['finish_or_forward_allowed']is False and g['Model_GO']is False
 assert g['CPU_receipt_SHA256']==p['CPU_receipt_SHA256']and g['Native_SHA256']==p['Native_SHA256']
 return m,g,p
