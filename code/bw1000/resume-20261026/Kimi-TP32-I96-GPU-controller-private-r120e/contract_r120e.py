"""CPU-only shared gates. Never imports Torch, loads a DSO, or submits work."""
import pathlib,json,hashlib,re,os
ROOT=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
INC=ROOT/'incoming/Kimi-TP32-I96-GPU-controller-r120e'
SOURCE=ROOT/'incoming/Kimi-TP32-I96-MoE-r120'
LABEL='Kimi-TP32-I96-once-r120e'
NODES=['a02r4n08','a02r4n09','a02r4n10','a02r4n11']
CPU='/refactor/candidates/Kimi-TP32-I96-MoE-r120d/CPU-build.json'
CPU_SHA='f206430f2b313fb818375e223527038adb256170450aa88f09fe6cacbf3a1e3f'
DSO_SHA='91e5af672ef9e1bad95bff612c85ed7d4585e22612f60df39d5ee7fea49a0182'
NATIVE_SHA='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def mapped(p):return pathlib.Path(str(p).replace('/refactor',str(ROOT),1))
def clean_env():return {k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
def gate(manifest_sha,go_file,container=False):
 inc=pathlib.Path('/refactor/incoming/Kimi-TP32-I96-GPU-controller-r120e')if container else INC
 src=pathlib.Path('/refactor/incoming/Kimi-TP32-I96-MoE-r120')if container else SOURCE
 mp=inc/'manifest-r120e.json';assert sha(mp)==manifest_sha;m=json.loads(mp.read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 gp=pathlib.Path(go_file).resolve();assert gp.is_relative_to(pathlib.Path('/refactor').resolve()if container else ROOT.resolve())
 g=json.loads(gp.read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['authorized_by']=='parent-root'
 assert g['controller_manifest_SHA256']==manifest_sha and g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
 assert g['CPU_receipt_SHA256']==CPU_SHA and g['nodes']==NODES and g['parent_allocation']=='910375'and g['master_port']==m['master_port']
 cp=pathlib.Path(CPU)if container else mapped(CPU);assert sha(cp)==CPU_SHA;c=json.loads(cp.read_bytes())
 assert c['completed']and c['CPU_only']and not c['GPU']and not c['GPU_initialized']and not c['dlopen']and c['normal_CPU_exit']and c['producer']=='910407.443'
 assert c['actual_fresh_MD_set_equal']and c['HOST_TUs_compiled']==0 and c['reused_compiled_HOST_TUs']==1 and c['GPU_TUs_compiled']==0 and not c['relink']
 assert c['library_SHA256']==DSO_SHA and c['native_SHA256']==NATIVE_SHA
 for key in ['library','native']:assert sha(c[key]if container else mapped(c[key]))==c[key+'_SHA256']
 sm=src/'source-manifest-r120.json';assert sha(sm)==m['original_source_manifest_SHA256']==c['source_manifest_SHA256'];s=json.loads(sm.read_bytes())
 for n,h in s['package_source_SHA256'].items():assert sha(src/n)==h,n
 for p,h in s['pinned_libraries'].items():assert sha(p if container else mapped(p))==h,p
 assert c['watched_before_SHA256']==c['watched_after_SHA256']
 for p,h in c['dependency_SHA256'].items():
  if container or p.startswith('/refactor/'):assert sha(p if container else mapped(p))==h,p
 for p,h in m['preserved_failed_receipts'].items():
  pp=pathlib.Path(p)if container else mapped(p);assert sha(pp)==h and not json.loads(pp.read_bytes())['completed']
 return m,g,c
