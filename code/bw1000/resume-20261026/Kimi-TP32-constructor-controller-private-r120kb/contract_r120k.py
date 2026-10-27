"""CPU-only immutable source/CPU/GO gates, before any GPU/DSO entry."""
import pathlib,json,hashlib,re,os
ROOT=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');INC=ROOT/'incoming/Kimi-TP32-constructor-controller-r120k';SOURCE=ROOT/'incoming/Kimi-TP32-Native-r120f';LABEL='Kimi-TP32-construct-once-r120k';NODES=['a02r4n08','a02r4n09','a02r4n10','a02r4n11']
CPU='/refactor/candidates/Kimi-TP32-Native-r120f/CPU-build.json';CPU_SHA='fe7af006875dbef35077bd921bb4147423e0fcf689ae8591a9ab01b4d7c1fd5d';NATIVE_SHA='0f1347909ad24d37716a987fb736182cc1e35e81b92b4fd94e1aafac7bdbfe92';SOURCE_SHA='4072d9b9c2576e35748b356604a7a83d1cbe1930d7f1c5900bfbae33e4440ad7';WRAPPER_CPU='/refactor/candidates/Kimi-TP32-construct-r120j/CPU-build.json';COMMON='/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/_infinilm.cpython-310-x86_64-linux-gnu.so';COMMON_SHA='2e1458609cc81a31351c451004c045b8e84f7581844afedf1a577df2c940ccb2';REPAIR_SHA='63b4c6bf35953448e6496b76dbd5811ebfba3389fda7a1ed829d5afd0968cfc4';MODE='private_Native0f_plus_explicit_existing_Common2e14';WRAPPER_CPU_SHA='087f1be2717681ddfd5b5697fc1f93c7cb52d62c50ddf87469d4597e602cf8b3';WRAPPER_SHA='a891091529b621f95468265c1e2617ece2c6b0761a2ec05e9453b1c76bdc94d8'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def mapped(p):return pathlib.Path(str(p).replace('/refactor',str(ROOT),1))
def clean_env():return {k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
def gate(manifest_sha,go_file,container=False):
 inc=pathlib.Path('/refactor/incoming/Kimi-TP32-constructor-controller-r120k')if container else INC;src=pathlib.Path('/refactor/incoming/Kimi-TP32-Native-r120f')if container else SOURCE
 mp=inc/'manifest-r120k.json';assert sha(mp)==manifest_sha;m=json.loads(mp.read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 gp=pathlib.Path(go_file).resolve();assert gp.is_relative_to(pathlib.Path('/refactor').resolve()if container else ROOT.resolve());g=json.loads(gp.read_bytes())
 assert g['GPU_authorized']is True and g['source_published']is True and g['authorized_by']=='parent-root'and g['constructor_only']is True and g['fullModel_GO']is False
 assert g['controller_manifest_SHA256']==manifest_sha and g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
 assert g['Native_CPU_receipt_SHA256']==CPU_SHA and g['Native_SHA256']==NATIVE_SHA and g['nodes']==NODES and g['parent_allocation']=='910375'and g['master_port']==m['master_port']
 assert g['Common_SHA256']==COMMON_SHA and g['provider_mode']==MODE
 for k in ['wrapper_CPU_receipt_SHA256','wrapper_SHA256']:assert re.fullmatch('[0-9a-f]{64}',g[k]),k
 assert g['wrapper_CPU_receipt_SHA256']==WRAPPER_CPU_SHA and g['wrapper_SHA256']==WRAPPER_SHA and g['wrapper_actual_CPU_step']=='910407.464'
 cp=pathlib.Path(CPU)if container else mapped(CPU);wp=pathlib.Path(WRAPPER_CPU)if container else mapped(WRAPPER_CPU);assert sha(cp)==CPU_SHA and sha(wp)==g['wrapper_CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());w=json.loads(wp.read_bytes())
 assert c['completed']and c['CPU_only']and not c['GPU']and not c['GPU_initialized']and not c['dlopen']and c['normal_CPU_exit']and c['producer']=='910407.454'and c['HOST_TUs_compiled']==5 and c['GPU_TUs_compiled']==0 and c['pure_CPU_contract']['passed']
 assert c['source_manifest_SHA256']==SOURCE_SHA and c['native_SHA256']==NATIVE_SHA
 assert w['completed']and w['CPU_only']and not w['GPU']and not w['GPU_initialized']and not w['dlopen']and w['normal_CPU_exit']and w['HOST_TUs_compiled']==0 and w['GPU_TUs_compiled']==0 and w['reused_compiled_constructor_HOST_TUs']==1
 assert w['parent_CPU_receipt_SHA256']==CPU_SHA and w['native_SHA256']==NATIVE_SHA and w['library_SHA256']==g['wrapper_SHA256']and w['producer']==g['wrapper_actual_CPU_step']
 assert w['repair_manifest_SHA256']==REPAIR_SHA and w['provider_mode']==MODE and w['Common_provider']==COMMON and w['Common_SHA256']==COMMON_SHA and w['explicit_direct_Common_dependency']and w['no_new_Native_objectset_or_math']
 assert w['watched_before_SHA256']==w['watched_after_SHA256']and w['construct_object_SHA256']=='42dfc445a865e7bef440eccfac36f78338bafcef0cbae3cf04f2c7124a93aa95'
 assert w['preserved_failed457_receipt_SHA256']=='a21a9e10e33306e48a07894301eedd7d4f0fdfdb5d05d363c390c78a340cca6b'and w['preserved_failed460_receipt_SHA256']=='9e797c0bc698226ad301381b129b47bf82373b1e0f5004dd9f259510af797045'
 assert sha(COMMON if container else mapped(COMMON))==COMMON_SHA and all(x['actual_public_provider']==COMMON and x['provider_SHA256']==COMMON_SHA for x in w['actual_Common_provider_audit'])
 for p,h in [(c['native'],NATIVE_SHA),(w['library'],g['wrapper_SHA256'])]:assert sha(p if container else mapped(p))==h,p
 sm=src/'manifest-r120f.json';assert sha(sm)==SOURCE_SHA;s=json.loads(sm.read_bytes())
 for n,h in s['source_SHA256'].items():assert sha(src/n)==h,n
 for p,h in c['required_DSO_SHA256'].items():assert sha(p if container else mapped(p))==h,p
 for p,h in c['all_linked_object_SHA256'].items():assert sha(p if container else mapped(p))==h,p
 for p,h in w['actual_linked_object_SHA256'].items():assert sha(p if container else mapped(p))==h,p
 for p,h in w['fresh_MD_SHA256'].items():
  if container or p.startswith('/refactor/'):assert sha(p if container else mapped(p))==h,p
 # Original false templates are source pins, never edited into executable GO.
 assert json.loads((src/'fullNative-manifest-TEMPLATE-r120f.json').read_bytes())['full_working_set_gate']['qualified']is False
 assert json.loads((src/'constructor-runtime-TEMPLATE-r120f.json').read_bytes())['explicit_constructor_GO']is False
 return m,g,c,w
