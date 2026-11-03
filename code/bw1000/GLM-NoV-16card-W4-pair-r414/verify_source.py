"""Pure metadata/bytes verification reused by CPU CLI and authorized Model CLI."""
from pathlib import Path
import hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_source(root,source,digest,manifest_file='GPU_MANIFEST.json'):
 root=Path(root);source=Path(source);assert manifest_file in ['GPU_MANIFEST.json','GPU_MANIFEST.template.json']
 p=source/manifest_file;assert sha(p)==digest;m=json.loads(p.read_bytes())
 assert sha(source/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256']
 for n,h in json.loads((source/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(source/n)==h,n
 assert sha(source/'owner/SOURCE_MANIFEST.json')==m['owner_manifest_SHA256']
 for n,h in json.loads((source/'owner/SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(source/'owner'/n)==h,n
 parent=root/'incoming/GLM-V-on-demand-Model-r404';assert sha(parent/'GPU_MANIFEST_c.json')==m['parent_850_GPU_manifest_SHA256']=='7bc63bf491a54ce527f48191fb4710f41a23f12f9032f670e07e55c2ceb54739'
 assert sha(parent/'owner_v2/SOURCE_MANIFEST.json')==m['parent_850_owner_manifest_SHA256']=='8180a28a6bb7b5097a99d068fa08c7f880e0f347f5ae93a0a69c8e9b51aae5dc'
 p=root/'candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json';assert sha(p)==m['Native_CPU804_receipt_SHA256']=='241c1af8a7ff7c7f689ea7e14cc32971beed70aced62e37413c60648400ddc32';native=json.loads(p.read_bytes());assert native['library_SHA256']==m['Native_SHA256']=='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
 assert sha(root/Path(native['library']).relative_to('/refactor'))==m['Native_SHA256']
 p=root/'candidates/GLM-V-on-demand-Model-r404-CPU-layout-v4/CPU-result.json';assert sha(p)==m['CPU820_receipt_SHA256']=='58835b6366290498bfa6f757d23bc69ba790a090b782d2a30030ca5a1d239694';layout=json.loads(p.read_bytes());assert layout['passed']and layout['owner_manifest_SHA256']==m['parent_850_owner_manifest_SHA256']
 watch=json.loads((root/'results/GLM-V-on-demand-Model-r404-model-20261103-c-watch.json').read_bytes());assert watch['completed']and watch['passed']and watch['actual_GPU']=='910376.850'and watch['CPU_keeper']=='910407.822'and not watch['forced_cleanup']
 assert(m['TP'],m['PP'],m['rows'],m['context'],m['ordinary_output_budget'],m['HTTP'],m['Native'])==(8,2,1,32768,4096,6,6)
 assert m['fixture_tokens']==[9741]and m['windows']==[2,4,2,4,4,2]and m['warmups']==2 and m['formal_indices']==[2,3,4,5]
 assert(m['GPU_max_s'],m['Native_max_s'],m['service_max_s'],m['keeper_max_s'])==(5400,5100,4500,6000)
 assert sha(source/'controller/controller.py')==m['controller_SHA256']and sha(source/'controller/resource_wave_host_alias_outer.sh')==m['outer_wrapper_SHA256']
 return m
