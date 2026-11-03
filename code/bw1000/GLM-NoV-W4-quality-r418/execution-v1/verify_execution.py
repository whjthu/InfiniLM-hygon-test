"""Pure original-byte checks for one fixed two-replica quality package."""
from pathlib import Path,PurePosixPath
import hashlib,json
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def host(root,value):
 p=PurePosixPath(value);return Path(root).joinpath(*p.relative_to('/refactor').parts)
def verify(root,source,digest):
 R=Path(root);S=Path(source);D=S/'execution-v1';p=D/'GPU_MANIFEST.json';assert sha(p)==digest;m=json.loads(p.read_bytes())
 assert sha(D/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256']
 for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(D/'owner/SOURCE_MANIFEST.json')==m['loader_manifest_SHA256']
 for n,h in json.loads((D/'owner/SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/'owner'/n)==h,n
 policy=R/'incoming/GLM-NoV-W4-normal-profile-r417';assert sha(policy/'SOURCE_MANIFEST.json')==m['profile_manifest_SHA256']=='695b1a12bfd2929851b3ffa89a9267564f9fd9dc0045c4a9adfbcf7142b21584'
 for n,h in json.loads((policy/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(policy/n)==h,n
 assert sha(S/'client-v1/SOURCE_MANIFEST.json')==m['client_manifest_SHA256']=='895314245a39fe3c5e8be917033782a29d0cfe978ea3923c47e84a8df0f8112b'
 for n,h in json.loads((S/'client-v1/SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(S/'client-v1'/n)==h,n
 assert sha(S/'client-v1/quality_client.py')==m['client_source_SHA256']
 receipt=R/'candidates/GLM-NoV-W4-quality-encoding-r418-CPU-v2/encoding.json';assert sha(receipt)==m['encoding_receipt_SHA256']=='2051a3a74e5ce471b53c38ddffd0f744e1a0669bdc2312590f6bd6d2a5013c12';q=json.loads(receipt.read_bytes());assert q['completed']and q['passed']and len(q['cases'])==64 and q['source_manifest_SHA256']==m['encoding_source_SHA256']=='e9570b0e6d797bb2399e7fac58fd34c53093bbc8b92b670dc124f2bedc6bc149'
 assert not q['Torch_CUDA_initialized']and not q['unexpected_Native_library_maps']and not q['Native_constructed']and not q['weights_loaded']
 F=S/'fixtures-r1';assert sha(F/'MANIFEST.json')=='c172ddf32b3e414066bcf91d2b0766ed89be763041e8a5ec5f873e7c2383c9df'
 for n,h in json.loads((F/'MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(F/n)==h,n
 native=R/'candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json';assert sha(native)=='241c1af8a7ff7c7f689ea7e14cc32971beed70aced62e37413c60648400ddc32';b=json.loads(native.read_bytes());assert sha(host(R,b['library']))==m['Native_SHA256']=='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
 assert(m['TP'],m['PP'],m['rows'],m['context'],m['cards'],m['replicas'],m['HTTP'],m['Native'],m['warmups'])==(8,2,1,32768,32,2,128,128,0)
 assert(m['service_max_s'],m['Native_max_s'],m['GPU_max_s'],m['keeper_max_s'],m['Native_cleanup_reserve_s'])==(13500,14100,14400,15000,600)
 assert m['nodes']==[['a02r4n12','a02r4n13'],['a02r4n14','a02r4n15']]
 return m
