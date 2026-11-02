from pathlib import Path
import json,hashlib,sys
D=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
RECEIPT='895304d5617b8bc01cfd291c9ce6165733385ee7d000f2e085561a226b7a2064'
LIB='a116adc27d070aea76c65b54527118f0653461620e55d1224c63d0d430f300e7'
MF='228f44719936ed966ceecfc678e976b40179b74e48e8d461609df46cb3521553'
def read(path):
 o=json.loads(Path(path).read_bytes());q=o['receipt']['data'];assert o['receipt']['SHA256']==RECEIPT==sha(json.dumps(q,indent=2).encode())
 assert any(x.split('|')[0]=='910407.802'and x.split('|')[2:4]==['COMPLETED','0:0']for x in o['accounting'])
 assert q['completed']and q['HOST_TUs_compiled']==q['GPU_TUs_compiled']==q['link_invocations']==q['GPU']==0 and not q['Native_dlopen']and not q['checkpoint_payload_IO']
 assert q['original_CPU800_remains_FAILED']and q['private_template_exception_only']and not q['all_parent_exports_retained']and q['all_parent_strong_retained']and q['all_parent_CABI_retained']and q['all_other_parent_weak_retained']and q['ldd_r_parent_equal']
 assert q['source_manifest_SHA256']==MF and q['library_SHA256']==LIB
 g=D/'gate-only-v2';assert sha((g/'CPU_SOURCE_MANIFEST.json').read_bytes())==MF
 for n,h in json.loads((g/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha((g/n).read_bytes())==h,n
 assert q['exact_removed_private_weak']==json.loads((g/'SYMBOL_ALLOWLIST.json').read_bytes())['missing_exact']
 original=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
 for n,h in original['files_SHA256'].items():assert sha((D/n).read_bytes())==h,n
 assert q['actual_chunk_contract_inherited_CPU800']['passed']and q['actual_chunk_contract_inherited_CPU800']['U32768_calls']==4
 assert q['actual_recording_contract_inherited_CPU800']['schedule']['inspect_failure_primary_persisted']and q['actual_recording_contract_inherited_CPU800']['schedule']['query_calls']==6 and q['actual_recording_contract_inherited_CPU800']['schedule']['seed_calls']==384
 return dict(passed=True,CPU='910407.802',receipt_SHA256=RECEIPT,library_SHA256=LIB,source_manifest_SHA256=MF,original800_FAILED=True,compile=0,link=0,GPU=0,Native=False,exact_private_weak_exceptions=2,strong_CABI_retained=True,ldd_parent_equal=True)
if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
