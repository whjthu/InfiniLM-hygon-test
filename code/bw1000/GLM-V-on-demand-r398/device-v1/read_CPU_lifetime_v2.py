from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent/'lifetime-v2'
MF='b799ed6f8ba0a8d36408d1f2d8fba82c0f4b5713ea369d146e3e51136628ce3c'
RECEIPT='92103aba80235b27780abf942a43ad2e4a3cdc80119143297285e768c89d9f7c'
LIB='fe3c3f39bc2903b2d6076a1275f25db9304579eccb0f8a6fb494c157e1f75eed'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(path):
 o=json.loads(Path(path).read_bytes());q=o['receipt']['data'];assert o['receipt']['SHA256']==RECEIPT==sha(json.dumps(q,indent=2).encode())
 assert any(x.split('|')[0]=='910407.798'and x.split('|')[2:4]==['COMPLETED','0:0']for x in o['accounting'])
 assert q['completed']and q['producer']=='910407.798'and q['HOST_TUs_compiled']==1 and not q['caller_object_recompiled']and q['GPU_TUs_compiled']==0
 assert not q['Native_dlopen']and not q['checkpoint_payload_IO']and q['GPU']==0 and q['added_sync']==0 and not q['math_changed']and not q['Runtime_exception_guarantee_assumed']
 assert q['all_parent_exports_retained']and q['ldd_r_parent_equal']and q['library_SHA256']==LIB and q['source_manifest_SHA256']==MF
 assert sha((D/'CPU_SOURCE_MANIFEST.json').read_bytes())==MF
 for n,h in json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha((D/n).read_bytes())==h,n
 assert sha((D/'attention_component_r398.cpp').read_bytes())==q['compiled']['source_SHA256']
 return dict(passed=True,CPU='910407.798',receipt_SHA256=RECEIPT,library_SHA256=LIB,source_manifest_SHA256=MF,HOST_TUs=1,caller_recompiled=False,GPU=0,Native=False,math_changed=False,added_sync=0,HOST_lifetime='Work-owned SelectionMap survives all exception unwinds through destructor body drain',Model_GO=False)
if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
