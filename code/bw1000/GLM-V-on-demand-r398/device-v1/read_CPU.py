from pathlib import Path
import json,hashlib,sys
D=Path(__file__).resolve().parent
MF='94185fd94c76edd0483a261965948053379ecedcf84c421f615fd41fa5bf5d62'
RECEIPT='3b4e14f883afade96706b26e19fa7a2322903eec7dea92c7bb1df4a456ced24c'
LIB='0c8cffd41a36a230da9f98c9ba191a77eb1c70e6a32f191f5fb8f7b12c24958e'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(path):
 q=json.loads(Path(path).read_bytes());v=q['receipt']['data'];assert q['receipt']['SHA256']==RECEIPT==sha(json.dumps(v,indent=2).encode())
 assert any(x.split('|')[0]=='910407.796'and x.split('|')[2:4]==['COMPLETED','0:0']for x in q['accounting'])
 assert v['completed']and v['HOST_TUs_compiled']==1 and v['GPU_TUs_compiled']==0 and v['caller_object_recompiled']is False and v['GPU']==0 and not v['Native_dlopen']and not v['checkpoint_payload_IO']
 assert v['source_manifest_SHA256']==MF and v['library_SHA256']==LIB and v['all_parent_exports_retained']and v['ldd_r_parent_equal']and v['old_consumer_body_byteidentical_prefix']
 assert v['actual_metric_contract']['passed']and v['actual_recording_contract']['passed']and len(v['checkpoint_keys_present'])==12
 assert sha((D/'CPU_SOURCE_MANIFEST.json').read_bytes())==MF
 for n,h in json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha((D/n).read_bytes())==h,n
 assert sha((D/'attention_component_r398.cpp').read_bytes())==v['compiled']['source_SHA256']
 return dict(passed=True,CPU='910407.796',CPU_receipt_SHA256=RECEIPT,library_SHA256=LIB,source_manifest_SHA256=MF,HOST_TUs=1,oldcaller_reused=True,GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,Model_GO=False)
if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
