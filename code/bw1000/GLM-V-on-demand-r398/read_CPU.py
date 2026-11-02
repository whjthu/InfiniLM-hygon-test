"""stdlib-only independent reader of the preserved actual CPU observation."""
from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent
MF='6b085d827ecaf8c2a66d7e808a25f50926f078f70e06b61001e70a64ca53f2bc'
RECEIPT='399f26b7668f3fa21a22ab81a041cb5b1b1835aad50a3d18fa6a6ae8196f311b'
LIB='431c5b85730549861202ed9aff1b479fa3d23727a1af575777b2c6fc915f88bc'
sha=lambda x:hashlib.sha256(x).hexdigest()
def read(path):
 q=json.loads(Path(path).read_bytes());r=q['receipt'];v=r['data']
 assert r['SHA256']==RECEIPT==sha(json.dumps(v,indent=2).encode())
 assert any(row.split('|')[0]=='910407.789'and row.split('|')[2:4]==['COMPLETED','0:0']for row in q['accounting'])
 assert v['completed']and v['producer']=='910407.789'and v['CPU_only']and v['HOST_TUs_compiled']==2 and v['GPU_TUs_compiled']==0
 assert not v['GPU_initialized']and not v['Native_dlopen']and not v['checkpoint_IO']and not v['GPU_GO']and not v['Model_GO']
 assert v['library_SHA256']==LIB and v['source_manifest_SHA256']==MF
 assert v['all_parent_exports_retained']and v['ldd_r_parent_equal']and len(v['new_exports'])==10
 assert v['actual_selection_contract']==dict(passed=True,actual_parent_selection_map=True,decode_U=2048,prefill512_U=32768,invalid_boundary_rejected=True,GPU=0)
 c=v['actual_consumer_recording'];assert c['passed']and c['actual_consumer_cases']==12 and c['seed_calls']==774 and c['query_calls']==12 and c['partialfailure_saved']and c['normal_and_error_close']and not c['Native_executed']
 assert sha((D/'CPU_SOURCE_MANIFEST.json').read_bytes())==MF
 for n,h in json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha((D/n).read_bytes())==h,n
 assert len(v['compiled'])==2
 for x in v['compiled']:
  p=D/Path(x['source']).name;assert sha(p.read_bytes())==x['source_SHA256']
 return dict(passed=True,actual_CPU='910407.789',CPU_receipt_SHA256=RECEIPT,library_SHA256=LIB,source_manifest_SHA256=MF,HOST_TUs=2,selection_union=[2048,32768],recording_queries=12,GPU=0,Native_dlopen=False,weights=False,device_numeric_qualified=False,allocator_peak_qualified=False,Model_GO=False)
if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
