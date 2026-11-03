from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('observation');a=p.parse_args();q=json.loads(Path(a.observation).read_bytes());r=q['receipt'];raw=r['rawtext'].encode();assert len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['SHA256']=='99b3723b72c04c392dae061ab8ebe89754e97dd12e91062539c0021fc9155a48';v=json.loads(raw)
assert v['passed']and v['completed']and v['CPU_only']and v['GPU']==0 and not v['Native_dlopen']and not v['checkpoint_weight_payload']and not v['Torch_imported'];assert v['actual_CPU']=='910407.869'and v['source_manifest_SHA256']=='c8556c9d4f291a6eb3429af57afb491e16bad03e99dde5ee3005f8b38580acb2'
assert v['HOST_production_TUs']==v['GPU_TUs']==v['pure_geometry_contract_TUs']==1 and len(v['compiled'])==2 and all(c['rc']==0 for c in v['commands']);assert v['geometry_valid']==4 and v['geometry_negative']==9 and len(v['private_CABI'])==7
assert v['library_SHA256']=='a431011ac89f009dca15b029191007a29eae13b126ac56ff94e463b2c5623489';assert '910407.869|'in q['accounting']['stdout']and '|COMPLETED|0:0|'in q['accounting']['stdout']
print(json.dumps(dict(passed=True,actual_CPU=v['actual_CPU'],receipt_SHA256=r['SHA256'],library_SHA256=v['library_SHA256'],new_device_qualification=False)))
