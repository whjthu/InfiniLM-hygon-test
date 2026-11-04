"""Stdlib-only exact original978 receipt/log/accounting audit, not a device test."""
from pathlib import Path
import argparse,base64,hashlib,json
def audit(path):
 o=json.loads(Path(path).read_bytes());d={}
 for n,v in o['files'].items():
  b=base64.b64decode(v['base64']);assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['SHA256'];d[n]=b
 raw=d['candidates/Deep465-public-fixed-cohort-CPU-v1/result.json'];q=json.loads(raw)
 assert o['accounting']['rc']==0 and o['accounting']['rows']==['910407.978|Deep465-public-fixed-cohort-CPU-v1|COMPLETED|0:0|4|a14r2n20']
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.978'and q['source_manifest_SHA256']=='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec'
 assert not any(q[k]for k in['GPU','Native_dlopen','SDK','weights','new_compile','reencode','Model_READY'])
 c=q['changed_local_production_consumers'];assert c['passed']and not c['actual_Native']and c['client']['HTTP']==64 and c['client']['all64_wrong_not_early_stop']and c['client']['normal_join']
 assert len(c['boundaries'])==5 and all(x['passed']for x in c['boundaries'])
 b=q['typed_binding'];assert b['rows']==4 and b['objects']==50 and b['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and not b['Native_dlopen']
 assert q['actual_pilot_metadata']==dict(cases=32,HTTP=64,output_SHA256='983dbc515213a5f96d4423a4ddc6ac4401ec42568062a64dadaa6813bd866834',new_encode=False)
 assert q['actual_nonserve_CLI']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False,experimental_profile='deep-hc0-fixed-cohort-c4')
 assert len(q['commands'])==4 and all(z['rc']==0 for z in q['commands'])
 return dict(passed=True,actual_CPU=q['actual_CPU'],receipt_SHA256=hashlib.sha256(raw).hexdigest(),files=len(d),decoded_bytes=sum(map(len,d.values())),source_manifest_SHA256=q['source_manifest_SHA256'],CPU_facts=q,GPU=False,Model_READY=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('observation');p.add_argument('--output',required=True);a=p.parse_args();v=audit(a.observation);Path(a.output).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v[k]for k in['passed','actual_CPU','receipt_SHA256','files','GPU','Model_READY']}))
