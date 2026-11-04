"""Only original j0 actual execution gates later shards; no answer screening."""
from pathlib import Path
import hashlib,json

def check(root,deploy,m):
 root,deploy=Path(root),Path(deploy);j=json.loads((deploy/'SHARDS.json').read_bytes())[0];plan=json.loads((deploy/j['plan']).read_bytes())['assigned'];folder=root/'results/Deep-broad-fewshot-fullpre-quality-r450'/j['run'];label=j['label']+'-'+j['run']
 paths=[root/'results'/(label+'-watch.json'),root/'results'/(label+'-client/result.json'),folder/'MODEL_READY.json',folder/'owner.json']
 if not all(p.is_file()for p in paths):return dict(passed=False,pending=True,reason='Originalj0 artifacts not all visible; no submission')
 watch,client,ready,owner=[json.loads(p.read_bytes())for p in paths]
 assert watch['shard']==0 and watch['provider_source_SHA256']==m['provider_source_SHA256']and watch['deployment_source_SHA256']==m['source_manifest_SHA256']and watch['actual_GPU']and not watch['forced_cleanup']and not watch.get('error')
 identity=ready['identity'];assert identity['actual_step']==watch['actual_GPU']and identity['source_manifest_SHA256']==m['provider_source_SHA256']and identity['provider_config_SHA256']==m['configs_SHA256']['0']and owner['identity']==identity
 assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0 and owner['loaded_key_count']==67569 and owner['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and not owner.get('error')
 if client['completed_HTTP']<2:return dict(passed=False,pending=True,reason='Originalj0 first completepair not yet visible; no submission')
 import sys;sys.path.insert(0,str(deploy.parent));from coverage import validate
 rows=[]
 for entry in plan[:2]:
  values=[r for r in owner['requests']if r.get('r450_logical_index')==entry['logical_index']and'final_metadata'in r];assert len(values)==1
  row=values[0];proof=validate(row,entry);c=[r for r in client['rows']if r['logical_index']==entry['logical_index']];assert len(c)==1 and c[0]['actual_owner_coverage']==proof and c[0]['request_body_SHA256']==entry['request_body_SHA256']
  binding=row['r450_binding_proof'];assert binding['actual_HC_groups_per_owner']==86 and binding['HC0_fixed']and set(binding['providers'].values())=={'/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'}
  rows.append(dict(logical_index=entry['logical_index'],mode=entry['mode'],proof=proof,observed_correct=c[0]['correct'],observed_cap=c[0]['cap_hit'],observed_parse_failure=c[0]['parse_failure']))
 assert [r['mode']for r in rows]==[0,1]
 return dict(passed=True,pending=False,actual_GPU=watch['actual_GPU'],original_epoch=watch['original_GPU_start_unix'],actual_READY=True,first_pair_execution=rows,quality_used_as_gate=False,no_extra_warm=True,references=[dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest())for p in paths])
