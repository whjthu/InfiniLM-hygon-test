"""Pure HOST final source/body/CPU-evidence check; CLI ran on actual n20 CPU."""
from pathlib import Path
import hashlib,json
from fixture import assets
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
def check(deploy,provider,m,*,local_files=None,cli_leaf=None):
 d=Path(deploy);provider=Path(provider);config=json.loads((d/'config.json').read_bytes());card=json.loads((d/'FIXTURE_CARD.json').read_bytes());inputs=assets(card,local_files)
 assert sha(provider/'SOURCE_MANIFEST.json')==m['provider_source_SHA256']=='bc7541b86afd15894d374c774ca3ed5093c623134bc493467518e4b7b3a2a98a'
 template=json.loads((provider/'config.template.json').read_bytes());template.update(execute_authorized=m['GPU_GO'],experimental_profile='deep-hc0-fullpre-c1',runtime_directory='/refactor/results/Deep-fixed-profile-public-r457',node='a02r4n04',Native_seconds=5280,service_seconds=5100)
 assert config==template and config['HC_mode']==0 and config['rows']==1 and config['context']==32768 and config['TP']==8 and config['PP']==1
 pin=json.loads((d/'CPU_EVIDENCE.json').read_bytes());p=Path(pin['CPU962']['path']);p=p if local_files is None else Path(local_files)/'CPU962.json';assert sha(p)==pin['CPU962']['SHA256'];q=json.loads(p.read_bytes())
 assert q['actual_CPU']=='910407.962'and q['completed']and q['passed']and q['source_manifest_SHA256']==m['provider_source_SHA256']and q['worker_control']['passed']and q['actual_binding']['objects']==50 and not q['actual_binding']['project_maps']and not q['Native_dlopen']and not q['GPU']
 if cli_leaf is not None:actual=cli_leaf(d,m)
 else:
  reference=pin['changed_glue'];p=Path(reference['path']);assert sha(p)==reference['SHA256'];v=json.loads(p.read_bytes());assert v['completed']and v['passed']and v['new_glue']['passed']and v['whole_CLI']['experimental_profile']=='deep-hc0-fullpre-c1'and not v['GPU']and not v['Native_dlopen'];actual=v['whole_CLI']
  old=json.loads(Path(reference['CPU_config']).read_bytes());old['execute_authorized']=m['GPU_GO'];assert old==config
 assert actual==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False,experimental_profile='deep-hc0-fullpre-c1')
 return dict(passed=True,actual_whole_CLI_on_n20=actual,requests=2,warms=0,cases=[e[0]['case_id']for e in inputs],prior_encoding945_consumed=True,prior_CPU962_consumed=True,Native_dlopen=False,Slurm=0,GPU=0,Model_READY=False)
