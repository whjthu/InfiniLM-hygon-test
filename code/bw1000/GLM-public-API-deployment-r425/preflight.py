"""Real final CLI/schema/fixture handoff; no device imports or Slurm actions."""
from pathlib import Path
import contextlib,hashlib,io,json,sys

def check(deploy,provider,m,*,fixture_root=None):
 deploy,provider=Path(deploy),Path(provider)
 sys.path.insert(0,str(provider))
 from public_glm_native.cli import main,validate_deployment_paths
 config=json.loads((deploy/'config.json').read_bytes());validate_deployment_paths(config)
 assert config['execute_authorized']and config['nodes']==['a02r4n12','a02r4n13']
 assert (config['Native_seconds'],config['service_seconds'])==(5100,4500)
 cli=[]
 for stage in range(2):
  output=io.StringIO()
  with contextlib.redirect_stdout(output):main(['--config',str(deploy/'config.json'),'--config-sha',m['config_SHA256'],
   '--source-sha',m['provider_source_SHA256'],'--stage',str(stage),'--run','source-only-validation',
   '--actual-step','CPU.source-only','--gpu-start-epoch','0'])
  value=json.loads(output.getvalue());assert value['source_and_schema_valid']and not value['GPU']and not value['Native_dlopen'];cli.append(value)
 card=json.loads((deploy/'FIXTURE_CARD.json').read_bytes())
 assert card['plan']==[dict(index=0,fixture_index=0,profile='glm-nov16-w2',stream=False),dict(index=1,fixture_index=1,profile='glm-nov16-w4',stream=True)]
 from workload_r349 import body_and_gold
 fixtures=[]
 for original in card['fixtures']:
  f=dict(original)
  if fixture_root is not None:f['request_external_path']=str(Path(fixture_root)/f['local_untracked_request_file'])
  raw,body,gold=body_and_gold(f);fixtures.append(dict(id=f['id'],raw_SHA256=hashlib.sha256(raw).hexdigest(),
   complete_history_messages=len(body['messages']),budget=body['max_tokens'],independent_integer_gold=True))
 inside=(deploy/'inside.sh').read_text();outer=(deploy/'resource_wave_host_alias_outer.sh').read_text()
 assert 'python3 -B -m public_glm_native "${ARGS[@]}"'in inside and '--serve'in inside
 assert 'GLM_R425_GPUSTART' in inside and '+5100-time.time()'in inside
 assert 'public-GLM-native-provider-r425'in inside and '--rocm'in outer
 assert not any(name in sys.modules for name in ['torch','_infinilm','flash_attn'])
 return dict(actual_public_CLI_two_roles=cli,fixtures=fixtures,Native_imported=False,Slurm=0,Model_READY=False)
