"""Actual final source CLI/config/fixture handoff; no serving/Native/device import."""
from pathlib import Path
import contextlib,io,json,sys
def check(deploy,provider,m,*,fixture_root=None):
 deploy,provider=Path(deploy),Path(provider);sys.path.insert(0,str(provider))
 from public_deep_native.cli import main,validate_config
 config=json.loads((deploy/'config.json').read_bytes());validate_config(config)
 assert config['execute_authorized']and config['node']=='a02r4n01'and(config['Native_seconds'],config['service_seconds'])==(2580,2400)
 output=io.StringIO()
 with contextlib.redirect_stdout(output):main(['--config',str(deploy/'config.json'),'--config-sha',m['config_SHA256'],'--source-sha',m['provider_source_SHA256'],'--run','source-only-validation','--actual-step','CPU.source-only','--gpu-start-epoch','0'])
 q=json.loads(output.getvalue());assert q['source_and_schema_valid']and not q['GPU']and not q['Native_dlopen']
 card=json.loads((deploy/'FIXTURE_CARD.json').read_bytes())
 assert card['plan']==[dict(index=0,name='ordinary-HTTP',stream=False,thinking=False,budget=4096),dict(index=1,name='thinking-SSE',stream=True,thinking=True,budget=8192)]
 if fixture_root is not None:card['fixture_path']=str(Path(fixture_root)/card['local_untracked_fixture'])
 from fixture import body_and_gold
 fixtures=[]
 for arm in card['plan']:
  raw,body,gold,h=body_and_gold(card,arm);fixtures.append(dict(**arm,full_messages=len(body['messages']),message_tools_SHA256=h,independent_integer_gold=True))
 assert fixtures[0]['message_tools_SHA256']==fixtures[1]['message_tools_SHA256']
 inside=(deploy/'inside.sh').read_text();outer=(deploy/'resource_wave_host_alias_outer.sh').read_text()
 assert 'python3 -B -m public_deep_native "${ARGS[@]}"'in inside and '--serve'in inside
 assert '+2580-time.time()'in inside and '--rocm'in outer and 'source-v3'in inside
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 return dict(actual_source_CLI=q,fixture_plan=fixtures,Native_imported=False,Slurm=0,Model_READY=False)
