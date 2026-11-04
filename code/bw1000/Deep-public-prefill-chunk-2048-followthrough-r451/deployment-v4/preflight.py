"""Actual whole source CLI in the existing CPU-only container, before any GPU."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def whole_cli(deploy,m,publication):
 d=Path(deploy);manifest=sha(d/'SOURCE_MANIFEST.json')
 command=[sys.executable,'-B',str(d/'metadata_probe.py'),'--source-sha',manifest,'--publication',publication]
 z=subprocess.run(command,capture_output=True,text=True,timeout=130)
 assert z.returncode==0,(z.stdout,z.stderr)
 summary=json.loads(z.stdout);path=Path(summary['output']);q=json.loads(path.read_bytes())
 assert path.parent.parent==R/'results/Deep451-deployment-v4-metadata'and str(path.parent.name).endswith('-pid'+str(q['HOST_PID']))
 assert q['source_manifest_SHA256']==manifest and q['publication']==publication and q['completed']
 assert q['returncode']==0 and q['actual_schema_valid']and q['actual_original_CLI_returned']and q['owned_PGID_still_live']is False
 assert not q['signals']and not q.get('timed_out_original90s')and not q.get('output_limit_exceeded_observed_bytes')
 expected=dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 assert q['actual_CLI_result']==expected
 for name in ['stdout','stderr']:
  v=q[name];p=Path(v['path']);assert not v['truncated']and v['hash_scope']=='complete_file'and p.stat().st_size==v['bytes']and sha(p)==v['SHA256']
 assert q['stdout']['bytes']==94 and json.loads(Path(q['stdout']['path']).read_bytes())==expected
 return dict(actual_whole_CLI=expected,container_rocm=False,serving=False,Native_imported=False,argv=q['original_argv'],marked_argv=q['marked_argv'],metadata_result=dict(path=str(path),bytes=path.stat().st_size,SHA256=sha(path)),metadata_PID=q['HOST_PID'],child_PID=q['child_PID'],child_PGID=q['owned_PGID'],child_wall_s=q['child_wall_elapsed_s'],stdout=q['stdout'],stderr=q['stderr'],old_timeout_root_cause='UNKNOWN')
def check(deploy,provider,m,*,fixture_root=None,evidence_root=None,cli_leaf=None,publication=None):
 deploy,provider=Path(deploy),Path(provider)
 from evidence import check as verify_cpu
 cpu=verify_cpu(deploy,provider,local_files=evidence_root)
 config=json.loads((deploy/'config.json').read_bytes());assert config['execute_authorized']and config['node']=='a02r4n12'and(config['Native_seconds'],config['service_seconds'])==(2580,2400)
 assert config['default_text_delivery']=='P1_when_stream_true_else_P0'and config['text_delivery_experiment']is True
 card=json.loads((deploy/'FIXTURE_CARD.json').read_bytes());assert [v['delivery']for v in card['plan']]==['P1']*6 and [v['mode']for v in card['plan']]==[0]*6 and [v['chunk']for v in card['plan']]==config['r451_chunks']==[1024,2048,1024,2048,2048,1024] and config['r451_fullpre_fixed']==0 and [v['warmup']for v in card['plan']]==[True,True,False,False,False,False]
 if fixture_root is not None:card['fixture_path']=str(Path(fixture_root)/card['local_untracked_fixture'])
 from fixture import body_and_gold
 inputs=[]
 for arm in card['plan']:
  raw,body,gold,h=body_and_gold(card,arm);assert body['max_tokens']==128 and body['stream']and body['chat_template_kwargs']=={'enable_thinking':False};inputs.append(h)
 assert len(set(inputs))==1 and card['original_prompt_tokens']==8489 and config['r451_output_budget']==128
 expected=json.loads((provider/'config.template.json').read_bytes());expected['execute_authorized']=True;assert config==expected
 inside=(deploy/'inside.sh').read_bytes();outer=(deploy/'resource_wave_host_alias_outer.sh').read_bytes();assert b'\r'not in inside+outer and b'\0'not in inside+outer
 assert b'python3 -B -m public_deep_model "${ARGS[@]}"'in inside and b'--serve'in inside and b'+2580-time.time()'in inside and b'--rocm'in outer
 actual=whole_cli(deploy,m,publication)if cli_leaf is None else cli_leaf(deploy,m)
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 return dict(actual_CLI=actual,CPU_scopes=cpu,requests=6,warmups=2,formal=4,Native_imported=False,Slurm=0,Model_READY=False)
