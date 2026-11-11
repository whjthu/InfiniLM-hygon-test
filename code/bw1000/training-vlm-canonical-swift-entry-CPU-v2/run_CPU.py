ENTRY_SHA='8f72975bdfe86477b95da8d51ed34f755aff9012f05f33072ae128bc6c6829cb'
import hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
epoch=float(os.environ['RECORDER_CPU_EPOCH']);deadline=float(os.environ['RECORDER_CPU_DEADLINE'])
assert deadline==epoch+150 and epoch<=time.time()<deadline
signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
assert socket.gethostname().split('.')[0]=='f11r4n09' and os.environ['SLURM_JOB_ID']=='1087656'
assert all(os.environ[n]=='-1' for n in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
source=Path('/shared-training');out=Path('/evidence');started=time.monotonic();stage='canonical_metadata'
owner=dict(node='f11r4n09',step='1087656.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
with (out/'owned-node.json').open('x') as f:json.dump(owner,f)
try:
 entry_path=source/'vlm/runtime/swift_entry.py'
 assert hashlib.sha256(entry_path.read_bytes()).hexdigest()==ENTRY_SHA
 spec=importlib.util.spec_from_file_location('actual_shared_swift_entry',entry_path);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
 function=entry.resolve_cli();assert function.__module__=='swift.cli.main'
 sft=Path('/swift-runtime/site/swift/cli/sft.py');assert sft.stat().st_size==583 and hashlib.sha256(sft.read_bytes()).hexdigest()=='97162d10eb5b82cdfadf1ec30a0eac50cc0bc9646140b8342806df9e8e6e07f0'
 environment=os.environ.copy()
 for name in ('NPROC_PER_NODE','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','NNODES','NODE_RANK','MASTER_ADDR'):environment.pop(name,None)
 environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',TOKENIZERS_PARALLELISM='false')
 stage='actual_sft_first_parser_help'
 help_result=subprocess.run([sys.executable,'-B',str(entry_path),'sft','--help'],env=environment,capture_output=True,timeout=min(90,deadline-time.time()-15))
 (out/'canonical-help.stdout').write_bytes(help_result.stdout);(out/'canonical-help.stderr').write_bytes(help_result.stderr)
 assert help_result.returncode==0,'canonical_sft_help_returncode_'+str(help_result.returncode)
 assert b'/swift-runtime/site/swift/cli/sft.py --help' in help_result.stdout and b'--tuner_backend' in help_result.stdout,'canonical first parser help differs from pinned source'
 stage='actual_normal_runtime_probe'
 sys.path.insert(0,str(source/'tools'))
 from hygon_vlm_recipe import RUNTIME_PROBE
 probe=subprocess.run([sys.executable,'-B','-c',RUNTIME_PROBE,str(source/'generation/src')],env=environment,capture_output=True,timeout=min(30,deadline-time.time()-5))
 (out/'runtime-probe.stdout').write_bytes(probe.stdout);(out/'runtime-probe.stderr').write_bytes(probe.stderr)
 assert probe.returncode==0,'runtime_probe_returncode_'+str(probe.returncode)
 runtime=json.loads(probe.stdout);assert runtime['accelerator']=='hip' and runtime['cuda_count']==0 and runtime['cuda_available'] is False,'masked HIP runtime probe differs'
 result=dict(passed=True,owner=owner,framework_commit='0345a8c9ba2966b0fb7301edfbbe86d9cdd608b4',canonical_source_SHA256=entry.SOURCE_SHA256,sft_source_SHA256='97162d10eb5b82cdfadf1ec30a0eac50cc0bc9646140b8342806df9e8e6e07f0',actual_sft_help_returncode=help_result.returncode,help_stdout_SHA256=hashlib.sha256(help_result.stdout).hexdigest(),actual_normal_runtime_probe_parsed=True,runtime_probe=runtime,runtime_probe_stdout_SHA256=hashlib.sha256(probe.stdout).hexdigest(),help_scope='Pinned first tuner parser only; not complete SFT argument parsing or model execution',full_training_arguments_qualified=False,original_model_loaded=False,original_weight_values_read=False,GPU_cards=0,wall_seconds=time.monotonic()-started,Model_GO=False)
except BaseException as error:
 (out/'failure-traceback.private.txt').write_text(traceback.format_exc())
 result=dict(passed=False,owner=owner,stage=stage,error_type=type(error).__name__,message=str(error)[:300],wall_seconds=time.monotonic()-started,Model_GO=False)
finally:
 with (out/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
