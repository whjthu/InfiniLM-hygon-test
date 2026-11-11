import hashlib,importlib.metadata as meta,importlib.util,json,os,runpy,signal,socket,subprocess,sys,time,unittest
from pathlib import Path
epoch=float(os.environ['RECORDER_CPU_EPOCH']);deadline=float(os.environ['RECORDER_CPU_DEADLINE']);assert deadline==epoch+150 and epoch<=time.time()<deadline
signal.setitimer(signal.ITIMER_REAL,deadline-time.time());assert socket.gethostname().split('.')[0]=='f11r4n09' and os.environ['SLURM_JOB_ID']=='1087656'
assert all(os.environ[x]=='-1' for x in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
source=Path('/shared-training');out=Path('/evidence');sys.path.insert(0,str(source));started=time.monotonic()
owner=dict(node='f11r4n09',step='1087656.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
with (out/'owned-node.json').open('x') as f:json.dump(owner,f)
result=None
try:
 runpy.run_path('/stage-source/verify_entry.py',run_name='verify_shared_entry')
 path=source/'vlm/tests/test_swift_entry.py';spec=importlib.util.spec_from_file_location('test_canonical_swift',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 observed=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(module.SwiftEntryTest));assert observed.testsRun==2 and observed.wasSuccessful()
 entry_path=source/'vlm/runtime/swift_entry.py';spec=importlib.util.spec_from_file_location('actual_shared_swift_entry',entry_path);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry);function=entry.resolve_cli();assert function.__module__=='swift.cli.main'
 # The original model bind is checked by filesystem identity only; no model load.
 root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');model=root/'results/training-minicpm-original-model-v1/MiniCPM-V-4.6'
 assert model.is_dir() and (model/'model.safetensors').stat().st_size==2600957528
 assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='72d83ac9089bebddf28131b9953908acaf0a5740b868263c8aaac7c40b97991c'
 environment=os.environ.copy()
 for name in ('NPROC_PER_NODE','RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','NNODES','NODE_RANK','MASTER_ADDR'):environment.pop(name,None)
 environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',TOKENIZERS_PARALLELISM='false')
 help_result=subprocess.run([sys.executable,'-B',str(entry_path),'sft','--help'],env=environment,capture_output=True,timeout=min(90,deadline-time.time()-15))
 (out/'canonical-help.stdout').write_bytes(help_result.stdout);(out/'canonical-help.stderr').write_bytes(help_result.stderr)
 assert help_result.returncode==0,'canonical_sft_help_returncode_'+str(help_result.returncode)
 assert all(flag in help_result.stdout for flag in (b'--model',b'--max_steps',b'--lora_dropout',b'--gradient_accumulation_steps'))
 sys.path.insert(0,str(source/'tools'))
 from hygon_vlm_recipe import RUNTIME_PROBE
 probe=subprocess.run([sys.executable,'-B','-c',RUNTIME_PROBE,str(source/'generation/src')],env=environment,capture_output=True,timeout=min(30,deadline-time.time()-5))
 (out/'runtime-probe.stdout').write_bytes(probe.stdout);(out/'runtime-probe.stderr').write_bytes(probe.stderr)
 assert probe.returncode==0
 runtime=json.loads(probe.stdout);assert runtime['accelerator']=='hip' and runtime['cuda_count']==0 and runtime['cuda_available'] is False
 result=dict(passed=True,owner=owner,unit_tests=2,canonical_entry='swift.cli.main:cli_main',canonical_source_SHA256=entry.SOURCE_SHA256,canonical_distribution=meta.version('ms-swift'),actual_sft_help_returncode=help_result.returncode,help_stdout_SHA256=hashlib.sha256(help_result.stdout).hexdigest(),help_stderr_SHA256=hashlib.sha256(help_result.stderr).hexdigest(),container_model_namespace_verified=True,actual_normal_runtime_probe_parsed=True,runtime_probe=runtime,runtime_probe_stdout_SHA256=hashlib.sha256(probe.stdout).hexdigest(),original_model_loaded=False,original_weight_values_read=False,child_GPU_visibility='-1',GPU_cards=0,wall_seconds=time.monotonic()-started,Model_GO=False)
except BaseException as error:result=dict(passed=False,owner=owner,error_type=type(error).__name__,message=str(error)[:300],wall_seconds=time.monotonic()-started,Model_GO=False)
finally:
 with (out/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
