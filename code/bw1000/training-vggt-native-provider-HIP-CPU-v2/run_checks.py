"""Actual target HIP-container CPU/provider/meta qualification, without weights or data."""
from pathlib import Path
import ast,copy,hashlib,importlib,importlib.metadata as metadata,io,json,os,resource,signal,sys,time,unittest
W=Path('/evidence');S=Path('/input-stage');F=Path('/input-framework');U=Path('/vggt-source');epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);start=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==220
def expired(*_):raise TimeoutError('original target CPU epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n19' and os.environ['SLURM_CPUS_PER_TASK']=='4'
phase='source';result=None;versions={}
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==323 and fm['framework_commit']=='3b1d6909efe4b9d25d4a1562068e2006298f82ed'
 for name,pin in fm['files'].items():
  p=F/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
 up=json.loads((S/'UPSTREAM_FILES.json').read_bytes());assert len(up['files'])==69
 for pin in up['files']:
  p=U/pin['name'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 deps=json.loads((S/'DEPENDENCIES.json').read_bytes());versions={k:metadata.version(k) for k in deps['versions']};assert versions==deps['versions']
 import torch
 assert torch.version.hip==deps['HIP'] and not torch.cuda.is_initialized();torch.set_num_threads(2)
 def forbid(*_a,**_kw):raise RuntimeError('device context forbidden during VGGT target CPU qualification')
 torch.cuda._lazy_init=forbid
 sys.path[:0]=[str(F/'generation/src'),str(U),str(U/'training')]
 phase='portable-AST-regressions';reuse=json.loads((S/'REUSED_CPU172.json').read_bytes());previous=Path('/previous-vggt');assert sha(previous/'outcome.json')==reuse['outcome_SHA256'] and sha(previous/'evidence/CPU-result.json')==reuse['CPU_result_SHA256'] and sha(previous/'evidence/tests.private.txt')==reuse['tests_log_SHA256']
 previous_result=json.loads((previous/'evidence/CPU-result.json').read_bytes());assert previous_result['passed'] is False and previous_result['phase']=='target-recorder-CPU-tests' and previous_result['versions']==versions and previous_result['owner']['step']==reuse['original_step']
 text=(previous/'evidence/tests.private.txt').read_text();assert 'FAILED (failures=1)' in text and reuse['original_failed_test'] in text
 from gencuda.ast_identity import portable_ast_sha256
 def methods(path):
  tree=ast.parse(path.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VGGTFirstUpdateTest');return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
 old_methods=methods(previous/'framework/generation/tests/test_vggt_first_update_capture.py');new_methods=methods(F/'generation/tests/test_vggt_first_update_capture.py')
 for name in reuse['unchanged_passed_methods']:
  assert name+' (test_vggt_first_update_capture.VGGTFirstUpdateTest) ... ok' in text and portable_ast_sha256(old_methods[name])==portable_ast_sha256(new_methods[name])
 for helper in ['fixture','step']:
  assert portable_ast_sha256(old_methods[helper])==portable_ast_sha256(new_methods[helper])
 suite=unittest.defaultTestLoader.discover(str(F/'generation/tests'),pattern='test_ast_identity.py');suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_vggt_first_update_capture.VGGTFirstUpdateTest.test_stripped_backend_matches_frozen_native_math_ast'));log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.private.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==3
 phase='full-native-model-meta';from accelerate import init_empty_weights
 from vggt.models.vggt import VGGT
 from gencuda.backends import vggt_training as native
 from gencuda.vggt_first_update import parameter_manifest,validate_structure
 from gencuda.hy_resume_checkpoint import json_sha256
 with init_empty_weights(include_buffers=False):model=VGGT(enable_camera=True,enable_point=False,enable_depth=False,enable_track=False)
 head,key=native._vggt_bind_trainable_scope(torch,model,'camera_head');assert key=='camera_head' and all(p.is_meta for p in model.parameters())
 original=json.loads((S/'ORIGINAL_A100_STRUCTURE.json').read_bytes());cfg=json.loads((S/'original-config.json').read_bytes());named=[(n,p) for n,p in head.named_parameters() if p.requires_grad];manifest=parameter_manifest(named);assert manifest==original['trainable_manifest'] and len(manifest)==69 and sum(p.numel() for _,p in named)==216174610
 optimizer=torch.optim.AdamW([p for _,p in named],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);native._vggt_assert_optimizer_binding(optimizer,[p for _,p in named]);binding=dict(case='T12.camera-head',platform='hip',config_canonical_SHA256=json_sha256(cfg),trainable_manifest=manifest);assert binding['config_canonical_SHA256']==original['config_canonical_SHA256'];validate_structure(model,cfg,binding)
 assert not optimizer.state and model.camera_head.training and not model.aggregator.training and not any(p.requires_grad for p in model.aggregator.parameters())
 phase='original-CO3D-module-import';from data.datasets.co3d import Co3dDataset
 phase='actual-target-provider-evidence';evidence=dict(sources=native._vggt_source_evidence(model,optimizer),optimizer=native._vggt_optimizer_evidence(optimizer,head));provider_bytes=(json.dumps(evidence,indent=2)+'\n').encode();(W/'TARGET_PROVIDER_EVIDENCE.json').write_bytes(provider_bytes)
 assert not torch.cuda.is_initialized();result=dict(passed=True,phase='TERMINAL_FULL323_VGGT_NATIVE_HIP_PROVIDER_META_AND_PORTABLE_AST_CPU_PASS',owner=owner,versions=versions,HIP=torch.version.hip,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_files=323,upstream_Python_files=69,upstream_Python_bytes=sum(p['bytes'] for p in up['files']),changed_CPU_tests=3,reused_unchanged_recorder_CPU_tests=6,total_case_CPU_tests=9,all_model_parameters_meta=True,original_A100_trainable_manifest_exact=True,trainable_tensors=69,trainable_parameters=216174610,trainable_dtype='torch.float32',frozen_aggregator=True,online_aggregator_recipe_preserved=True,actual_CO3D_module_import=True,target_provider_evidence_SHA256=hashlib.sha256(provider_bytes).hexdigest(),production_initial_values_qualified=False,model_weight_bytes_read=0,raw_input_bytes_read=0,CUDA_initialized=False,GPU_cards=0,target_weight_data_path_mapping_qualified=False,cross_platform_numeric_alignment_qualified=False,fresh_process_HIP_recovery_qualified=False,formal_performance_qualified=False,heldout_effects_qualified=False,Model_GO=False)
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,message=str(error)[-700:],versions=versions,owner=owner,GPU_cards=0,model_weight_bytes_read=0,raw_input_bytes_read=0,Model_GO=False)
result.update(wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,epoch=epoch,deadline=deadline)
(W/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
