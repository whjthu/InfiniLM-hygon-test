from pathlib import Path
import copy,hashlib,importlib.metadata as metadata,io,json,os,resource,sys,time,unittest
W=Path(__file__).resolve().parents[1];stage=W/'src';F=W/'framework';sys.path.insert(0,str(F/'generation/src'));start=time.monotonic();phase='source';result=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 reference=json.loads((stage/'ORIGINAL_REFERENCE.json').read_bytes());assert sha(Path(reference['CPU_result_path']))==reference['CPU_result_SHA256'];assert sha(Path(reference['historical_result_path']))==reference['historical_result_SHA256'];original=json.loads(Path(reference['historical_result_path']).read_bytes());evidence=copy.deepcopy(original['metrics']['checkpoint_identity_evidence']);cfg_raw=(stage/'original-config.json').read_bytes();assert hashlib.sha256(cfg_raw).hexdigest()=='206dfea1f2d30487f9466af40b8d6d71e64162640b4b900254e6138b1962bbfa';cfg=json.loads(cfg_raw)
 upstream=json.loads((stage/'UPSTREAM_FILES.json').read_bytes());up=Path(upstream['root'])
 for pin in upstream['files']:assert (up/pin['name']).stat().st_size==pin['bytes'] and sha(up/pin['name'])==pin['SHA256']
 import torch
 assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0;torch.set_num_threads(2)
 def forbid(*_a,**_kw):raise RuntimeError('GPU context forbidden in native VGGT CPU qualification')
 torch.cuda._lazy_init=forbid
 phase='real-AdamW-capture-tests';suite=unittest.defaultTestLoader.discover(str(F/'generation/tests'),pattern='test_vggt_first_update_capture.py');log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.private.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==7
 phase='full-native-meta-structure';sys.path.insert(0,str(up));sys.path.insert(0,str(up/'training'))
 from accelerate import init_empty_weights
 from vggt.models.vggt import VGGT
 from gencuda.backends import vggt_training as native
 from gencuda.vggt_first_update import parameter_manifest,validate_structure
 from gencuda.hy_resume_checkpoint import json_sha256
 with init_empty_weights(include_buffers=False):model=VGGT(enable_camera=True,enable_point=False,enable_depth=False,enable_track=False)
 module,key=native._vggt_bind_trainable_scope(torch,model,'camera_head');assert key=='camera_head' and all(p.is_meta for p in model.parameters())
 named=[(n,p) for n,p in module.named_parameters() if p.requires_grad];assert named and all(str(p.dtype)=='torch.float32' for _,p in named);optimizer=torch.optim.AdamW([p for _,p in named],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);native._vggt_assert_optimizer_binding(optimizer,[p for _,p in named]);binding=dict(case='T12.camera-head',platform='cuda',config_canonical_SHA256=json_sha256(cfg),original_config_SHA256=hashlib.sha256(cfg_raw).hexdigest(),trainable_manifest=parameter_manifest(named),model_weights_SHA256=reference['model_weights_SHA256'],selected_batch_SHA256=reference['selected_batch_SHA256'],initial_values_qualified=False)
 verified=validate_structure(model,cfg,binding);assert [id(p) for _,p in verified]==[id(p) for _,p in named]
 evidence['workload']['backend_source']=sha(F/'generation/src/gencuda/backends/vggt_training.py');evidence['sources']=native._vggt_source_evidence(model,optimizer);evidence['optimizer']=native._vggt_optimizer_evidence(optimizer,module);assert evidence['frozen_model']['weights_model_pt']==reference['model_weights_SHA256'] and evidence['data']['selected_batch_content_sha256']==reference['selected_batch_SHA256'];binding['native_identity']={key:json_sha256(value) for key,value in evidence.items()}
 rejected=[]
 for name,bad in [('training_scope','camera_backbone'),('cache_frozen_aggregator_features',True),('max_steps',19),('warmup_steps',4),('checkpoint_steps',[10,25]),('image_size',112),('frames_per_step',1),('learning_rate',.01)]:
  changed=copy.deepcopy(cfg);changed[name]=bad
  try:validate_structure(model,changed,binding)
  except ValueError:rejected.append(name)
  else:raise AssertionError('bad original contract accepted')
 for name,change,restore in [('aggregator_train_mode',lambda:model.aggregator.train(),lambda:model.aggregator.eval()),('camera_head_eval_mode',lambda:model.camera_head.eval(),lambda:model.camera_head.train())]:
  change()
  try:validate_structure(model,cfg,binding)
  except ValueError:rejected.append(name)
  else:raise AssertionError('bad native mode accepted')
  restore()
 p=next(model.aggregator.parameters());p.requires_grad_(True)
 try:validate_structure(model,cfg,binding)
 except ValueError:rejected.append('aggregator_trainable')
 else:raise AssertionError('trainable aggregator accepted')
 p.requires_grad_(False);manifest=binding['trainable_manifest'];binding['trainable_manifest']=manifest[:-1]
 try:validate_structure(model,cfg,binding)
 except ValueError:rejected.append('missing_parameter')
 else:raise AssertionError('missing parameter accepted')
 binding['trainable_manifest']=manifest;validate_structure(model,cfg,binding);assert len(rejected)==12
 params=sum(p.numel() for _,p in named);largest=max(p.numel()*p.element_size() for _,p in named);assert largest*2+65536<256*1024**2 and params*4*6+64*1024**2<12*1024**3
 (W/'NATIVE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n');(W/'EXPECTED_NATIVE_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n');assert not torch.cuda.is_initialized()
 result=dict(passed=True,phase='TERMINAL_NATIVE_FIRST_UPDATE_RECORDER_AND_META_CONTRACT',shared_tests=7,shared_test_failures=len(tests.failures),shared_test_errors=len(tests.errors),native_contract_valid=1,native_contract_rejected=rejected,actual_model_class=type(model).__module__+'.'+type(model).__qualname__,native_trainable_tensors=len(named),native_trainable_parameters=params,trainable_dtypes=sorted(set(str(p.dtype) for _,p in named)),aggregator_parameters=sum(p.numel() for p in model.aggregator.parameters()),frozen_aggregator=True,all_model_parameters_meta=True,source_model_weight_values_read=False,production_model_forward_backward=False,largest_parameter_bytes=largest,largest_optimizer_payload_bound_bytes=largest*2+65536,planned_first_update_value_bytes=params*4*6,artifact_limit_bytes=256*1024**2,total_limit_bytes=12*1024**3,expected_artifact_files=len(named)*5+6,binding_SHA256=sha(W/'NATIVE_BINDING.json'),expected_evidence_SHA256=sha(W/'EXPECTED_NATIVE_EVIDENCE.json'),versions={k:metadata.version(k) for k in ('torch','numpy','accelerate','huggingface-hub')},GPU_initialized=False,GPU_cards=0,framework_commit='5d12acb6fc4693243acca44cd27ede1b24567889',framework_files=321,raw_payload_transfer_bytes=0,wall_seconds=time.monotonic()-start,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,qualification='Real tiny CPU Linear/AdamW first-update recorder and actual full VGGT meta camera-head/frozen-aggregator structural contract. Expected frozen-weight/input identity is reused from independently matched original artifacts; no production parameter values,newweightread,ModelGPU,numerical,recovery,performance or effects qualification.')
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,message=str(error)[-500:],GPU_cards=0,raw_payload_transfer_bytes=0,wall_seconds=time.monotonic()-start)
(W/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['passed']:raise SystemExit(1)
