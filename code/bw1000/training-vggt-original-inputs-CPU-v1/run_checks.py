from pathlib import Path
import hashlib,importlib.metadata as metadata,json,os,random,resource,signal,sys,time
W=Path(__file__).resolve().parents[1];stage=W/'src';start=time.monotonic();epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);phase='source';result=None
remaining=deadline-time.time();assert 0<remaining<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original VGGT CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,remaining)
def sha(p):
 before=p.stat();assert p.is_file() and not p.is_symlink();h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):
   if time.time()>=deadline:raise TimeoutError('hash deadline exhausted')
   h.update(raw)
 after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev)
 return h.hexdigest()
try:
 cfg_raw=(stage/'original-config.json').read_bytes();assert hashlib.sha256(cfg_raw).hexdigest()=='206dfea1f2d30487f9466af40b8d6d71e64162640b4b900254e6138b1962bbfa';cfg=json.loads(cfg_raw);assert cfg['max_steps']==20 and cfg['warmup_steps']==5 and cfg['checkpoint_steps']==[10,20,25] and cfg['training_scope']=='camera_head' and cfg['cache_frozen_aggregator_features'] is False and cfg['image_size']==224 and cfg['frames_per_step']==2 and cfg['cached_batches']==4
 shared=json.loads((stage/'SHARED_FILES.json').read_bytes());source=Path(shared['root'])
 for name,pin in shared['files'].items():assert (source/name).stat().st_size==pin['bytes'] and sha(source/name)==pin['SHA256']
 upstream=json.loads((stage/'UPSTREAM_FILES.json').read_bytes());up=Path(upstream['root']);assert len(upstream['files'])==69
 for row in upstream['files']:assert (up/row['name']).stat().st_size==row['bytes'] and sha(up/row['name'])==row['SHA256']
 annotations={}
 for name in ('manifest.json','apple_train.jgz','apple_test.jgz'):
  p=Path(cfg['annotation_root'])/name;assert p.stat().st_size<2*1024**2;digest=sha(p);side=p.with_name(p.name+'.source-sha256');expected=side.read_text().strip() if side.is_file() else None;assert expected is None or digest==expected;annotations[name]=dict(bytes=p.stat().st_size,SHA256=digest,source_companion_matches=expected==digest if expected else None)
 phase='original-model-byte-identity';stat=json.loads((stage/'MODEL_STAT.json').read_bytes());weight=Path(stat['path']);before=weight.stat();assert before.st_size==stat['bytes']==5026874952 and before.st_mtime_ns==stat['mtime_ns'] and before.st_ino==stat['inode'] and before.st_dev==stat['device'];hash_started=time.monotonic();weight_sha=sha(weight);weight_hash_wall=time.monotonic()-hash_started;side=weight.with_name(weight.name+'.source-sha256');companion=side.read_text().strip() if side.is_file() else None;assert companion is None or companion==weight_sha
 phase='native-CO3D-CPU-inputs'
 import numpy as np
 import torch
 import cv2
 assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
 def forbid(*_a,**_kw):raise RuntimeError('device initialization forbidden during VGGT source CPU input capture')
 torch.cuda._lazy_init=forbid;torch.set_num_threads(2);cv2.setNumThreads(2);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
 sys.path.insert(0,str(source/'generation/src'));sys.path.insert(0,str(up));sys.path.insert(0,str(up/'training'))
 from gencuda.backends.vggt_training import _common_config,_tensor_batch,valid_depth_batch,_vggt_batch_content_sha256
 from data.datasets.co3d import Co3dDataset
 from gencuda.hy_resume_checkpoint import tree_sha256
 dataset=Co3dDataset(_common_config(cfg['image_size']),split='train',CO3D_DIR=cfg['dataset_root'],CO3D_ANNOTATION_DIR=cfg['annotation_root'],min_num_images=cfg['minimum_sequence_frames'],len_train=1000);assert dataset.sequence_list
 batches=[];data_files={};attempted=[];candidate=0;maximum=cfg['cached_batches']*len(dataset.sequence_list)*8
 while len(batches)<cfg['cached_batches'] and candidate<maximum:
  name=sorted(dataset.sequence_list)[candidate%len(dataset.sequence_list)];count=len(dataset.data_store[name]);ids=np.asarray([(candidate*17+i*37)%count for i in range(cfg['frames_per_step'])],dtype=np.int64)
  for index in ids:
   path=Path(cfg['dataset_root'])/dataset.data_store[name][int(index)]['filepath']
   for p in (path,Path(str(path).replace('/images','/depths')+'.geometric.png'),Path(str(path).replace('/images','/depth_masks').replace('.jpg','.png'))):
    assert p.is_file() and p.stat().st_size<=32*1024**2
    if str(p) not in data_files:data_files[str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p))
  raw=dataset.get_data(seq_name=name,ids=ids,aspect_ratio=1.0);batch=_tensor_batch(torch,raw);valid=valid_depth_batch(batch['point_masks']);attempted.append(dict(candidate=candidate,valid=valid,seq_name=batch['seq_name'],ids=batch['ids']))
  if valid:batches.append(batch)
  candidate+=1
 assert len(batches)==4
 rows=[]
 for index,batch in enumerate(batches):
  fields={}
  for name,value in batch.items():
   if isinstance(value,torch.Tensor):
    assert value.device.type=='cpu' and (not value.is_floating_point() or bool(torch.isfinite(value).all()));fields[name]=dict(shape=list(value.shape),dtype=str(value.dtype),SHA256=tree_sha256(value))
  assert fields['images']['shape']==[1,2,3,224,224];rows.append(dict(batch=index,fields=fields,content_SHA256=tree_sha256(batch)))
 selected=_vggt_batch_content_sha256(batches);payload=W/'original-CPU-batches.pt';torch.save(dict(batches=batches,config_SHA256=hashlib.sha256(cfg_raw).hexdigest(),selected_batch_content_SHA256=selected),payload);assert payload.stat().st_size<32*1024**2
 with (W/'DATA_FILES.private.json').open('x') as f:json.dump(dict(files=data_files,attempts=attempted),f,indent=2)
 historical=Path('/home/whj/nvidia-baseline/results/vggt-cuda-baseline-r1/result.json');history=None
 if historical.is_file() and historical.stat().st_size<=2*1024**2:
  old=json.loads(historical.read_bytes());metrics=old.get('metrics',{});evidence=metrics.get('checkpoint_identity_evidence',{});history=dict(result_SHA256=sha(historical),global_optimizer_step=metrics.get('global_optimizer_step'),measurement_steps=metrics.get('measurement_steps'),weights_SHA256=evidence.get('frozen_model',{}).get('weights_model_pt'),selected_batch_SHA256=evidence.get('data',{}).get('selected_batch_content_sha256'));history['weights_match']=history['weights_SHA256']==weight_sha if history['weights_SHA256'] else None;history['batches_match']=history['selected_batch_SHA256']==selected if history['selected_batch_SHA256'] else None
 assert not torch.cuda.is_initialized()
 result=dict(passed=True,phase='TERMINAL_ORIGINAL_WEIGHT_AND_NATIVE4_BATCH_CPU_IDENTITIES',config_SHA256=hashlib.sha256(cfg_raw).hexdigest(),recipe=cfg,upstream_source_files=69,upstream_source_bytes=sum(x['bytes'] for x in upstream['files']),shared_modules_verified=len(shared['files']),model_weight=dict(path=str(weight),bytes=stat['bytes'],SHA256=weight_sha,source_companion_matches=companion==weight_sha if companion else None,hash_wall_seconds=weight_hash_wall),annotations=annotations,selected_batch_content_SHA256=selected,batches=rows,valid_batches=4,rejected_batches=sum(not x['valid'] for x in attempted),candidate_batches=len(attempted),sequence_count=len(dataset.sequence_list),input_source_files=len(data_files),input_source_bytes=sum(x['bytes'] for x in data_files.values()),data_files_manifest_SHA256=sha(W/'DATA_FILES.private.json'),private_CPU_batch_artifact=dict(path=str(payload),bytes=payload.stat().st_size,SHA256=sha(payload)),historical_reference_identity=history,versions=dict(**{k:metadata.version(k) for k in ('torch','numpy','torchvision','Pillow')},opencv_module=cv2.__version__),GPU_initialized=False,GPU_cards=0,model_instantiated=False,model_deserialized=False,online_feature_cache_enabled=False,framework_model_math_byte_identical_to_original=True,wall_seconds=time.monotonic()-start,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,raw_payload_transfer_bytes=0,qualification='Original singleweight byte hash,69sourcefiles,9sharedmodules and native4x2 CO3D CPU input capture only. No GPU/model forward,first-update numerics,25newupdates,recovery,formal performance or heldout effects. Raw batches and data paths remain on the original source host.')
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,message=str(error)[-400:],GPU_cards=0,raw_payload_transfer_bytes=0,wall_seconds=time.monotonic()-start)
(W/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['passed']:raise SystemExit(1)
