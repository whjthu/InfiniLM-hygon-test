"""Run the unchanged shared native Qwen template on all original COCO64 rows."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,importlib.metadata as metadata,io,json,os,resource,signal,sys,time,unittest
W=Path(__file__).resolve().parents[1];stage=W/'src';framework=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE']);start=time.monotonic()
assert 0<deadline-time.time()<=260
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Qwen input CPU deadline exhausted')))
signal.setitimer(signal.ITIMER_REAL,deadline-time.time());sys.path.insert(0,str(framework))
weight_open_attempts=[]
def forbid_weights(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):
  weight_open_attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('weight opening forbidden in CPU input scope')
sys.addaudithook(forbid_weights)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2)
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
dump('CPU-owner.json',owner());phase='source'
try:
 fm=json.loads((stage/'FRAMEWORK_FILES.json').read_bytes())
 for name,pin in fm['files'].items():
  p=framework/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 assets=json.loads((stage/'ORIGINAL_INPUT_FILES.json').read_bytes())
 for pin in assets['files']:
  p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 observed=json.loads((stage/'PROCESSOR_METADATA.json').read_bytes());pin=observed['template_source'];p=Path(pin['path'])
 assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 phase='shared-stdlib-tests';log=io.StringIO()
 suite=unittest.defaultTestLoader.loadTestsFromName('vlm.tests.test_template_inputs');unit=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 (W/'tests.txt').write_text(log.getvalue());assert unit.wasSuccessful() and unit.testsRun==5
 from tools.hygon_vlm_recipe import build_launch
 from vlm.runtime.template_inputs import capture_native_inputs,settings_from_launch
 model=Path(observed['original_model_root']);dataset=Path(observed['dataset']['path'])
 launch=build_launch('T01.sft',(stage/'original-launch.json').read_bytes(),PurePosixPath('/unused/swift'),PurePosixPath(str(model.parent)),PurePosixPath(str(dataset)),PurePosixPath('/unused/plugins'),PurePosixPath('/unused/output'),29501)
 settings=settings_from_launch(launch);assert settings['model_type']==settings['template_type']=='qwen3_vl' and settings['max_length']==4096
 phase='native-Qwen-64-row-template'
 capture=capture_native_inputs(settings,model,dataset,W/'encoded',observed['files'],observed['dataset']['SHA256'],epoch=epoch,deadline=deadline)
 rows=json.loads((W/'encoded/input-identities.json').read_bytes())
 assert len(rows)==64 and all(r['row_ordinal']==i+1 for i,r in enumerate(rows))
 assert capture['passed'] and capture['rows']==64 and not capture['model_loaded'] and not capture['CUDA_initialized'] and not capture['raw_encoded_payload_saved']
 lengths=[row['sequence_length'] for row in rows];config=json.loads((model/'config.json').read_bytes());vocab=config['text_config']['vocab_size']
 assert len(lengths)==64 and max(lengths)<=4096
 row_bounds=[]
 for row in rows:
  for field in row['fields'].values():assert field['finite'] and len(field['SHA256'])==64
  row_bounds.append(dict(row_ordinal=row['row_ordinal'],sequence_length=row['sequence_length'],
   hypothetical_full_F32_logits_bytes=row['sequence_length']*vocab*4,
   encoded_tensor_elements=sum(field['elements'] for field in row['fields'].values())))
 dump('pre-template-capture-size.json',dict(rows=row_bounds,vocab=vocab,max_length=max(lengths),
  maximum_full_F32_logits_bytes=max(lengths)*vocab*4,existing_per_artifact_limit=2*1024**3-1,
  all_pre_template_lengths_fit_F32_output_limit=max(lengths)*vocab*4<=2*1024**3-1,
  final_native_collator_observed=False,actual_distributed_first_batch_observed=False,
  complete_serialized_capture_budget_qualified=False,qualification='Actual complete pre-model native template lengths; output-size arithmetic only, not actual logits or final forward input sizes.'))
 actual_sources={}
 for module in list(sys.modules.values()):
  name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
  if path and (name.startswith(('swift.template.templates.qwen','transformers.models.qwen3_vl.','transformers.models.qwen2_vl.')) or name in ('vlm.runtime.template_inputs','vlm.runtime.first_update_audit','tools.hygon_vlm_recipe')):
   p=Path(path).resolve()
   if p.suffix=='.py' and p.is_file():actual_sources[str(p)]=dict(bytes=p.stat().st_size,SHA256=sha(p))
 result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
  framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),shared_tests=5,original_input_files_verified=len(assets['files']),
  processor_files_verified=len(observed['files']),processor_bytes=sum(x['bytes'] for x in observed['files'].values()),capture=capture,
  identities_SHA256=sha(W/'encoded/input-identities.json'),size_analysis_SHA256=sha(W/'pre-template-capture-size.json'),
  sequence_length_min=min(lengths),sequence_length_max=max(lengths),encoded_fields=sum(len(r['fields']) for r in rows),
  actual_sources=actual_sources,weight_file_open_attempts=weight_open_attempts,model_weight_values_read=False,
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,
  model_loaded=False,GPU_cards=0,GPU_GO=False,Model_GO=False,first_update_numerics_qualified=False,capture_size_qualified=False,
  qualification='All 64 original Qwen CPU native-template identities before model post-encode or final collator; no actual distributed first batch,model forward,update,numerics,recovery or performance.')
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
 result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,GPU_cards=0,GPU_GO=False,Model_GO=False,wall_seconds=time.monotonic()-start)
dump('CPU-result.json',result)
print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
