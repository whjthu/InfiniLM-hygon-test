"""Actual one-owner/six-cohort device entry; no HTTP or extra warm requests."""
import argparse,hashlib,json,os,socket,sys,time,traceback
from pathlib import Path
from support_gpu import D,PACKAGE,R,RUN,NODE,sha,verify,publish

def writable_results(root=None):
 return Path('/refactor/results')if root is None else Path(root)

def make_output_directory(run,root=None):
 output=writable_results(root)/'Deep458-fixed-cohort-private'/run;output.mkdir(parents=True,exist_ok=False);return output

def execute(cfg,m):
 output=make_output_directory(RUN)
 report=dict(completed=False,passed=False,phase='source_verified',PID=os.getpid(),node=socket.gethostname().split('.')[0],
   actual_step=cfg['actual_step'],gpu_start_epoch=cfg['gpu_start_epoch'],Native_seconds=5280,GPU_seconds=5400,
   source_manifest_SHA256=m['entry_manifest_SHA256'],deployment_manifest_SHA256=cfg['deployment_manifest_SHA256'],
   Native_SHA256=cfg['Native_SHA256'],profile=cfg['profile'],HTTP=0,logical_requests=0,completed_cohorts=0,
   infrastructure_error=None,normal_Native_shutdown=False,Model_READY=False,online_SLO=False)
 path=output/'entry-report.json'
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
 def deadline_check(phase,minimum=0):
  if time.time()+minimum>=cfg['gpu_start_epoch']+5280:raise TimeoutError('Original Native deadline: '+phase)
 def canceled():return (output/'CANCEL').exists()
 save()
 try:
  assert report['node']==NODE and len(os.sched_getaffinity(0))>=32
  assert os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'
  assert os.environ['INFINILM_DEEP_MOE_TAIL512_R7211']==os.environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']=='0'
  assert not any(os.environ.get(k)for k in ['BASH_ENV','PYTHONSTARTUP','PYTHONINSPECT','INFINILM_DEEP_LAYER_TRACE_R4026','INFINILM_DEEP_POSITION_PROOF_R8726'])
  from fixed_cohort.entry import prepare
  prepared=prepare(PACKAGE/'CPU_config.json')
  report['input_descriptors']=[dict(row=r['row'],case_id=r['case_id'],ordinal=r['ordinal'],prompt_tokens=len(r['prompt_token_ids']),prompt_IDs_SHA256=r['prompt_IDs_SHA256'],body_SHA256=r['body_SHA256'],budget=r['max_new_tokens'])for r in prepared['records']]
  with(output/'four-inputs-private.json').open('x')as f:json.dump(prepared['records'],f,ensure_ascii=False);f.write('\n')
  report['inputs_SHA256']=sha(output/'four-inputs-private.json');report['phase']='Torch_Common_prepare';save();deadline_check(report['phase'],120)
  import torch
  torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
  report['GPU_properties']=[str(torch.cuda.get_device_properties(i))for i in range(8)]
  report['GPU_visible_count']=8;report['entry_owner_mask']=sorted(os.sched_getaffinity(0))
  # Exact same public/Common processor initialization as the already successful
  # current diagnostic entry; only decoding is used, frozen inputs are not encoded.
  public=Path(cfg['public_package_root'])/'public-python'
  sys.path[:0]=[str(public),cfg['Common_python_root']]
  import infinilm,_infinilm
  assert Path(infinilm.__file__).resolve()==(public/'infinilm/__init__.py').resolve()
  assert Path(_infinilm.__file__).resolve()==Path(cfg['Common_extension']).resolve()and sha(_infinilm.__file__)==cfg['Common_extension_SHA256']
  tokenpin=json.loads((D/'TOKENIZER_PINS.json').read_bytes())
  for name,h in tokenpin['tokenizer_assets_SHA256'].items():assert sha(Path(tokenpin['processor_path'])/name)==h
  from infinilm.processors import AutoInfinilmProcessor
  processor=AutoInfinilmProcessor.from_pretrained(tokenpin['processor_path']);tokenizer=processor.get_tokenizer()
  from safetensors import safe_open
  from fixed_cohort.checkpoint_loader import bind_library
  binding=bind_library(prepared['receipt']['library'],cfg['Native_SHA256'])
  from session import run_session
  report['phase']='Native_create_load';save();deadline_check(report['phase'],120)
  run_session(prepared,cfg,binding=binding,torch=torch,safe_open=safe_open,tokenizer=tokenizer,
    output=output,report=report,save=save,deadline_check=deadline_check,canceled=canceled)
  assert report['engine_worker_shutdown_completed']and report['normal_Native_shutdown']and report['observed_all8_threads_exited']
  assert report['all_cohorts_completed_normal']and report['completed_cohorts']==6 and report['logical_requests']==24
  report.update(completed=True,passed=True,phase='all_six_cohorts_normal_closed',Model_READY=False);save()
 except BaseException as error:
  report.update(completed=True,passed=False,infrastructure_error=repr(error),traceback=traceback.format_exc(),phase='FAILED_NO_RETRY');save();raise
 return report

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--source-only',action='store_true');a=p.parse_args(argv)
 digest=os.environ['DEEP458_C4_CONTROLLER_SHA'];m=verify(digest,require_ready=not a.source_only)
 if a.source_only:
  print(json.dumps(dict(source_verified=True,GPU=0,Native_dlopen=False,Model_READY=False)));return 0
 cfg=json.loads((D/'config.json').read_bytes());epoch=float(os.environ['DEEP458_C4_START'])
 cfg.update(gpu_start_epoch=epoch,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],deployment_manifest_SHA256=digest)
 assert cfg['actual_step'].startswith('910374.')and time.time()+120<epoch+5280
 publish(writable_results()/('Deep458-fixed-cohort-'+RUN+'-actual-config.json'),cfg)
 execute(cfg,m);return 0

if __name__=='__main__':raise SystemExit(main())
