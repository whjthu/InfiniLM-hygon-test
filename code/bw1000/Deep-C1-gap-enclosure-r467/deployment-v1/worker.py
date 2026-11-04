"""Actual same-package nonserve CLI then five-continuation current Native entry."""
from pathlib import Path
import argparse,json,os,socket,sys,time,traceback
from support import D,PACKAGE,RUN,LABEL,NODE,verify,sha,publish

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--source-only',action='store_true');a=p.parse_args(argv)
 digest=os.environ['DEEP467_PROFILE_DEPLOY_SHA'];m=verify(digest,require_ready=not a.source_only)
 if a.source_only:print(json.dumps(dict(source_verified=True,execution_READY=m['execution_READY'],Native_dlopen=False,GPU=False,Model_READY=False)));return 0
 epoch=float(os.environ['DEEP467_PROFILE_GPUSTART']);step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert step.startswith('910374.')and socket.gethostname().split('.')[0]==NODE
 assert time.time()+120<epoch+2580 and len(os.sched_getaffinity(0))>=32 and os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'
 W=Path('/refactor/results');process=W/(LABEL+'-'+RUN+'-GPU_PROCESS.json')
 identity=dict(PID=os.getpid(),PGID=os.getpgrp(),UID=os.getuid(),GID=os.getgid(),startticks=Path('/proc/self/stat').read_text().split()[21],node=NODE,actual_step=step,gpu_start_epoch=epoch,deployment_manifest_SHA256=digest,entry_source_manifest_SHA256=m['entry_source_manifest_SHA256'],HTTP=0)
 publish(process,dict(**identity,phase='same_package_nonserve_CLI'))
 try:
  c=json.loads((D/'config.json').read_bytes());cfg_path=W/(LABEL+'-'+RUN+'-actual-config.json')
  # First use the original unarmed/epoch-null schema with actual CLI args.
  sys.path.insert(0,str(PACKAGE));import entry
  assert Path(entry.__file__).resolve()==(PACKAGE/'entry.py').resolve()
  entry.main(['--config',str(D/'config.json'),'--config-sha',sha(D/'config.json'),'--source-sha',m['entry_source_manifest_SHA256']])
  import torch
  torch.set_num_threads(1);assert torch.cuda.device_count()==8;props=[]
  for i in range(8):
   q=torch.cuda.get_device_properties(i);arch=str(getattr(q,'gcnArchName',''));assert arch.split(':')[0]=='gfx936'and q.multi_processor_count==80 and q.total_memory>=63*1024**3
   props.append(dict(device=i,arch=arch,compute_units=q.multi_processor_count,total_memory=q.total_memory,name=q.name))
  publish(W/(LABEL+'-'+RUN+'-GPU_PROPERTIES.json'),dict(**identity,devices=props))
  c.update(execute_authorized=True,gpu_start_epoch=epoch,actual_step=step);publish(cfg_path,c)
  q=entry.main(['--config',str(cfg_path),'--config-sha',sha(cfg_path),'--source-sha',m['entry_source_manifest_SHA256'],'--execute'])
  assert q['completed']and q['passed']and q['normal_Native_shutdown']and q['engine_worker_shutdown_completed']and q['observed_all8_threads_exited']
  publish(W/(LABEL+'-'+RUN+'-GPU_PROCESS_CLOSED.json'),dict(**identity,normal=True,entry_report_SHA256=sha(Path(c['output_root'])/'entry-report.json'),requests=len(q['requests']),profile_coverage_complete=q['profile_coverage_complete'],ended_unix=time.time()))
 except BaseException as error:
  publish(W/(LABEL+'-'+RUN+'-GPU_PROCESS_FAILED.json'),dict(**identity,error=repr(error),traceback=traceback.format_exc(),ended_unix=time.time()));raise
 return 0
if __name__=='__main__':raise SystemExit(main())
