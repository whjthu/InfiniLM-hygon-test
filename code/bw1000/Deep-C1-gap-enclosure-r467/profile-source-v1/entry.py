"""Source-pinned current a314 rows1 owner entry. No HTTP or alternate Native."""
from pathlib import Path,PurePosixPath
import argparse,ctypes as C,hashlib,importlib,json,os,socket,sys,time,traceback
from profile_control import run_loaded,CapiTimer
from precision import select_original,verify_provider
from forward_policy0 import make_forward
NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
IDS_FILE_SHA='1717324f35a2f7e987928e4a87b7c878a8115f0910f3af56fe4e611cd62c769f'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree(root,pin):
 root=Path(root).resolve();assert H(root/'SOURCE_MANIFEST.json')==pin;m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():p=root/n;assert p.resolve().is_relative_to(root)and H(p)==h,n
 return m
def schema(c):
 assert(c['TP'],c['PP'],c['rows'],c['context'],c['HC_mode'],c['fullpre'],c['r366'],c['precision_policy'])==(8,1,1,32768,0,0,0,0)
 assert c['Native_SHA256']==NATIVE and c['HTTP']==0 and c['requests']==5 and c['observer_schedule']==[False,False,False,True,False]and c['samples']==[8,87]
 assert(c['Native_seconds'],c['service_seconds'],c['GPU_seconds'],c['keeper_seconds'])==(2580,2400,2700,3000)
 for k in['parent_root','base_provider_root','Native_library','input_path','output_root','model_path','Common_python_root','Common_extension','public_package_root']:
  assert type(c[k])is str and '\\'not in c[k]and PurePosixPath(c[k]).is_absolute(),k
 assert c['input_file_SHA256']==IDS_FILE_SHA and type(c['execute_authorized'])is bool
 return c

def execute(c):
 schema(c);assert c['execute_authorized']is True and socket.gethostname().split('.')[0]==c['node']and os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']==c['actual_step']
 assert H(c['input_path'])==IDS_FILE_SHA;prompt=json.loads(Path(c['input_path']).read_bytes())['prompt_token_ids']
 parent=Path(c['parent_root']);m=tree(parent,c['parent_manifest_SHA256'])
 for path,h in m['parents_manifest_SHA256'].items():tree(path,h)
 for p in[c['Common_python_root'],str(Path(c['public_package_root'])/'public-python'),c['base_provider_root'],str(parent)]:sys.path.insert(0,p)
 worker=importlib.import_module('public_deep_model.worker');assert Path(worker.__file__).resolve()==(parent/'public_deep_model/worker.py').resolve();worker.configure_process(c)
 from public_deep_model.native_context import NativeContext
 from public_deep_model.native_decode import generate
 from public_deep_model.fullpre_control import QuiescentMode
 from enum import Enum
 class Reasons(Enum):EOS_TOKEN='EOS';LENGTH='cap';CANCELED='canceled'
 out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);report=dict(completed=False,passed=False,phase='parent_binding',identity={k:c[k]for k in['run','actual_step','gpu_start_epoch','Native_SHA256']},HTTP=0,Model_READY=False,normal_Native_shutdown=False)
 def save():
  p=out/'entry-report.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');t.replace(p)
 def deadline_check(where,minimum=0):
  if time.time()+minimum>=c['gpu_start_epoch']+min(c['Native_seconds'],c['service_seconds']):raise TimeoutError('Original profile epoch: '+where)
 save()
 try:
  _,receipt,_=worker.verify_native_chain();assert receipt['library_SHA256']==NATIVE;torch=worker.torch_setup();safe_open=worker.safe_open_dependency();cfg,mapping=worker.read_model_metadata(c['model_path']);binding=worker.bind_library(c['Native_library'],NATIVE)
  assert H(c['Common_extension'])==c['Common_extension_SHA256'];report['precision_policy']=select_original(binding.lib,receipt['library'])
  from existing_profile_consumer import bind_inner
  bind_inner(binding.lib);report['profile_actual_providers']={n:verify_provider(getattr(binding.lib,n),receipt['library'])for n in ['deep_owner_arm_r8900','deep_owner_read_r8900','deep_attention_inner_arm_r9193','deep_attention_inner_read_r9193']};save()
  with NativeContext(binding,receipt,config=c,cfg=cfg,mapping=mapping,torch=torch,safe_open=safe_open,report=report,deadline_check=deadline_check)as context:
   # No global/module mutation. Original rows1 constructor/load unchanged;
   # before its first forward use the existing explicitly checked mode0 caller.
   timer=CapiTimer(binding.lib);report['actual_single_CAPI_calls']=timer.rows;context.forward=make_forward(timer,context.ctx,cfg,report,binding.check);control=QuiescentMode(context);report['phase']='five_continuations';save()
   import numpy as np
   def finite(logits):
    assert logits.dtype==np.uint16 and logits.shape==(1,cfg['vocab_size']);assert not np.any((logits&0x7f80)==0x7f80),'Nonfinite original BF16 logits'
   run_loaded(context,prompt,out/'scalar-events',report,save,generate=generate,reasons=Reasons,control=control,finite=finite,deadline=c['gpu_start_epoch']+c['service_seconds']);report['actual_single_CAPI_calls']=timer.rows;report['phase']='normal_context_exit';save()
  report.update(completed=True,passed=True,phase='normal_closed');save()
 except BaseException as e:
  report.update(completed=True,passed=False,error=repr(e),traceback=traceback.format_exc());save();raise
 return report

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
 tree(Path(__file__).parent,a.source_sha);assert H(a.config)==a.config_sha;c=schema(json.loads(Path(a.config).read_bytes()))
 if not a.execute:print(json.dumps(dict(source_schema=True,Native_dlopen=False,Runtime=False,weights=False,GPU=False,Model_READY=False)));return
 return execute(c)
if __name__=='__main__':main()
