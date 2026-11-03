"""One existing official8489 fixture callback only; no129-suite reencoding or Model."""
from pathlib import Path
import os,sys,types,importlib.machinery,json,hashlib

def run(output):
 root=Path('/refactor');owner=root/'incoming/Deep-paired-quality-r375/owner';sys.path.insert(0,str(owner))
 assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 from source_pin_r281 import verify
 verify('1b93c3db22ab59c6768e401dc46f71b630a768934fe316135b53d780d1b3fafb')
 # Exact qualified375 CPU encoder isolation pattern, reduced to one original
 # fixture. No Native root, AsyncLLMEngine, payload file or CUDA initialization.
 U=root/'candidates/latest-family-service-r270/Deep/python/infinilm'
 class ForbiddenNative(types.ModuleType):
  def __getattr__(self,name):raise RuntimeError('Native API forbidden in CPU encoding: '+name)
 package=types.ModuleType('infinilm');package.__path__=[str(U)];package.__package__='infinilm';sys.modules['infinilm']=package
 core=ForbiddenNative('infinilm._core');core.__file__='FORBIDDEN_NATIVE';core.__path__=[];core.__spec__=importlib.machinery.ModuleSpec(core.__name__,None);package._core=core;sys.modules[core.__name__]=core
 llm=types.ModuleType('infinilm.llm');llm.__path__=[str(U/'llm')];llm.__package__='infinilm.llm';sys.modules[llm.__name__]=llm
 for suffix,name in [('scheduler','SchedulerOutput'),('static_scheduler','StaticSchedulerOutput')]:
  mod=types.ModuleType('infinilm.llm.'+suffix);setattr(mod,name,type(name,(),{}));sys.modules[mod.__name__]=mod
 class ForbiddenEngine:
  def __init__(self,*a,**kw):raise RuntimeError('Model construction forbidden in CPU encoding')
 llm.AsyncLLMEngine=ForbiddenEngine
 def audit(event,args):
  if event=='open'and isinstance(args[0],(str,bytes,os.PathLike))and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weight/model payload forbidden')
 sys.addaudithook(audit)
 import torch
 torch.set_num_threads(1)
 def forbidden(*a,**kw):raise RuntimeError('GPU API forbidden in CPU encoding')
 for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,forbidden)
 torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
 from infinilm.llm.sampling_params import SamplingParams
 from infinilm.llm.request import FinishReason
 llm.SamplingParams=SamplingParams;llm.FinishReason=FinishReason
 from prepared_family_entry_r268 import prepare_app
 prepared=prepare_app('Deep','/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8',processor_path=root/'results/latest-three-processors-r260/Deep-model-metadata')
 assert prepared['identity']['Native_loaded_by_factory']is False and prepared['identity']['weights_opened_by_factory']is False
 from contract import load_plan
 from runtime_leaf import encode_once
 plan=load_plan(Path(__file__).resolve().parent/'BENCHMARK_PLAN.json');output=Path(output);(output/'results').mkdir(parents=True)
 provider=dict(prepared=prepared);encode_once(provider,plan,output,'cpu-fixture','CPU');evidence={name:{k:v for k,v in value.items()if k not in ['ids','fixture']}for name,value in provider['PD_inputs'].items()}
 assert not prepared['engine']._requests;prepared['engine'].stop()
 return dict(completed=True,passed=True,GPU=0,Native_dlopen=False,weights_opened=False,actual_prepared_encoder=True,service_requests_enqueued=0,encodes=2,identity=prepared['identity'],fixtures=evidence)
if __name__=='__main__':print(json.dumps(run(sys.argv[1])))
