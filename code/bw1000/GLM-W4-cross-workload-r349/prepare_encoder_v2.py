"""Retain CPU670 source; add the proven CPU272 guarded import boundary."""
import pathlib, hashlib
D=pathlib.Path(__file__).parent
s=(D/'CPU_encode_fixtures.py').read_text(encoding='utf-8')
s=s.replace("import pathlib,json,hashlib,copy,os,sys,time", "import pathlib,json,hashlib,copy,os,sys,time,types,importlib.machinery")
s=s.replace("report={'completed':False", "report={'root_core_llm_guarded':True,'actual_root_or_ABI_qualified':False,'optional_GPU_import_isolated':True,'CPU670_preserved':True,'completed':False")
guard='''
 U=R/'candidates/latest-family-service-r270/GLM/python/infinilm'
 class ForbiddenNative(types.ModuleType):
  def __getattr__(self,name):raise RuntimeError('InfiniLM Native API forbidden: '+name)
 root=types.ModuleType('infinilm');root.__path__=[str(U)];root.__package__='infinilm';sys.modules['infinilm']=root
 core=ForbiddenNative('infinilm._core');core.__file__='FORBIDDEN_NATIVE';core.__path__=[];core.__spec__=importlib.machinery.ModuleSpec(core.__name__,None);root._core=core;sys.modules[core.__name__]=core
 llm=types.ModuleType('infinilm.llm');llm.__path__=[str(U/'llm')];llm.__package__='infinilm.llm';sys.modules[llm.__name__]=llm
 for suffix,cls in [('scheduler','SchedulerOutput'),('static_scheduler','StaticSchedulerOutput')]:
  mod=types.ModuleType('infinilm.llm.'+suffix);setattr(mod,cls,type(cls,(),{}));sys.modules[mod.__name__]=mod
 class ForbiddenEngine:
  def __init__(self,*a,**k):raise RuntimeError('Async Model engine construction forbidden')
 llm.AsyncLLMEngine=ForbiddenEngine
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.fsdecode(args[0]).lower().endswith(('.safetensors','.bin','.pt','.pth')):raise RuntimeError('Weight/model-binary read forbidden')
 sys.addaudithook(audit)
 import torch
 torch.set_num_threads(1)
 def forbidden(*a,**k):raise RuntimeError('GPU API forbidden in CPU encoding')
 for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,forbidden)
 torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
 from infinilm.llm.sampling_params import SamplingParams
 from infinilm.llm.request import FinishReason
 llm.SamplingParams=SamplingParams;llm.FinishReason=FinishReason
'''
s=s.replace('try:\n from prepared_family_entry_r268', 'try:\n'+guard+' from prepared_family_entry_r268')
s=s.replace("prepare_agent_request(copy.deepcopy(body),server.tool_call_parser)", "prepare_agent_request(copy.deepcopy(body),'glm5')\n  data.update(_agent_schemas=schemas,_agent_choice=choice)")
s=s.replace("'GLM349-official-encoding-'+step", "'GLM349-official-encoding-v2-'+step")
s=s.replace("  ids=list(request.prompt_token_ids);", "  assert engine.take(0) is request\n  ids=list(request.prompt_token_ids);")
s=s.replace("  report['fixtures'].append", "  engine.record_started(request);engine.record_finished(request,0,True)\n  report['fixtures'].append")
out=D/'CPU_encode_fixtures_v2.py';out.write_bytes(s.encode());compile(s,str(out),'exec')
shell=(D/'CPU_encode_inside.sh').read_bytes().replace(b'encoder_cpu/',b'encoder_cpu_v2/').replace(b'CPU_encode_fixtures.py',b'CPU_encode_fixtures_v2.py')
assert b'\r' not in shell
(D/'CPU_encode_inside_v2.sh').write_bytes(shell)
upload=(D/'prepare_encoder_upload.py').read_text().replace("'CPU_encode_fixtures.py','CPU_encode_inside.sh'", "'CPU_encode_fixtures_v2.py','CPU_encode_inside_v2.sh'").replace('encoder_cpu','encoder_cpu_v2').replace('CPU_encode_inside.sh','CPU_encode_inside_v2.sh').replace('GLM349-official-encoder','GLM349-official-encoder-v2').replace('GLM349-encoder','GLM349-encoder-v2').replace('remote_upload_encoder.py','remote_upload_encoder_v2.py')
(D/'prepare_encoder_upload_v2.py').write_bytes(upload.encode())
print(hashlib.sha256(out.read_bytes()).hexdigest())
