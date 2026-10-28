"""Actual assigned-visible-GPU protocol import, before constructing/loading model."""
import hashlib,json,pathlib,sys
def validate():
 r=pathlib.Path('/refactor');d=r/'incoming/GLM-regular-service-r204';b=r/'candidates/GLM-regular-service-r204';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 p=b/'CPU-prepare.json';cp=json.loads(p.read_bytes());assert cp['completed']and cp['CPU_only']and cp['static_only']and not cp['GPU_initialized'];assert cp['source_manifest_SHA256']==sha(d/'source-manifest-r204.json')
 for name,h in cp['overlay_files_SHA256'].items():assert sha(name)==h,name
 import torch
 assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 from infinilm.server.inference_server import InferenceServer
 from infinilm.llm import FinishReason
 actual_protocol_path=pathlib.Path(sys.modules[InferenceServer.__module__].__file__).resolve();assert actual_protocol_path.is_relative_to(b),("private protocol module",str(actual_protocol_path),str(b))
 assert FinishReason.EOS_TOKEN.value=='eos_token'and FinishReason.LENGTH.value=='length'
 return {'actual_protocol_imported':True,'CPU_receipt_SHA256':sha(p),'source_manifest_SHA256':cp['source_manifest_SHA256'],'protocol_server_path':sys.modules[InferenceServer.__module__].__file__,'weights_loaded':False,'native_model_constructed':False}
