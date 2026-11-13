"""Run on the original receiver host with Python 3.12. Emits receipt metadata only."""
import hashlib,json,pathlib,time
root=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
items=[
 ('vae','training-hunyuanvideo-public-vae-v1','vae.incomplete/diffusion_pytorch_model.safetensors','vae/diffusion_pytorch_model.safetensors','5332883ea359448f9e675a9abd1e6991'),
 ('transformer','training-hunyuanvideo-original-model-v1','model.incomplete/transformer/480p_t2v/diffusion_pytorch_model.safetensors.partial','hunyuanvideo-1.5/transformer/480p_t2v/diffusion_pytorch_model.safetensors','0a610475811d4237955e139ab9dda71c')]
out={}
for name,base,part,final,request in items:
 p=root/base
 def receipt(name):
  path=p/'retained-suffix-v2'/name
  if not path.is_file():return None
  with path.open('rb') as stream:raw=stream.read(16385)
  assert len(raw)<=16384
  v=json.loads(raw)
  assert v['request_id']==request and v['claim']=='e124758359448cf6ff5d05ed4141783b7a745bf8'
  return dict(receipt_SHA256=hashlib.sha256(raw).hexdigest(),**{k:v[k] for k in ('request_id','claim','deadline','passed','offset','error_type','whole_SHA256','prefix_SHA256') if k in v})
 def size(name):
  path=p/name
  return path.stat().st_size if path.is_file() else None
 out[name]=dict(intent=receipt('intent.json'),progress=receipt('progress.json'),
  outcome=receipt('outcome.json'),last_fsynced=receipt('last-fsynced.json'),partial_bytes=size(part),final_bytes=size(final))
print(json.dumps(dict(observed_utc_epoch=time.time(),assets=out)))
