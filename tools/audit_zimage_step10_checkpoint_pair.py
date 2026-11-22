from hashlib import sha256
import json,os,socket
from pathlib import Path
import torch
from gencuda import hy_resume_checkpoint as base,zimage_training_resume_checkpoint as z
assert socket.gethostname().split(".")[0]=="f13r2n07" and os.environ["SLURM_JOB_ID"]=="1103606"
assert not torch.cuda.is_initialized()
assert torch.cuda.device_count()==0, "Mask not enforced"
for m,h in ((base,"aeb21195bb7bfb64f36ea4299e52b359a695d986d4556a70f0b1c5c3915de8d7"),(z,"8a394d733970ed558ca0a3c85f27dbbf955849ce6fb8746d35abb9c0d2d6a5c5")):
 src=Path(m.__file__).resolve()
 assert str(src).startswith("/common-framework/generation/src/gencuda/") and sha256(src.read_bytes()).hexdigest()==h
def read(path,digest=None):
 path=Path(path)
 size=path.stat().st_size
 assert size==60424609
 h=sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
 observed=h.hexdigest()
 assert digest is None or observed==digest
 p=z.load_checkpoint(path)
 assert p["metadata"]["global_step"]==10
 assert p["integrity_sha256"]==base.tree_sha256({key:value for key,value in p.items() if key!="integrity_sha256"})
 assert p["adapter_sha256"]==base.tree_sha256(p["state"]["trainable"])
 return p,{"bytes":size,"SHA256":observed,"integrity_SHA256":p["integrity_sha256"],"adapter_SHA256":p["adapter_sha256"]}
a,da=read("/prior/checkpoint-10.private.pt","28e3f585b84531cbac5494b4b7a1bacb1bca9c3f5fb2098eab23ec35577d4bbb")
b,db=read("/resident/checkpoint-10.private.pt")
def sections(x,y):
 assert set(x)==set(y)
 return {k:base.tree_sha256(v)==base.tree_sha256(y[k]) for k,v in sorted(x.items())}
def changed(x,y):
 assert set(x)==set(y)
 names=[key for key,v in x.items() if base.tree_sha256(v)!=base.tree_sha256(y[key])]
 return len(names)
result=dict(reader_step="1103606."+os.environ["SLURM_STEP_ID"],original=da,native_resident=db,
 top_level_match=sections(a,b),state_match=sections(a["state"],b["state"]),rng_match=sections(a["rng"],b["rng"]),
 trainable_tensors=len(a["state"]["trainable"]),changed_trainable_tensor_count=changed(a["state"]["trainable"],b["state"]["trainable"]),
 ready=False,exported_tensor_bytes=0,torch_cuda_initialized=torch.cuda.is_initialized())
print(json.dumps(result,allow_nan=False))
