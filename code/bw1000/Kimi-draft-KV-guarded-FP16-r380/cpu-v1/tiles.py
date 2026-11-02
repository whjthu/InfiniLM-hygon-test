"""Read only four exact BF16 KV payloads; never create a draft/head/model."""
import pathlib,json,hashlib,math,numpy as np
R=pathlib.Path('/refactor');CHECKPOINT=R/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors'
HEADER_SHA='d250fe199d420a34c0bba811511d6e0ee79f11c95ad8f9482cd014ce9180fb24'
FULL_SHA_INHERITED='1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2'
SPECS={
 'layers.0.self_attn.kv_a_proj_with_mqa.weight':dict(dtype='BF16',shape=[576,7168],data_offsets=[3702156288,3710413824]),
 'layers.0.self_attn.kv_b_proj.weight':dict(dtype='BF16',shape=[16384,512],data_offsets=[3710413824,3727191040]),
 'layers.4.self_attn.kv_a_proj_with_mqa.weight':dict(dtype='BF16',shape=[576,7168],data_offsets=[7183494144,7191751680]),
 'layers.4.self_attn.kv_b_proj.weight':dict(dtype='BF16',shape=[16384,512],data_offsets=[7191751680,7208528896])}
def load():
 before=CHECKPOINT.stat();assert before.st_size==7385765464;arrays={};proof=[]
 with CHECKPOINT.open('rb')as f:
  n=int.from_bytes(f.read(8),'little');assert 2<=n<=1<<20;raw=f.read(n);assert hashlib.sha256(raw).hexdigest()==HEADER_SHA;h=json.loads(raw);payload=n+8
  assert len([k for k in h if k!='__metadata__'])==87
  for key,spec in SPECS.items():
   assert h[key]==spec;a,b=spec['data_offsets'];assert b-a==2*math.prod(spec['shape']);f.seek(payload+a);raw=f.read(b-a);assert len(raw)==b-a;arrays[key]=np.frombuffer(raw,dtype='<u2').reshape(spec['shape']).copy();proof.append(dict(key=key,**spec,payload_start=payload+a,payload_end=payload+b,payload_SHA256=hashlib.sha256(raw).hexdigest()))
 after=CHECKPOINT.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
 return arrays,dict(checkpoint=str(CHECKPOINT),file_bytes=before.st_size,file_mtime_ns=before.st_mtime_ns,file_inode=before.st_ino,header_SHA256=HEADER_SHA,expected_full_SHA256_inherited=FULL_SHA_INHERITED,fresh_full_file_hash=False,fresh_selected_payload_hashes=True,only_four_selected_tensors=True,all_other83_payloads_unread=True,tensors=proof)
