"""Run remotely. Assemble real ordinary-AR taps across prefill+decode calls."""
import pathlib
LAYERS=(19,37,66,78,90)
def assemble_sequence(stage_packets,begin=0,end=512):
 import torch
 if not 0<=begin<end<=1024:raise ValueError('bounded target positions')
 by_layer={layer:{}for layer in LAYERS}
 for packet in stage_packets:
  for record in packet['records']:
   if record['point']!='decoder' or record['layer']not in by_layer:continue
   shape=record['shape'];positions=record['positions'];raw=pathlib.Path(record['remote_owned_buffer']).read_bytes()
   if shape!=[1,len(positions),7168]or len(raw)!=len(positions)*7168*2:raise ValueError('tap bytes/shape')
   values=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(shape)
   for index,position in enumerate(positions):
    if begin<=position<end:
     row=values[0,index].clone();old=by_layer[record['layer']].get(position)
     if old is not None and not torch.equal(old,row):raise ValueError('duplicate position differs within ordinary arm')
     by_layer[record['layer']][position]=row
 expected=set(range(begin,end))
 if any(set(rows)!=expected for rows in by_layer.values()):raise ValueError('missing real target positions/stage')
 return torch.cat([torch.stack([by_layer[l][p]for p in range(begin,end)]).unsqueeze(0)for l in LAYERS],dim=-1),list(range(begin,end))
